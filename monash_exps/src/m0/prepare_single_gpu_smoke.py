#!/usr/bin/env python3
"""Resolve the M0 smoke config and create the deterministic initial LoRA G0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from bhaskera.config import load_config
from bhaskera.models import build_model
from peft import get_peft_model_state_dict

SCHEMA_VERSION = 1
WORKFLOW_VERSION = "m0-olmo2-7b-g0-v1"
SEED = 20260820


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def normalize_key(name: str) -> str:
    name = name.removeprefix("module.")
    return re.sub(r"(lora_[AB])\.[^.]+\.(weight)", r"\1.\2", name)


def normalize_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value) or "lora_" not in name:
            continue
        key = normalize_key(name)
        if key in normalized:
            raise RuntimeError(f"duplicate normalized LoRA key: {key}")
        normalized[key] = value.detach().cpu().float().contiguous()
    return normalized


def tensor_content_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_summary(
    state: dict[str, torch.Tensor], *, expected_layers: int
) -> dict[str, Any]:
    if not state:
        raise RuntimeError("G0 contains no LoRA tensors")
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise RuntimeError("G0 contains a non-finite tensor")

    target_counts = {"q_proj": 0, "v_proj": 0}
    a_count = 0
    b_count = 0
    for name, tensor in state.items():
        targets = [target for target in target_counts if f".{target}." in name]
        if len(targets) != 1:
            raise RuntimeError(f"unexpected G0 target tensor: {name}")
        target_counts[targets[0]] += 1
        if ".lora_A.weight" in name:
            a_count += 1
            if not torch.count_nonzero(tensor):
                raise RuntimeError(f"LoRA A tensor is unexpectedly all-zero: {name}")
        elif ".lora_B.weight" in name:
            b_count += 1
            if torch.count_nonzero(tensor):
                raise RuntimeError(f"LoRA B tensor is not zero at G0: {name}")
        else:
            raise RuntimeError(f"unexpected LoRA tensor name: {name}")

    expected_per_target = expected_layers * 2
    if target_counts != {
        "q_proj": expected_per_target,
        "v_proj": expected_per_target,
    }:
        raise RuntimeError(
            f"unexpected target tensor counts: {target_counts}; "
            f"expected {expected_per_target} per target"
        )
    if a_count != expected_layers * 2 or b_count != expected_layers * 2:
        raise RuntimeError(f"unexpected A/B counts: A={a_count}, B={b_count}")

    return {
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "float32_bytes": sum(tensor.numel() * 4 for tensor in state.values()),
        "tensor_content_sha256": tensor_content_sha256(state),
        "target_tensor_counts": target_counts,
        "lora_a_tensors": a_count,
        "lora_b_tensors": b_count,
        "schema": [
            {
                "name": name,
                "shape": list(state[name].shape),
                "dtype": str(state[name].dtype),
            }
            for name in sorted(state)
        ],
    }


def resolve_path(value: str, workspace_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def validate_template(raw: dict[str, Any]) -> None:
    expected = {
        "model.dtype": "bfloat16",
        "model.attn_impl": "flash_attention_2",
        "model.use_liger_kernel": True,
        "data.seq_len": 1024,
        "lora.r": 16,
        "lora.alpha": 64,
        "lora.dropout": 0.03,
        "lora.target_modules": ["q_proj", "v_proj"],
        "training.batch_size": 2,
        "training.grad_accum": 4,
        "training.lr": 1.0e-4,
        "training.max_steps": 12,
        "training.seed": SEED,
        "training.distributed.strategy": "ddp",
    }

    def nested(key: str) -> Any:
        value: Any = raw
        for component in key.split("."):
            value = value[component]
        return value

    actual = {key: nested(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"smoke template mismatch: actual={actual}, expected={expected}")


def validate_token_cache(
    raw: dict[str, Any], workspace_root: Path, token_manifest_path: Path
) -> None:
    master = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    if master.get("seq_len") != 1024 or len(master.get("views", {})) != 7:
        raise RuntimeError("formal tokenized-view manifest is incomplete")
    view = master["views"]["australia_nz"]
    cache = view["cache"]
    configured = resolve_path(raw["data"]["tokenized_path"], workspace_root).resolve()
    manifested = resolve_path(cache["cache_path"], workspace_root).resolve()
    if configured != manifested or not configured.is_dir():
        raise RuntimeError(
            f"smoke cache mismatch: configured={configured}, manifested={manifested}"
        )
    if cache["rows"] != 9337 or cache["seq_len"] != 1024:
        raise RuntimeError(f"unexpected Australia/NZ cache metadata: {cache}")
    for item in cache["parquet"]:
        parquet = configured / item["path"]
        if not parquet.is_file() or sha256_file(parquet) != item["sha256"]:
            raise RuntimeError(f"token cache hash mismatch: {parquet}")


def git_revision(slakshna_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(slakshna_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_spec(raw: dict[str, Any], revision: str) -> dict[str, Any]:
    return {
        "model": {
            key: raw["model"][key]
            for key in (
                "name",
                "trust_remote_code",
                "dtype",
                "attn_impl",
                "use_liger_kernel",
                "quantization",
            )
        },
        "lora": {
            key: raw["lora"][key]
            for key in (
                "r",
                "alpha",
                "dropout",
                "target_modules",
                "include_experts",
                "freeze_router",
                "modules_to_save",
            )
        },
        "seed": SEED,
        "slakshna_revision": revision,
    }


def validate_existing_g0(
    *, g0_path: Path, manifest_path: Path, spec: dict[str, Any], expected_layers: int
) -> dict[str, Any] | None:
    if not g0_path.exists() and not manifest_path.exists():
        return None
    if not g0_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("incomplete G0 artifact; preserve it and choose a new G0 path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("spec") != spec:
        raise RuntimeError("existing G0 was built from a different model/LoRA specification")
    if sha256_file(g0_path) != manifest.get("state_file_sha256"):
        raise RuntimeError("existing G0 file hash mismatch")
    loaded = torch.load(g0_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("existing G0 is not a tensor dictionary")
    summary = state_summary(normalize_state(loaded), expected_layers=expected_layers)
    if summary != manifest.get("state"):
        raise RuntimeError("existing G0 tensor content/schema mismatch")
    return manifest


def create_g0(
    *, cfg: Any, g0_path: Path, manifest_path: Path, spec: dict[str, Any]
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model, profile = build_model(cfg, torch.device("cpu"))
    expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if expected_layers <= 0 or profile.is_moe:
        raise RuntimeError(
            f"unexpected OLMo 2 profile: layers={expected_layers}, moe={profile.is_moe}"
        )
    state = normalize_state(get_peft_model_state_dict(model))
    summary = state_summary(state, expected_layers=expected_layers)

    g0_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{g0_path.name}.", suffix=".tmp", dir=g0_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, g0_path)
        g0_path.chmod(0o600)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": spec,
        "model_layers": expected_layers,
        "state_file": str(g0_path),
        "state_file_bytes": g0_path.stat().st_size,
        "state_file_sha256": sha256_file(g0_path),
        "state": summary,
    }
    atomic_json(manifest, manifest_path)
    del model
    return manifest


def main() -> int:
    inferred_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g0-manifest", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    template_path = args.template.resolve()
    run_root = args.run_root.resolve()
    resolved_path = args.resolved_config.resolve()
    g0_path = args.g0.resolve()
    g0_manifest_path = args.g0_manifest.resolve()
    slakshna_root = workspace_root / "Slakshna"
    token_manifest_path = (
        workspace_root
        / "monash_exps/.runtime/manifests/m0/tokenized-formal-views.json"
    )
    stack_path = workspace_root / "monash_exps/.runtime/manifests/m0/stack.json"

    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    validate_template(raw)
    validate_token_cache(raw, workspace_root, token_manifest_path)
    revision = git_revision(slakshna_root)
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    if revision != stack.get("slakshna_revision"):
        raise RuntimeError(
            f"Slakshna revision {revision} does not match M0 stack manifest "
            f"{stack.get('slakshna_revision')}"
        )

    resolved = json.loads(json.dumps(raw))
    resolved["model"]["name"] = str(
        resolve_path(raw["model"]["name"], workspace_root).resolve()
    )
    resolved["data"]["tokenized_path"] = str(
        resolve_path(raw["data"]["tokenized_path"], workspace_root).resolve()
    )
    resolved["lora"]["resume_path"] = str(g0_path)
    resolved["checkpoint"]["save_dir"] = str(run_root / "checkpoints")
    atomic_yaml(resolved, resolved_path)
    cfg = load_config(str(resolved_path))

    model_config = json.loads(
        (Path(cfg.model.name) / "config.json").read_text(encoding="utf-8")
    )
    expected_layers = int(model_config["num_hidden_layers"])
    spec = build_spec(raw, revision)
    g0_manifest = validate_existing_g0(
        g0_path=g0_path,
        manifest_path=g0_manifest_path,
        spec=spec,
        expected_layers=expected_layers,
    )
    if g0_manifest is None:
        print("Creating deterministic OLMo 2 7B LoRA G0; this loads the base model once.")
        g0_manifest = create_g0(
            cfg=cfg,
            g0_path=g0_path,
            manifest_path=g0_manifest_path,
            spec=spec,
        )
        print("Created deterministic G0.")
    else:
        print("Verified existing deterministic G0.")

    preparation = {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": "m0-single-a100-smoke-preparation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slakshna_revision": revision,
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "tokenized_views_manifest": str(token_manifest_path),
        "tokenized_views_manifest_sha256": sha256_file(token_manifest_path),
        "g0_manifest": str(g0_manifest_path),
        "g0_manifest_sha256": sha256_file(g0_manifest_path),
        "g0_state_sha256": g0_manifest["state_file_sha256"],
        "run_root": str(run_root),
    }
    atomic_json(preparation, args.preparation_manifest.resolve())
    print(f"Resolved config: {resolved_path}")
    print(f"G0: {g0_path}")
    print(f"G0 tensors: {g0_manifest['state']['tensor_count']}")
    print(f"G0 parameters: {g0_manifest['state']['parameter_count']}")
    print("M0 SINGLE-GPU SMOKE PREPARATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
