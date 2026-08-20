#!/usr/bin/env python3
"""Audit the M0 OLMo 2 7B single-A100 Bhaskera smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from bhaskera.config import load_config
from safetensors.torch import load_file

SCHEMA_VERSION = 1
WORKFLOW_VERSION = "m0-single-a100-smoke-audit-v1"
LOSS_PATTERN = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\]\[step\s+(?P<step>\d+)\]\s+"
    r"loss=(?P<loss>[0-9.eE+-]+)\s+lr=(?P<lr>[0-9.eE+-]+)\s+"
    r"grad_norm=(?P<grad>[0-9.eE+-]+)(?:\s+tok/s=(?P<tps>[0-9.eE+-]+))?"
    r"(?:\s+MFU=(?P<mfu>[0-9.eE+-]+)%)?"
)


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


def vector_norm(state: dict[str, torch.Tensor]) -> float:
    return math.sqrt(
        sum(torch.sum(tensor.float() ** 2).item() for tensor in state.values())
    )


def audit_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    cfg = load_config(str(config_path))
    expected = {
        "model.dtype": "bfloat16",
        "model.attn_impl": "flash_attention_2",
        "model.use_liger_kernel": True,
        "data.name": "local",
        "data.seq_len": 1024,
        "lora.r": 16,
        "lora.alpha": 64,
        "lora.dropout": 0.03,
        "lora.target_modules": ["q_proj", "v_proj"],
        "training.batch_size": 2,
        "training.grad_accum": 4,
        "training.lr": 1.0e-4,
        "training.max_steps": 12,
        "training.distributed.strategy": "ddp",
    }
    actual = {
        "model.dtype": cfg.model.dtype,
        "model.attn_impl": cfg.model.attn_impl,
        "model.use_liger_kernel": cfg.model.use_liger_kernel,
        "data.name": cfg.data.name,
        "data.seq_len": cfg.data.seq_len,
        "lora.r": cfg.lora.r,
        "lora.alpha": cfg.lora.alpha,
        "lora.dropout": cfg.lora.dropout,
        "lora.target_modules": cfg.lora.target_modules,
        "training.batch_size": cfg.training.batch_size,
        "training.grad_accum": cfg.training.grad_accum,
        "training.lr": cfg.training.lr,
        "training.max_steps": cfg.training.max_steps,
        "training.distributed.strategy": cfg.training.distributed.strategy,
    }
    if actual != expected:
        raise RuntimeError(f"resolved smoke config mismatch: {actual}")
    if not Path(cfg.model.name).is_absolute() or not Path(
        cfg.data.tokenized_path
    ).is_absolute():
        raise RuntimeError("resolved model/cache paths are not absolute")
    if not Path(cfg.lora.resume_path).is_absolute() or not Path(
        cfg.checkpoint.save_dir
    ).is_absolute():
        raise RuntimeError("resolved G0/checkpoint paths are not absolute")
    return cfg, actual


def audit_log(log_path: Path, expected_steps: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    required_markers = [
        "Loading pre-tokenized dataset",
        "Liger Kernel applied",
        "Loading existing LoRA weights",
        "LoRA applied",
        "DDP wrap complete",
        "Optimizer: AdamW (Default)",
        "Saved 128 LoRA tensors to adapter_model.safetensors",
        "Training complete.",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(f"training log is missing native-path markers: {missing}")
    forbidden = [
        "Tokenizing dataset",
        "Non-finite grad_norm",
        "CUDA out of memory",
        "torch.OutOfMemoryError",
    ]
    present = [marker for marker in forbidden if marker in text]
    if present:
        raise RuntimeError(f"training log contains failure markers: {present}")

    by_step: dict[int, dict[str, float | int | None]] = {}
    for match in LOSS_PATTERN.finditer(text):
        step = int(match.group("step"))
        record: dict[str, float | int | None] = {
            "epoch": int(match.group("epoch")),
            "step": step,
            "loss": float(match.group("loss")),
            "lr": float(match.group("lr")),
            "grad_norm": float(match.group("grad")),
            "tokens_per_second": (
                float(match.group("tps")) if match.group("tps") else None
            ),
            "mfu_percent": (
                float(match.group("mfu")) if match.group("mfu") else None
            ),
        }
        by_step[step] = record
    expected_sequence = list(range(1, expected_steps + 1))
    if sorted(by_step) != expected_sequence:
        raise RuntimeError(
            f"expected loss steps {expected_sequence}, found {sorted(by_step)}"
        )
    records = [by_step[step] for step in expected_sequence]
    for record in records:
        values = (record["loss"], record["lr"], record["grad_norm"])
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"non-finite training record: {record}")
        if float(record["loss"]) <= 0 or float(record["loss"]) >= 20:
            raise RuntimeError(f"implausible smoke loss: {record}")

    losses = [float(record["loss"]) for record in records]
    tps = [
        float(record["tokens_per_second"])
        for record in records
        if record["tokens_per_second"] is not None
    ]
    mfu = [
        float(record["mfu_percent"])
        for record in records
        if record["mfu_percent"] is not None
    ]
    return {
        "records": records,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_mean": sum(losses) / len(losses),
        "tokens_per_second_mean": sum(tps) / len(tps) if tps else None,
        "tokens_per_second_max": max(tps) if tps else None,
        "mfu_percent_mean": sum(mfu) / len(mfu) if mfu else None,
        "mfu_percent_max": max(mfu) if mfu else None,
        "native_markers": required_markers,
    }


def audit_gpu_csv(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 7:
                continue
            try:
                records.append(
                    {
                        "timestamp": row[0].strip(),
                        "index": int(row[1]),
                        "name": row[2].strip(),
                        "utilization_percent": float(row[3]),
                        "memory_used_mib": float(row[4]),
                        "memory_total_mib": float(row[5]),
                        "power_w": float(row[6]),
                    }
                )
            except ValueError:
                continue
    if not records:
        raise RuntimeError(f"GPU telemetry contains no valid records: {path}")
    names = sorted({str(record["name"]) for record in records})
    if len(names) != 1 or "A100" not in names[0]:
        raise RuntimeError(f"smoke did not run on exactly one A100: {names}")
    if min(float(record["memory_total_mib"]) for record in records) < 75_000:
        raise RuntimeError("visible A100 does not have the expected 80 GB memory class")
    return {
        "samples": len(records),
        "gpu_name": names[0],
        "peak_memory_used_mib": max(
            float(record["memory_used_mib"]) for record in records
        ),
        "peak_utilization_percent": max(
            float(record["utilization_percent"]) for record in records
        ),
        "mean_utilization_percent": sum(
            float(record["utilization_percent"]) for record in records
        )
        / len(records),
        "peak_power_w": max(float(record["power_w"]) for record in records),
        "first_sample": records[0],
        "last_sample": records[-1],
    }


def audit_checkpoint(
    *, cfg: Any, g0_path: Path, slakshna_root: Path
) -> dict[str, Any]:
    checkpoint_root = Path(cfg.checkpoint.save_dir)
    candidates = sorted(
        path
        for path in checkpoint_root.glob("step_*")
        if path.is_dir() and (path / ".complete").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one completed checkpoint: {candidates}")
    checkpoint = candidates[0]
    meta_path = checkpoint / "meta.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not meta_path.is_file() or not adapter_path.is_file():
        raise RuntimeError(f"checkpoint is incomplete: {checkpoint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("step") != cfg.training.max_steps:
        raise RuntimeError(f"checkpoint stopped at unexpected step: {meta}")

    g0_loaded = torch.load(g0_path, map_location="cpu", weights_only=True)
    if not isinstance(g0_loaded, dict):
        raise TypeError("G0 is not a tensor dictionary")
    g0 = normalize_state(g0_loaded)
    trained = normalize_state(load_file(adapter_path, device="cpu"))
    if set(g0) != set(trained):
        raise RuntimeError(
            f"trained/G0 adapter schemas differ: "
            f"missing={sorted(set(g0) - set(trained))[:5]}, "
            f"extra={sorted(set(trained) - set(g0))[:5]}"
        )

    delta: dict[str, torch.Tensor] = {}
    changed = 0
    for name in sorted(g0):
        if trained[name].shape != g0[name].shape:
            raise RuntimeError(f"adapter shape changed: {name}")
        if not torch.isfinite(trained[name]).all():
            raise RuntimeError(f"trained adapter contains non-finite values: {name}")
        value = trained[name] - g0[name]
        delta[name] = value
        changed += int(bool(torch.count_nonzero(value)))
    dense_norm = vector_norm(delta)
    if changed == 0 or not math.isfinite(dense_norm) or dense_norm <= 0:
        raise RuntimeError("training did not change the LoRA adapter")

    sys.path.insert(0, str(slakshna_root))
    from federated_communication import (
        DELTA_QUANTIZATION,
        DELTA_SPARSITY,
        MAX_DELTA_PAYLOAD_BYTES,
        apply_differential_privacy_and_clipping,
        decode_delta_envelope,
        encode_delta_envelope,
        validate_peer_delta,
    )

    clipped = apply_differential_privacy_and_clipping(
        delta, max_norm=100.0, noise_multiplier=0.0
    )
    clipped_norm = vector_norm(clipped)
    payload, reconstructed, transport = encode_delta_envelope(
        clipped, sparsity=DELTA_SPARSITY, sender="m0-single-a100-smoke", round_number=1
    )
    decoded = decode_delta_envelope(payload, torch.device("cpu"))
    if set(decoded) != set(reconstructed):
        raise RuntimeError("wire-format decode schema mismatch")
    if not validate_peer_delta(decoded):
        raise RuntimeError(
            "the current receiver policy rejects the representative smoke delta"
        )
    gossip_limit = 10 * 1024 * 1024
    if transport["serialized_bytes"] > MAX_DELTA_PAYLOAD_BYTES:
        raise RuntimeError("representative delta exceeds Slakshna payload limit")
    if transport["base64_bytes"] >= gossip_limit:
        raise RuntimeError("base64 delta leaves no room under the 10 MiB Gossip limit")

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_meta": meta,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": sha256_file(adapter_path),
        "tensor_count": len(trained),
        "parameter_count": sum(tensor.numel() for tensor in trained.values()),
        "changed_tensors": changed,
        "dense_delta_l2_norm": dense_norm,
        "post_clip_delta_l2_norm": clipped_norm,
        "transport": {
            **transport,
            "sparsity": DELTA_SPARSITY,
            "quantization": DELTA_QUANTIZATION,
            "payload_limit_bytes": MAX_DELTA_PAYLOAD_BYTES,
            "gossip_limit_bytes": gossip_limit,
            "base64_gossip_fraction": transport["base64_bytes"] / gossip_limit,
            "receiver_validation": "passed",
        },
    }


def main() -> int:
    inferred_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g0-manifest", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--started-epoch", type=int, required=True)
    parser.add_argument("--completed-epoch", type=int, required=True)
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    config_path = args.config.resolve()
    g0_path = args.g0.resolve()
    g0_manifest_path = args.g0_manifest.resolve()
    preparation_path = args.preparation_manifest.resolve()
    training_log = args.training_log.resolve()
    gpu_csv = args.gpu_csv.resolve()
    for required in (
        config_path,
        g0_path,
        g0_manifest_path,
        preparation_path,
        training_log,
        gpu_csv,
    ):
        if not required.is_file():
            parser.error(f"missing audit input: {required}")
    if args.completed_epoch < args.started_epoch:
        parser.error("completed time predates start time")

    cfg, config_summary = audit_config(config_path)
    training = audit_log(training_log, cfg.training.max_steps)
    gpu = audit_gpu_csv(gpu_csv)
    checkpoint = audit_checkpoint(
        cfg=cfg, g0_path=g0_path, slakshna_root=workspace_root / "Slakshna"
    )
    g0_manifest = json.loads(g0_manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if sha256_file(g0_path) != g0_manifest["state_file_sha256"]:
        raise RuntimeError("G0 changed after smoke preparation")
    if sha256_file(config_path) != preparation["resolved_config_sha256"]:
        raise RuntimeError("resolved smoke config changed after preparation")

    result = {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": args.completed_epoch - args.started_epoch,
        "scope": "single-GPU DDP-path smoke; not a multi-GPU DDP validation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_summary": config_summary,
        "g0_manifest": str(g0_manifest_path),
        "g0_manifest_sha256": sha256_file(g0_manifest_path),
        "g0_state_sha256": g0_manifest["state_file_sha256"],
        "preparation_manifest": str(preparation_path),
        "training_log": str(training_log),
        "training_log_sha256": sha256_file(training_log),
        "gpu_csv": str(gpu_csv),
        "gpu_csv_sha256": sha256_file(gpu_csv),
        "training": training,
        "gpu": gpu,
        "checkpoint": checkpoint,
    }
    atomic_json(result, args.output.resolve())
    print(f"Loss: {training['loss_first']:.4f} -> {training['loss_last']:.4f}")
    print(f"Peak GPU memory: {gpu['peak_memory_used_mib']:.0f} MiB")
    print(
        "Wire delta: "
        f"{checkpoint['transport']['serialized_bytes']} raw bytes, "
        f"{checkpoint['transport']['base64_bytes']} base64 bytes"
    )
    print(f"Audit manifest: {args.output.resolve()}")
    print("M0 SINGLE-GPU OLMO2-7B SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
