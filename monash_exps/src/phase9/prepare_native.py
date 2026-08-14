#!/usr/bin/env python3
"""Prepare pinned OLMo configs and offline Bhaskera caches for Phase 9."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

MODEL_ID = "allenai/OLMo-1B-hf"
MODEL_REVISION = "aee7752d9c08ee4775e9b0091426d8410e8f6a89"
SLAKSHNA_REVISION = "9f93ec45ae0d3eb9c901aff3b50d4325b5050488"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def download_model(model_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    model_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID}@{MODEL_REVISION} to {model_root}", flush=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(model_root),
            allow_patterns=[
                "*.json",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.txt",
            ],
        )
    ).resolve()
    required = [snapshot_path / "config.json"]
    if not any(snapshot_path.glob("*.safetensors")) and not any(
        snapshot_path.glob("pytorch_model*.bin")
    ):
        required.append(snapshot_path / "model.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete OLMo snapshot; missing: {missing}")
    return snapshot_path


def model_manifest(model_path: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(model_path).as_posix()
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(model_path),
        "bytes": total_bytes,
        "files": files,
    }


def load_profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported profile schema in {path}")
    active = payload.get("active_profile")
    profiles = payload.get("profiles", {})
    if active not in profiles or not profiles:
        raise RuntimeError(f"Invalid active profile in {path}")
    return payload


def resolve_template(
    *,
    template_text: str,
    profile_name: str,
    profile: dict[str, Any],
    model_path: Path,
    data_root: Path,
    cache_root: Path,
    checkpoint_root: Path,
    workers: int,
) -> dict[str, Any]:
    replacements = {
        "__PHASE9_MODEL_PATH__": str(model_path),
        "__PHASE9_TRAIN_PATH__": str((data_root / profile["train"]).resolve()),
        "__PHASE9_VAL_PATH__": str((data_root / profile["validation"]).resolve()),
        "__PHASE9_TOKENIZED_CACHE_ROOT__": str(cache_root.resolve()),
        "__PHASE9_TOKENIZED_TRAIN_PATH__": str(cache_root.resolve()),
        "__PHASE9_TOKENIZED_VAL_PATH__": str(cache_root.resolve()),
        "__PHASE9_CHECKPOINT_PATH__": str(checkpoint_root.resolve()),
        "__PHASE9_TOKENIZER_WORKERS__": str(workers),
        "__PHASE9_MAX_STEPS__": str(int(profile["max_steps"])),
        "__PHASE9_PROFILE__": profile_name,
    }
    resolved = template_text
    for placeholder, value in replacements.items():
        resolved = resolved.replace(placeholder, value)
    if "__PHASE9_" in resolved:
        raise RuntimeError(f"Unresolved Phase 9 placeholder in {profile_name}")
    config = yaml.safe_load(resolved)
    for split in ("train_path", "val_path"):
        if not Path(config["data"][split]).is_file():
            raise RuntimeError(f"{profile_name} missing {split}: {config['data'][split]}")
    return config


def audit_config(config_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    from bhaskera.config import load_config

    cfg = load_config(str(config_path))
    actual = {
        "model_name": cfg.model.name,
        "dtype": cfg.model.dtype,
        "attn_impl": cfg.model.attn_impl,
        "use_liger_kernel": cfg.model.use_liger_kernel,
        "data_name": cfg.data.name,
        "seq_len": cfg.data.seq_len,
        "format": cfg.data.format,
        "train_path": cfg.data.train_path,
        "val_path": cfg.data.val_path,
        "lora_enabled": cfg.lora.enabled,
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "target_modules": cfg.lora.target_modules,
        "batch_size": cfg.training.batch_size,
        "grad_accum": cfg.training.grad_accum,
        "lr": cfg.training.lr,
        "max_steps": cfg.training.max_steps,
        "strategy": cfg.training.distributed.strategy,
    }
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(
                f"Config audit failed for {config_path}: {key}={actual.get(key)!r}, "
                f"expected {value!r}"
            )
    return actual


def run_tokenizer(config_path: Path, profile_name: str, workers: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["RAY_TMPDIR"] = f"/tmp/sl_p9_tok_{profile_name.replace('-', '_')}"
    env["OMP_NUM_THREADS"] = str(max(1, workers))
    command = [
        sys.executable,
        "-m",
        "bhaskera.launcher.tokenize",
        "--config",
        str(config_path),
        "--split",
        "both",
        "--num-workers",
        str(workers),
    ]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def cache_for_split(cache_root: Path, split: str) -> Path:
    matches = []
    for metadata_path in cache_root.glob(f"local_{split}_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_name") == f"local_{split}":
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one local_{split} cache in {cache_root}, found {matches}"
        )
    return matches[0].resolve()


def audit_cache(cache_path: Path, expected_rows: int, model_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    metadata_path = cache_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_name": str(model_path),
        "seq_len": 512,
        "num_rows": expected_rows,
        "format_name": "alpaca",
        "is_cpt": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Cache audit failed at {cache_path}: {key}={metadata.get(key)!r}, "
                f"expected {value!r}"
            )
    parquet_files = sorted(cache_path.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files in {cache_path}")
    parquet_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
    if parquet_rows != expected_rows:
        raise RuntimeError(
            f"Parquet row mismatch at {cache_path}: {parquet_rows} != {expected_rows}"
        )
    first_parquet = pq.ParquetFile(parquet_files[0])
    sample = next(first_parquet.iter_batches(batch_size=1)).to_pylist()
    required_columns = {"input_ids", "attention_mask", "labels"}
    if not sample or not required_columns.issubset(sample[0]):
        raise RuntimeError(
            f"Tokenized cache at {cache_path} lacks a complete training row"
        )
    tensor_lengths = {column: len(sample[0][column]) for column in required_columns}
    if set(tensor_lengths.values()) != {512}:
        raise RuntimeError(
            f"Tokenized row at {cache_path} is not seq_len=512: {tensor_lengths}"
        )
    return {
        "path": str(cache_path),
        "rows": parquet_rows,
        "columns": sorted(required_columns),
        "sample_tensor_lengths": tensor_lengths,
        "metadata": metadata,
        "metadata_sha256": sha256_file(metadata_path),
        "parquet": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in parquet_files
        ],
    }


def verify_model_and_lora(config_path: Path, output_path: Path) -> None:
    import torch
    from bhaskera.config import load_config
    from bhaskera.models.loader import build_model
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 9 native preparation requires a visible GPU")
    cfg = load_config(str(config_path))
    device = torch.device("cuda:0")
    model, profile = build_model(cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, local_files_only=True)
    encoded = tokenizer("A short Phase 9 compatibility check.", return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**encoded).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("OLMo compatibility forward produced non-finite logits")
    trainable = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or not all("lora_" in name for name in trainable):
        raise RuntimeError("OLMo compatibility model did not expose LoRA-only trainables")
    payload = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "model_type": profile.model_type,
        "attention_impl": cfg.model.attn_impl,
        "liger_requested": cfg.model.use_liger_kernel,
        "lora_targets": cfg.lora.target_modules,
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "logits_shape": list(logits.shape),
        "finite_logits": True,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    del model, logits
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slakshna-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slakshna_root = args.slakshna_root.resolve()
    data_root = args.data_root.resolve()
    model_root = args.model_root.resolve()
    tokenized_root = args.tokenized_root.resolve()
    phase9_config_root = slakshna_root / "configs" / "phase9"
    resolved_root = slakshna_root / "phase9_runtime" / "configs"
    checkpoint_root = slakshna_root / "phase9_runtime" / "checkpoints"
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if not (data_root / "manifest.json").is_file():
        raise SystemExit(f"Missing Phase 9 data manifest: {data_root / 'manifest.json'}")

    profiles_payload = load_profiles(phase9_config_root / "profiles.json")
    data_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    template_text = (phase9_config_root / "node_template.yaml.in").read_text(
        encoding="utf-8"
    )
    model_path = download_model(model_root)
    model_info = model_manifest(model_path)

    resolved_root.mkdir(parents=True, exist_ok=True)
    tokenized_root.mkdir(parents=True, exist_ok=True)
    profile_manifests: dict[str, Any] = {}
    for profile_name, profile in profiles_payload["profiles"].items():
        cache_root = tokenized_root / profile_name
        config = resolve_template(
            template_text=template_text,
            profile_name=profile_name,
            profile=profile,
            model_path=model_path,
            data_root=data_root,
            cache_root=cache_root,
            checkpoint_root=checkpoint_root / profile_name,
            workers=args.workers,
        )
        config_path = resolved_root / f"node_template.{profile_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        expected_audit = {
            "model_name": str(model_path),
            "dtype": "bfloat16",
            "attn_impl": "flash_attention_2",
            "use_liger_kernel": True,
            "data_name": "local",
            "seq_len": 512,
            "format": "alpaca",
            "train_path": str((data_root / profile["train"]).resolve()),
            "val_path": str((data_root / profile["validation"]).resolve()),
            "lora_enabled": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "batch_size": 2,
            "grad_accum": 4,
            "lr": 0.0001,
            "max_steps": int(profile["max_steps"]),
            "strategy": "ddp",
        }
        config_audit = audit_config(config_path, expected_audit)

        run_tokenizer(config_path, profile_name, args.workers)
        train_cache = cache_for_split(cache_root, "train")
        val_cache = cache_for_split(cache_root, "val")

        country = "Australia" if profile_name.startswith("australia") else "India"
        country_manifest = data_manifest["countries"][country]
        prefix = "smoke" if profile_name.endswith("smoke") else "full"
        train_rows = country_manifest["split"][f"{prefix}_train_rows"]
        val_rows = country_manifest["split"][f"{prefix}_validation_rows"]
        train_audit = audit_cache(train_cache, train_rows, model_path)
        val_audit = audit_cache(val_cache, val_rows, model_path)

        config["data"]["tokenized_path"] = str(train_cache)
        config["data"]["val_tokenized_path"] = str(val_cache)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        config_audit = audit_config(config_path, expected_audit)
        profile_manifests[profile_name] = {
            "source": profile,
            "resolved_config": str(config_path),
            "resolved_config_sha256": sha256_file(config_path),
            "config_audit": config_audit,
            "train_cache": train_audit,
            "validation_cache": val_audit,
        }

    active_profile = profiles_payload["active_profile"]
    active_config = Path(profile_manifests[active_profile]["resolved_config"])
    shutil.copy2(active_config, slakshna_root / "node_template.yaml")
    verify_model_and_lora(
        active_config,
        args.manifest.parent / "model-compatibility.json",
    )

    core_files = [
        slakshna_root / "ml_engine.py",
        slakshna_root / "src" / "main.rs",
        slakshna_root / "src" / "history.rs",
        slakshna_root / "Bhaskera" / "src" / "bhaskera" / "trainer" / "loop.py",
    ]
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "slakshna_revision": SLAKSHNA_REVISION,
        "model": model_info,
        "data_manifest": str(data_root / "manifest.json"),
        "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "active_profile": active_profile,
        "active_node_template": str(slakshna_root / "node_template.yaml"),
        "active_node_template_sha256": sha256_file(
            slakshna_root / "node_template.yaml"
        ),
        "profiles": profile_manifests,
        "core_source": {
            str(path.relative_to(slakshna_root)): sha256_file(path)
            for path in core_files
        },
        "model_compatibility": str(args.manifest.parent / "model-compatibility.json"),
    }
    write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "active_profile": active_profile,
                "model_path": str(model_path),
                "profiles": {
                    name: {
                        "train_rows": info["train_cache"]["rows"],
                        "validation_rows": info["validation_cache"]["rows"],
                    }
                    for name, info in profile_manifests.items()
                },
                "manifest": str(args.manifest),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
