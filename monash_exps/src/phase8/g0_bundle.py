#!/usr/bin/env python3
"""Create and install a portable, self-verifying Phase 8 G0 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import max_abs_error, read_state, state_sha256, write_json  # noqa: E402
from phase8.protocol import sha256_bytes  # noqa: E402


FORMAT = "slakshna-phase8-g0-bundle"
VERSION = 1


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def training_contract(site_root: Path) -> dict[str, Any]:
    manifest = json.loads((site_root / "site-manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (site_root / "bootstrap/resolved-config.yaml").read_text(encoding="utf-8")
    )
    return {
        "dataset": manifest["dataset"],
        "partition_type": manifest["partition_type"],
        "model": {
            "id": manifest["model"]["id"],
            "revision": manifest["model"]["revision"],
            "dtype": config["model"]["dtype"],
            "attn_impl": config["model"]["attn_impl"],
            "quantization": config["model"]["quantization"],
        },
        "data": {
            "format": config["data"]["format"],
            "seq_len": config["data"]["seq_len"],
            "pack_sequences": config["data"]["pack_sequences"],
            "train_rows_per_site": manifest["shard"]["train"]["rows"],
            "validation_rows_per_site": manifest["shard"]["validation"]["rows"],
        },
        "lora": {key: config["lora"][key] for key in (
            "enabled", "r", "alpha", "dropout", "target_modules"
        )},
        "training": {key: config["training"][key] for key in (
            "batch_size", "grad_accum", "lr", "weight_decay", "warmup_steps",
            "max_grad_norm", "seed", "deterministic", "distributed"
        )},
        "federated_training": {
            "algorithm": "equal-weight-dense-fedavg",
            "sites": 2,
            "site_weights": [0.5, 0.5],
            "training_rounds": 5,
            "local_epochs_per_round": 10,
            "optimizer_steps_per_round": 720,
        },
    }


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def verify_bundle(bundle_dir: Path, site_root: Path | None = None) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest = json.loads((bundle_dir / "g0-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise RuntimeError("unsupported Phase 8 G0 bundle")
    contract = manifest["training_contract"]
    if contract_sha256(contract) != manifest["training_contract_sha256"]:
        raise RuntimeError("G0 training contract hash mismatch")
    adapter = bundle_dir / manifest["adapter"]["filename"]
    resume = bundle_dir / manifest["resume"]["filename"]
    if file_sha256(adapter) != manifest["adapter"]["file_sha256"]:
        raise RuntimeError("G0 safetensors file hash mismatch")
    if file_sha256(resume) != manifest["resume"]["file_sha256"]:
        raise RuntimeError("G0 resume file hash mismatch")
    adapter_state = read_state(adapter)
    resume_raw = torch.load(resume, map_location="cpu", weights_only=True)
    resume_state = {name: value.float().contiguous() for name, value in resume_raw.items()}
    if state_sha256(adapter_state) != manifest["adapter"]["state_sha256"]:
        raise RuntimeError("G0 adapter state hash mismatch")
    if not resume_state or not all(torch.isfinite(value).all() for value in resume_state.values()):
        raise RuntimeError("G0 resume state is empty or non-finite")
    if state_sha256(resume_state) != manifest["adapter"]["state_sha256"]:
        raise RuntimeError("G0 resume logical state hash mismatch")
    error = max_abs_error(adapter_state, resume_state)
    if error > 1.0e-7:
        raise RuntimeError(f"G0 resume state differs from safetensors: {error}")
    if site_root is not None:
        local_contract = training_contract(site_root.resolve())
        if contract_sha256(local_contract) != manifest["training_contract_sha256"]:
            raise RuntimeError("local site training contract is incompatible with G0")
    return manifest


def create(site_root: Path, output_dir: Path) -> None:
    site_root = site_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = output_dir / "global_adapter.safetensors"
    resume = output_dir / "global_adapter.pth"
    audit = output_dir / "creation-audit.json"
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_ROOT / "src/phase5/create_initial_adapter.py"),
            "--config", str(site_root / "bootstrap/resolved-config.yaml"),
            "--output", str(adapter),
            "--resume-output", str(resume),
            "--audit-output", str(audit),
        ],
        check=True,
    )
    state = read_state(adapter)
    contract = training_contract(site_root)
    site_manifest = json.loads((site_root / "site-manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "created_by_site": site_manifest["site"],
        "training_contract": contract,
        "training_contract_sha256": contract_sha256(contract),
        "adapter": {
            "filename": adapter.name,
            "file_sha256": file_sha256(adapter),
            "state_sha256": state_sha256(state),
            "tensor_count": len(state),
            "parameter_count": sum(value.numel() for value in state.values()),
        },
        "resume": {"filename": resume.name, "file_sha256": file_sha256(resume)},
    }
    write_json(output_dir / "g0-manifest.json", manifest)
    verify_bundle(output_dir, site_root)
    print(json.dumps({"status": "PASS", **manifest}, indent=2, sort_keys=True))


def install(bundle_dir: Path, site_root: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    site_root = site_root.resolve()
    manifest = verify_bundle(bundle_dir, site_root)
    destination = site_root / "global-0"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("global_adapter.safetensors", "global_adapter.pth", "g0-manifest.json"):
        source = bundle_dir / name
        temporary = destination / f"{name}.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(destination / name)
    verify_bundle(destination, site_root)
    print(json.dumps({
        "status": "PASS",
        "installed_site": json.loads((site_root / "site-manifest.json").read_text())["site"],
        "training_contract_sha256": manifest["training_contract_sha256"],
        "adapter_state_sha256": manifest["adapter"]["state_sha256"],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--site-root", type=Path, required=True)
    create_parser.add_argument("--output-dir", type=Path, required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--bundle-dir", type=Path, required=True)
    install_parser.add_argument("--site-root", type=Path, required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--bundle-dir", type=Path, required=True)
    inspect_parser.add_argument("--site-root", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        create(args.site_root, args.output_dir)
    elif args.command == "install":
        install(args.bundle_dir, args.site_root)
    else:
        print(json.dumps(verify_bundle(args.bundle_dir, args.site_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
