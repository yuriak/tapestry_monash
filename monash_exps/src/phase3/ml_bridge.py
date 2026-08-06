#!/usr/bin/env python3
"""One-round Slakshna ML-engine bridge backed by the audited Phase 1 harness."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}: {command}"
        )


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return Path(value).resolve()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ml_engine.py <node-id> [peer-id ...]")
    node_id = sys.argv[1]
    peers = sys.argv[2:]
    if peers:
        raise RuntimeError(f"Phase 3 is single-node only; unexpected peers: {peers}")

    artifact_root = required_path("SLAKSHNA_PHASE3_ARTIFACT_ROOT")
    experiment_root = required_path("SLAKSHNA_EXPERIMENT_ROOT")
    run_id = os.environ.get("SLAKSHNA_PHASE3_RUN_ID", artifact_root.name)
    artifact_root.mkdir(parents=True, exist_ok=True)

    invocation_marker = artifact_root / "ml-bridge.invoked"
    try:
        descriptor = os.open(invocation_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Phase 3 bridge refuses a second local-training round") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"node_id={node_id}\npid={os.getpid()}\n")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Phase 3 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )

    training_dir = artifact_root / "local-training"
    update_dir = artifact_root / "update"
    initial_adapter = artifact_root / "initial_adapter.safetensors"
    training_dir.mkdir(parents=True, exist_ok=True)
    update_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    resolved_config = training_dir / "resolved-config.yaml"
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/prepare_experiment.py"),
            "--template",
            str(experiment_root / "configs/phase1/phase1a_single_gpu.yaml"),
            "--run-dir",
            str(training_dir),
            "--checkpoint-dir",
            str(training_dir / "checkpoints"),
            "--run-name",
            f"{run_id}-slakshna-local",
        ],
        training_dir / "prepare.log",
    )
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/launch_training.py"),
            "--config",
            str(resolved_config),
            "--num-workers",
            "1",
            "--run-dir",
            str(training_dir),
            "--capture-initial-adapter",
            str(initial_adapter),
        ],
        training_dir / "train.log",
    )
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/verify_training.py"),
            "--mode",
            "phase1a",
            "--run-dir",
            str(training_dir),
            "--expected-workers",
            "1",
            "--expected-final-step",
            "4",
        ],
        training_dir / "verify.log",
    )

    trained_adapter = training_dir / "checkpoints/step_0000004/adapter_model.safetensors"
    if not trained_adapter.is_file():
        raise RuntimeError(f"trained adapter was not produced: {trained_adapter}")
    run_logged(
        [
            python,
            str(experiment_root / "src/phase2/adapter_delta.py"),
            "create",
            "--initial",
            str(initial_adapter),
            "--trained",
            str(trained_adapter),
            "--config",
            str(resolved_config),
            "--output-dir",
            str(update_dir),
        ],
        update_dir / "create.log",
    )

    phase1_summary = json.loads(
        (training_dir / "verification-summary.json").read_text(encoding="utf-8")
    )
    update_manifest = json.loads((update_dir / "update-manifest.json").read_text(encoding="utf-8"))
    if phase1_summary.get("status") != "PASS":
        raise RuntimeError("local training verifier did not pass")
    if update_manifest.get("nonzero_delta_values", 0) <= 0:
        raise RuntimeError("canonical delta is empty")

    delta_bytes = (update_dir / "delta.safetensors").read_bytes()
    model_hash = hashlib.sha256(delta_bytes).hexdigest()
    compressed_bytes = zlib.compress(delta_bytes, level=9)
    compressed_delta = base64.b64encode(compressed_bytes).decode("ascii")
    metrics = phase1_summary["checks"]["finite_metrics_and_loss_trend"]["detail"]
    final_loss = float(metrics["final_median_loss"])
    validation_score = math.exp(-final_loss)

    output = {
        "weights": {node_id: 1.0},
        "model_hash": model_hash,
        "validation_score": validation_score,
        "metadata": (
            f"phase3 canonical LoRA delta; final_median_loss={final_loss:.6f}; "
            "encoding=zlib+safetensors"
        ),
        "compressed_delta": compressed_delta,
    }
    audit = {
        "schema_version": 1,
        "node_id": node_id,
        "peers": peers,
        "python": python,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "iiitd_data_dir": os.environ.get("IIITD_DATA_DIR", ""),
        "delta_encoding": "base64(zlib(safetensors))",
        "delta_uncompressed_bytes": len(delta_bytes),
        "delta_compressed_bytes": len(compressed_bytes),
        "compression_ratio": len(compressed_bytes) / len(delta_bytes),
        "model_hash": model_hash,
        "validation_score": validation_score,
        "final_median_loss": final_loss,
        "update_manifest": update_manifest,
    }
    write_json(artifact_root / "ml-bridge-audit.json", audit)
    write_json(artifact_root / "ml-engine-output.json", output)

    # Rust consumes only the final stdout line. Keep every other detail in files.
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
