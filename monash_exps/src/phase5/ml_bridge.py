#!/usr/bin/env python3
"""Two-round ML bridge used by each real Slakshna peer in Phase 5."""
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
from safetensors.torch import load

# Rust executes a copied bridge from each peer's isolated runtime directory, so
# __file__ no longer has the source tree as a parent. The runner supplies the
# canonical experiment root explicitly.
_experiment_root = os.environ.get("SLAKSHNA_EXPERIMENT_ROOT")
if not _experiment_root:
    raise RuntimeError("required environment variable is unset: SLAKSHNA_EXPERIMENT_ROOT")
sys.path.insert(0, str(Path(_experiment_root).resolve() / "src"))
from phase2.adapter_delta import (  # noqa: E402
    TOLERANCE,
    atomic_safetensors,
    max_abs_error,
    read_state,
    require_compatible,
    state_sha256,
    write_json,
)


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return Path(value).resolve()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def encode_delta(path: Path) -> tuple[str, str, int, int]:
    raw = path.read_bytes()
    compressed = zlib.compress(raw, level=9)
    return (
        base64.b64encode(compressed).decode("ascii"),
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        len(compressed),
    )


def decode_delta(payload: str) -> tuple[bytes, dict[str, torch.Tensor]]:
    try:
        raw = zlib.decompress(base64.b64decode(payload, validate=True))
        state = {name: value.float().contiguous() for name, value in load(raw).items()}
    except Exception as exc:
        raise RuntimeError(f"invalid peer base64(zlib(safetensors)) payload: {exc}") from exc
    if not state:
        raise RuntimeError("decoded peer delta is empty")
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"decoded peer delta is non-finite: {name}")
    return raw, state


def atomic_torch_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.cpu().contiguous() for name, value in state.items()}, temporary)
    temporary.replace(path)


def prepare_round_two_base(
    peer_root: Path,
    global0_path: Path,
    data_dir: Path,
    peer_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    own_delta_path = peer_root / "round-1/update/delta.safetensors"
    own_delta = read_state(own_delta_path)
    peer_payload_path = data_dir / "network_deltas" / f"{peer_id}_delta.b64"
    if not peer_payload_path.is_file():
        raise RuntimeError(f"Rust did not stage the peer delta: {peer_payload_path}")
    payload = peer_payload_path.read_text(encoding="utf-8").strip()
    peer_raw, peer_delta = decode_delta(payload)
    require_compatible(own_delta, peer_delta, "own/peer Round 1 delta")
    peer_hash = hashlib.sha256(peer_raw).hexdigest()
    peer_delta_path = peer_root / "round-2/received-peer-delta.safetensors"
    peer_delta_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_peer = peer_delta_path.with_suffix(peer_delta_path.suffix + ".tmp")
    temporary_peer.write_bytes(peer_raw)
    temporary_peer.replace(peer_delta_path)

    delta_difference = max_abs_error(own_delta, peer_delta)
    if delta_difference == 0.0:
        raise RuntimeError("the two Round 1 peer deltas are identical")
    base = read_state(global0_path)
    require_compatible(base, own_delta, "G0/own delta")
    aggregate_delta = {
        name: (own_delta[name] * 0.5 + peer_delta[name] * 0.5).float().contiguous()
        for name in base
    }
    global1 = {
        name: (base[name] + aggregate_delta[name]).float().contiguous() for name in base
    }
    aggregation_dir = peer_root / "round-2/aggregation"
    delta_path = aggregation_dir / "aggregate_delta.safetensors"
    global_path = aggregation_dir / "global_adapter.safetensors"
    resume_path = aggregation_dir / "global_adapter.pth"
    atomic_safetensors(aggregate_delta, delta_path)
    atomic_safetensors(global1, global_path)
    atomic_torch_state(global1, resume_path)
    manifest = {
        "schema_version": 1,
        "algorithm": "accepted-peer-renormalized-fedavg",
        "weights": {"self": 0.5, "peer": 0.5},
        "global0_path": str(global0_path),
        "own_delta_path": str(own_delta_path),
        "peer_delta_transport_path": str(peer_payload_path),
        "peer_delta_path": str(peer_delta_path),
        "peer_id": peer_id,
        "peer_delta_hash": peer_hash,
        "peer_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "client_delta_max_abs_difference": delta_difference,
        "aggregate_delta_state_sha256": state_sha256(aggregate_delta),
        "global1_state_sha256": state_sha256(global1),
        "global1_change_max_abs": max_abs_error(base, global1),
        "tensor_count": len(global1),
        "parameter_count": sum(value.numel() for value in global1.values()),
        "tolerance": TOLERANCE,
    }
    write_json(aggregation_dir / "aggregation-manifest.json", manifest)
    return global_path, resume_path, manifest


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ml_engine.py <node-id> [peer-id ...]")
    node_id = sys.argv[1]
    neighbors = sys.argv[2:]
    peer_name = os.environ.get("SLAKSHNA_PHASE5_PEER_NAME", "")
    if peer_name not in {"peer-a", "peer-b"}:
        raise RuntimeError(f"invalid SLAKSHNA_PHASE5_PEER_NAME: {peer_name}")
    row_indices = os.environ.get("SLAKSHNA_PHASE5_ROW_INDICES", "")
    expected_rows = "0,2,4,6" if peer_name == "peer-a" else "1,3,5,7"
    if row_indices != expected_rows:
        raise RuntimeError(f"{peer_name} row assignment is {row_indices}, expected {expected_rows}")

    artifact_root = required_path("SLAKSHNA_PHASE5_ARTIFACT_ROOT")
    experiment_root = required_path("SLAKSHNA_EXPERIMENT_ROOT")
    global0_path = required_path("SLAKSHNA_PHASE5_GLOBAL0")
    global0_resume = required_path("SLAKSHNA_PHASE5_GLOBAL0_RESUME")
    data_dir = required_path("IIITD_DATA_DIR")
    run_id = os.environ.get("SLAKSHNA_PHASE5_RUN_ID", artifact_root.name)
    peer_root = artifact_root / peer_name
    peer_root.mkdir(parents=True, exist_ok=True)
    state_path = peer_root / "bridge-state.json"
    state = {"rounds_completed": 0}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    round_number = int(state.get("rounds_completed", 0)) + 1
    if round_number not in {1, 2}:
        raise RuntimeError(f"Phase 5 bridge refuses local-training round {round_number}")
    if round_number == 1 and neighbors:
        raise RuntimeError(f"Round 1 unexpectedly received peer history: {neighbors}")
    if round_number == 2 and len(neighbors) != 1:
        raise RuntimeError(f"Round 2 requires exactly one peer update, got {neighbors}")

    aggregation_manifest = None
    if round_number == 1:
        round_base = global0_path
        resume_path = global0_resume
    else:
        round_base, resume_path, aggregation_manifest = prepare_round_two_base(
            peer_root, global0_path, data_dir, neighbors[0]
        )

    round_dir = peer_root / f"round-{round_number}"
    training_dir = round_dir / "local-training"
    update_dir = round_dir / "update"
    training_dir.mkdir(parents=True, exist_ok=True)
    update_dir.mkdir(parents=True, exist_ok=True)
    initial_adapter = training_dir / "initial_adapter.safetensors"
    python = sys.executable
    resolved_config = training_dir / "resolved-config.yaml"
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/prepare_experiment.py"),
            "--template",
            str(experiment_root / "configs/phase4/client_two_steps.yaml"),
            "--run-dir",
            str(training_dir),
            "--checkpoint-dir",
            str(training_dir / "checkpoints"),
            "--run-name",
            f"{run_id}-{peer_name}-round{round_number}",
            "--lora-resume-path",
            str(resume_path),
            "--row-indices",
            row_indices,
            "--data-label",
            f"phase5-{peer_name}",
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
            "2",
        ],
        training_dir / "verify.log",
    )
    trained_adapter = training_dir / "checkpoints/step_0000002/adapter_model.safetensors"
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
    actual_start = read_state(initial_adapter)
    expected_start = read_state(round_base)
    start_error = max_abs_error(actual_start, expected_start)
    if start_error > TOLERANCE:
        raise RuntimeError(f"{peer_name} Round {round_number} start error {start_error}")
    training_summary = json.loads(
        (training_dir / "verification-summary.json").read_text(encoding="utf-8")
    )
    if training_summary.get("status") != "PASS":
        raise RuntimeError("local training verifier did not pass")

    payload, model_hash, raw_bytes, compressed_bytes = encode_delta(
        update_dir / "delta.safetensors"
    )
    final_loss = float(
        training_summary["checks"]["finite_metrics_and_loss_trend"]["detail"][
            "final_median_loss"
        ]
    )
    weights = {node_id: 1.0}
    if round_number == 2:
        weights = {node_id: 0.5, neighbors[0]: 0.5}
    output = {
        "weights": weights,
        "model_hash": model_hash,
        "validation_score": math.exp(-final_loss),
        "metadata": (
            f"phase5 {peer_name} round={round_number}; final_median_loss={final_loss:.6f}; "
            "encoding=zlib+safetensors"
        ),
        "compressed_delta": payload,
    }
    audit = {
        "schema_version": 1,
        "peer_name": peer_name,
        "round": round_number,
        "node_id": node_id,
        "neighbors": neighbors,
        "row_indices": [int(value) for value in row_indices.split(",")],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cpu_limit": os.environ.get("SLAKSHNA_CPU_LIMIT", ""),
        "round_base": str(round_base),
        "start_base_max_abs_error": start_error,
        "model_hash": model_hash,
        "delta_uncompressed_bytes": raw_bytes,
        "delta_compressed_bytes": compressed_bytes,
        "final_median_loss": final_loss,
        "aggregation": aggregation_manifest,
    }
    write_json(round_dir / "ml-bridge-audit.json", audit)
    write_json(round_dir / "ml-engine-output.json", output)
    write_json(state_path, {"rounds_completed": round_number, "node_id": node_id})
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
