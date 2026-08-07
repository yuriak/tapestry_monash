#!/usr/bin/env python3
"""Five-round Phase 7 training bridge plus one aggregation-only finalizer."""
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
import yaml
from safetensors.torch import load


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


TRAINING_ROUNDS = 5
LOCAL_EPOCHS = 10
EXPECTED_STEPS = 720


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return Path(value).resolve()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False
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
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("decoded peer delta is empty or non-finite")
    return raw, state


def atomic_torch_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.cpu().contiguous() for name, value in state.items()}, temporary)
    temporary.replace(path)


def global_paths(peer_root: Path, number: int) -> tuple[Path, Path]:
    root = peer_root / f"global-{number}"
    return root / "global_adapter.safetensors", root / "global_adapter.pth"


def aggregate_round(
    *,
    round_number: int,
    peer_name: str,
    peer_id: str,
    peer_root: Path,
    artifact_root: Path,
    data_dir: Path,
    global0: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    own_delta_path = peer_root / f"round-{round_number}/update/delta.safetensors"
    own_delta = read_state(own_delta_path)
    payload_path = data_dir / "network_deltas" / f"{peer_id}_delta.b64"
    if not payload_path.is_file():
        raise RuntimeError(f"Rust did not stage the peer delta: {payload_path}")
    payload = payload_path.read_text(encoding="utf-8").strip()
    peer_raw, peer_delta = decode_delta(payload)
    peer_hash = hashlib.sha256(peer_raw).hexdigest()
    other_name = "peer-b" if peer_name == "peer-a" else "peer-a"
    expected_output_path = artifact_root / other_name / f"round-{round_number}/ml-engine-output.json"
    if not expected_output_path.is_file():
        raise RuntimeError(f"peer Round {round_number} output is unavailable: {expected_output_path}")
    expected_peer_hash = json.loads(expected_output_path.read_text(encoding="utf-8"))["model_hash"]
    if peer_hash != expected_peer_hash:
        raise RuntimeError(
            f"received peer delta is not Round {round_number}: {peer_hash} != {expected_peer_hash}"
        )
    require_compatible(own_delta, peer_delta, f"Round {round_number} own/peer delta")

    if round_number == 1:
        base_path = global0
    else:
        base_path, _ = global_paths(peer_root, round_number - 1)
    base = read_state(base_path)
    require_compatible(base, own_delta, f"G{round_number - 1}/own delta")
    aggregate_delta = {
        name: (own_delta[name] * 0.5 + peer_delta[name] * 0.5).float().contiguous()
        for name in base
    }
    global_state = {
        name: (base[name] + aggregate_delta[name]).float().contiguous() for name in base
    }
    global_path, resume_path = global_paths(peer_root, round_number)
    aggregation_dir = global_path.parent
    aggregation_dir.mkdir(parents=True, exist_ok=True)
    received_path = aggregation_dir / "received_peer_delta.safetensors"
    temporary = received_path.with_suffix(received_path.suffix + ".tmp")
    temporary.write_bytes(peer_raw)
    temporary.replace(received_path)
    atomic_safetensors(aggregate_delta, aggregation_dir / "aggregate_delta.safetensors")
    atomic_safetensors(global_state, global_path)
    atomic_torch_state(global_state, resume_path)
    manifest = {
        "schema_version": 1,
        "round": round_number,
        "algorithm": "equal-weight-dense-fedavg",
        "weights": {"self": 0.5, "peer": 0.5},
        "base_path": str(base_path),
        "base_state_sha256": state_sha256(base),
        "own_delta_path": str(own_delta_path),
        "own_delta_state_sha256": state_sha256(own_delta),
        "peer_id": peer_id,
        "peer_transport_path": str(payload_path),
        "peer_delta_hash": peer_hash,
        "expected_peer_delta_hash": expected_peer_hash,
        "peer_delta_state_sha256": state_sha256(peer_delta),
        "aggregate_delta_state_sha256": state_sha256(aggregate_delta),
        "global_state_sha256": state_sha256(global_state),
        "global_change_max_abs": max_abs_error(base, global_state),
        "client_delta_max_abs_difference": max_abs_error(own_delta, peer_delta),
        "tensor_count": len(global_state),
        "parameter_count": sum(value.numel() for value in global_state.values()),
        "tolerance": TOLERANCE,
    }
    write_json(aggregation_dir / "aggregation-manifest.json", manifest)
    return global_path, resume_path, manifest


def resolve_round_config(
    source: Path,
    destination: Path,
    resume_path: Path,
    checkpoint_dir: Path,
    run_name: str,
) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["lora"]["resume_path"] = str(resume_path)
    config["training"]["num_epochs"] = LOCAL_EPOCHS
    config["training"]["max_steps"] = EXPECTED_STEPS
    config["checkpoint"]["save_dir"] = str(checkpoint_dir)
    config["logging"]["run_name"] = run_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def train_round(
    *,
    round_number: int,
    peer_name: str,
    node_id: str,
    neighbors: list[str],
    peer_root: Path,
    experiment_root: Path,
    bootstrap_config: Path,
    round_base: Path,
    resume_path: Path,
    run_id: str,
    aggregation: dict[str, Any] | None,
) -> dict[str, Any]:
    round_dir = peer_root / f"round-{round_number}"
    training_dir = round_dir / "local-training"
    update_dir = round_dir / "update"
    training_dir.mkdir(parents=True, exist_ok=False)
    update_dir.mkdir(parents=True, exist_ok=False)
    config_path = training_dir / "resolved-config.yaml"
    checkpoint_dir = training_dir / "checkpoints"
    resolve_round_config(
        bootstrap_config,
        config_path,
        resume_path,
        checkpoint_dir,
        f"{run_id}-{peer_name}-round{round_number}",
    )
    initial_adapter = training_dir / "initial_adapter.safetensors"
    python = sys.executable
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/launch_training.py"),
            "--config", str(config_path),
            "--num-workers", "1",
            "--run-dir", str(training_dir),
            "--capture-initial-adapter", str(initial_adapter),
        ],
        training_dir / "train.log",
    )
    run_logged(
        [
            python,
            str(experiment_root / "src/phase1/verify_training.py"),
            "--mode", "phase1a",
            "--run-dir", str(training_dir),
            "--expected-workers", "1",
            "--expected-final-step", str(EXPECTED_STEPS),
        ],
        training_dir / "verify.log",
    )
    checkpoints = sorted(checkpoint_dir.glob("step_*"))
    if not checkpoints:
        raise RuntimeError(f"Round {round_number} produced no checkpoint")
    trained_adapter = checkpoints[-1] / "adapter_model.safetensors"
    if not trained_adapter.is_file():
        raise RuntimeError(f"Round {round_number} final adapter is missing")
    run_logged(
        [
            python,
            str(experiment_root / "src/phase2/adapter_delta.py"),
            "create",
            "--initial", str(initial_adapter),
            "--trained", str(trained_adapter),
            "--config", str(config_path),
            "--output-dir", str(update_dir),
        ],
        update_dir / "create.log",
    )
    start = read_state(initial_adapter)
    expected_start = read_state(round_base)
    start_error = max_abs_error(start, expected_start)
    if start_error > TOLERANCE:
        raise RuntimeError(f"{peer_name} Round {round_number} start error {start_error}")
    summary = json.loads((training_dir / "verification-summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError(f"{peer_name} Round {round_number} training verification failed")
    losses = summary["checks"]["finite_metrics_and_loss_trend"]["detail"]
    payload, model_hash, raw_bytes, compressed_bytes = encode_delta(
        update_dir / "delta.safetensors"
    )
    weights = {node_id: 1.0} if round_number == 1 else {node_id: 0.5, neighbors[0]: 0.5}
    output = {
        "weights": weights,
        "model_hash": model_hash,
        "validation_score": math.exp(-float(losses["final_median_loss"])),
        "metadata": (
            f"phase7 {peer_name} training_round={round_number}; local_epochs={LOCAL_EPOCHS}; "
            f"optimizer_steps={EXPECTED_STEPS}; encoding=zlib+safetensors"
        ),
        "compressed_delta": payload,
    }
    audit = {
        "schema_version": 1,
        "peer_name": peer_name,
        "node_id": node_id,
        "training_round": round_number,
        "local_epochs": LOCAL_EPOCHS,
        "optimizer_steps": EXPECTED_STEPS,
        "neighbors": neighbors,
        "round_base": str(round_base),
        "round_base_state_sha256": state_sha256(expected_start),
        "start_base_max_abs_error": start_error,
        "model_hash": model_hash,
        "delta_uncompressed_bytes": raw_bytes,
        "delta_compressed_bytes": compressed_bytes,
        "initial_median_loss": losses["initial_median_loss"],
        "final_median_loss": losses["final_median_loss"],
        "relative_loss_drop": losses["relative_loss_drop"],
        "aggregation_before_training": aggregation,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_json(round_dir / "ml-bridge-audit.json", audit)
    write_json(round_dir / "ml-engine-output.json", output)
    return output


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ml_engine.py <node-id> [peer-id ...]")
    node_id = sys.argv[1]
    neighbors = sys.argv[2:]
    peer_name = os.environ.get("SLAKSHNA_PHASE7_PEER_NAME", "")
    if peer_name not in {"peer-a", "peer-b"}:
        raise RuntimeError(f"invalid SLAKSHNA_PHASE7_PEER_NAME: {peer_name}")
    artifact_root = required_path("SLAKSHNA_PHASE7_ARTIFACT_ROOT")
    experiment_root = required_path("SLAKSHNA_EXPERIMENT_ROOT")
    global0 = required_path("SLAKSHNA_PHASE7_GLOBAL0")
    global0_resume = required_path("SLAKSHNA_PHASE7_GLOBAL0_RESUME")
    data_dir = required_path("IIITD_DATA_DIR")
    run_id = os.environ.get("SLAKSHNA_PHASE7_RUN_ID", artifact_root.name)
    peer_root = artifact_root / peer_name
    bootstrap_config = artifact_root / "bootstrap" / peer_name / "resolved-config.yaml"
    state_path = peer_root / "bridge-state.json"
    state = {"invocations_completed": 0, "training_rounds_completed": 0, "finalized": False}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    invocation = int(state.get("invocations_completed", 0)) + 1
    if invocation not in range(1, TRAINING_ROUNDS + 2):
        raise RuntimeError(f"Phase 7 bridge refuses invocation {invocation}")
    if invocation == 1 and neighbors:
        raise RuntimeError(f"first invocation unexpectedly received peer history: {neighbors}")
    if invocation > 1 and len(neighbors) != 1:
        raise RuntimeError(f"invocation {invocation} requires exactly one peer, got {neighbors}")
    peer_root.mkdir(parents=True, exist_ok=True)

    aggregation = None
    if invocation == 1:
        round_base, resume_path = global0, global0_resume
    else:
        round_base, resume_path, aggregation = aggregate_round(
            round_number=invocation - 1,
            peer_name=peer_name,
            peer_id=neighbors[0],
            peer_root=peer_root,
            artifact_root=artifact_root,
            data_dir=data_dir,
            global0=global0,
        )

    if invocation <= TRAINING_ROUNDS:
        output = train_round(
            round_number=invocation,
            peer_name=peer_name,
            node_id=node_id,
            neighbors=neighbors,
            peer_root=peer_root,
            experiment_root=experiment_root,
            bootstrap_config=bootstrap_config,
            round_base=round_base,
            resume_path=resume_path,
            run_id=run_id,
            aggregation=aggregation,
        )
        next_state = {
            "invocations_completed": invocation,
            "training_rounds_completed": invocation,
            "finalized": False,
            "node_id": node_id,
        }
    else:
        prior = json.loads(
            (peer_root / f"round-{TRAINING_ROUNDS}/ml-engine-output.json").read_text(
                encoding="utf-8"
            )
        )
        output = {
            **prior,
            "weights": {node_id: 0.5, neighbors[0]: 0.5},
            "metadata": (
                f"phase7 {peer_name} aggregation-only finalizer; global=G{TRAINING_ROUNDS}"
            ),
        }
        write_json(
            peer_root / "finalization-audit.json",
            {
                "schema_version": 1,
                "invocation": invocation,
                "training_performed": False,
                "final_global_path": str(round_base),
                "final_global_state_sha256": state_sha256(read_state(round_base)),
                "aggregation": aggregation,
            },
        )
        next_state = {
            "invocations_completed": invocation,
            "training_rounds_completed": TRAINING_ROUNDS,
            "finalized": True,
            "node_id": node_id,
        }
    write_json(state_path, next_state)
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
