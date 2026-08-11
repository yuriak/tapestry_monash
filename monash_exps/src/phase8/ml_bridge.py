#!/usr/bin/env python3
"""Site-local Phase 8 bridge: five training rounds plus a final aggregation."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


_experiment_root = os.environ.get("SLAKSHNA_EXPERIMENT_ROOT")
if not _experiment_root:
    raise RuntimeError("required environment variable is unset: SLAKSHNA_EXPERIMENT_ROOT")
EXPERIMENT_ROOT = Path(_experiment_root).resolve()
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import (  # noqa: E402
    TOLERANCE,
    atomic_safetensors,
    max_abs_error,
    read_state,
    state_sha256,
    write_json,
)
from phase8.g0_bundle import verify_bundle  # noqa: E402
from phase8.protocol import (  # noqa: E402
    aggregate_equal_weight,
    decode_delta_payload,
    encode_delta_file,
    other_site,
    sha256_bytes,
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


def atomic_torch_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.cpu().contiguous() for name, value in state.items()}, temporary)
    temporary.replace(path)


def global_paths(site_root: Path, number: int) -> tuple[Path, Path]:
    root = site_root / f"global-{number}"
    return root / "global_adapter.safetensors", root / "global_adapter.pth"


def aggregate_round(
    *,
    round_number: int,
    site: str,
    peer_node_id: str,
    site_root: Path,
    data_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    own_delta_path = site_root / f"round-{round_number}/update/delta.safetensors"
    own_manifest_path = site_root / f"round-{round_number}/outbound-manifest.json"
    own_manifest = json.loads(own_manifest_path.read_text(encoding="utf-8"))
    own_delta = read_state(own_delta_path)
    base_path, _ = global_paths(site_root, round_number - 1)
    base = read_state(base_path)
    base_hash = state_sha256(base)
    if own_manifest["round"] != round_number:
        raise RuntimeError("local outbound manifest round mismatch")
    if own_manifest["base_state_sha256"] != base_hash:
        raise RuntimeError("local outbound manifest base mismatch")
    if own_manifest["delta_file_sha256"] != sha256_bytes(own_delta_path.read_bytes()):
        raise RuntimeError("local outbound delta file hash mismatch")

    payload_path = data_dir / "network_deltas" / f"{peer_node_id}_delta.b64"
    if not payload_path.is_file():
        raise RuntimeError(f"Rust did not stage the peer delta: {payload_path}")
    decoded = decode_delta_payload(
        payload_path.read_text(encoding="utf-8").strip(),
        expected_sender_site=other_site(site),
        expected_sender_node_id=peer_node_id,
        expected_round=round_number,
        expected_base_state_sha256=base_hash,
    )
    aggregate_delta, global_state = aggregate_equal_weight(base, own_delta, decoded.state)
    global_path, resume_path = global_paths(site_root, round_number)
    aggregation_dir = global_path.parent
    aggregation_dir.mkdir(parents=True, exist_ok=True)
    received_path = aggregation_dir / "received_peer_delta.safetensors"
    temporary = received_path.with_suffix(received_path.suffix + ".tmp")
    temporary.write_bytes(decoded.raw)
    temporary.replace(received_path)
    atomic_safetensors(aggregate_delta, aggregation_dir / "aggregate_delta.safetensors")
    atomic_safetensors(global_state, global_path)
    atomic_torch_state(global_state, resume_path)
    peer_manifest = {key: value for key, value in decoded.envelope.items() if key != "data"}
    peer_manifest["transport_node_id"] = peer_node_id
    write_json(aggregation_dir / "received-envelope-manifest.json", peer_manifest)
    manifest = {
        "schema_version": 1,
        "round": round_number,
        "site": site,
        "algorithm": "equal-weight-dense-fedavg",
        "weights": {site: 0.5, other_site(site): 0.5},
        "base_state_sha256": base_hash,
        "own_delta_file_sha256": own_manifest["delta_file_sha256"],
        "own_delta_state_sha256": state_sha256(own_delta),
        "peer_site": other_site(site),
        "peer_node_id": peer_node_id,
        "peer_delta_file_sha256": decoded.envelope["delta_file_sha256"],
        "peer_delta_state_sha256": decoded.envelope["delta_state_sha256"],
        "aggregate_delta_state_sha256": state_sha256(aggregate_delta),
        "global_state_sha256": state_sha256(global_state),
        "global_change_max_abs": max_abs_error(base, global_state),
        "client_delta_max_abs_difference": max_abs_error(own_delta, decoded.state),
        "tensor_count": len(global_state),
        "parameter_count": sum(value.numel() for value in global_state.values()),
        "tolerance": TOLERANCE,
    }
    write_json(aggregation_dir / "aggregation-manifest.json", manifest)
    return global_path, resume_path, manifest


def resolve_round_config(
    source: Path, destination: Path, resume_path: Path, checkpoint_dir: Path, run_name: str
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
    site: str,
    node_id: str,
    peer_node_id: str | None,
    site_root: Path,
    bootstrap_config: Path,
    round_base: Path,
    resume_path: Path,
    run_id: str,
    aggregation: dict[str, Any] | None,
) -> dict[str, Any]:
    round_dir = site_root / f"round-{round_number}"
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
        f"{run_id}-{site}-round{round_number}",
    )
    initial_adapter = training_dir / "initial_adapter.safetensors"
    run_logged(
        [
            sys.executable, str(EXPERIMENT_ROOT / "src/phase1/launch_training.py"),
            "--config", str(config_path), "--num-workers", "1",
            "--run-dir", str(training_dir),
            "--capture-initial-adapter", str(initial_adapter),
        ],
        training_dir / "train.log",
    )
    run_logged(
        [
            sys.executable, str(EXPERIMENT_ROOT / "src/phase1/verify_training.py"),
            "--mode", "phase1a", "--run-dir", str(training_dir),
            "--expected-workers", "1", "--expected-final-step", str(EXPECTED_STEPS),
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
            sys.executable, str(EXPERIMENT_ROOT / "src/phase2/adapter_delta.py"), "create",
            "--initial", str(initial_adapter), "--trained", str(trained_adapter),
            "--config", str(config_path), "--output-dir", str(update_dir),
        ],
        update_dir / "create.log",
    )
    expected_start = read_state(round_base)
    start_error = max_abs_error(read_state(initial_adapter), expected_start)
    if start_error > TOLERANCE:
        raise RuntimeError(f"{site} Round {round_number} start error {start_error}")
    summary = json.loads((training_dir / "verification-summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError(f"{site} Round {round_number} training verification failed")
    losses = summary["checks"]["finite_metrics_and_loss_trend"]["detail"]
    payload, outbound = encode_delta_file(
        update_dir / "delta.safetensors",
        sender_site=site,
        sender_node_id=node_id,
        round_number=round_number,
        base_state_sha256=state_sha256(expected_start),
    )
    write_json(round_dir / "outbound-manifest.json", outbound)
    weights = {node_id: 1.0} if peer_node_id is None else {node_id: 0.5, peer_node_id: 0.5}
    output = {
        "weights": weights,
        "model_hash": outbound["delta_file_sha256"],
        "validation_score": math.exp(-float(losses["final_median_loss"])),
        "metadata": (
            f"phase8 {site} round={round_number}; local_epochs={LOCAL_EPOCHS}; "
            f"optimizer_steps={EXPECTED_STEPS}; envelope=v1"
        ),
        "compressed_delta": payload,
    }
    audit = {
        "schema_version": 1,
        "site": site,
        "node_id": node_id,
        "peer_node_id": peer_node_id,
        "training_round": round_number,
        "local_epochs": LOCAL_EPOCHS,
        "optimizer_steps": EXPECTED_STEPS,
        "round_base_state_sha256": state_sha256(expected_start),
        "start_base_max_abs_error": start_error,
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
    site = os.environ.get("SLAKSHNA_PHASE8_SITE", "")
    if site not in {"site-a", "site-b"}:
        raise RuntimeError(f"invalid SLAKSHNA_PHASE8_SITE: {site}")
    site_root = required_path("SLAKSHNA_PHASE8_SITE_ROOT")
    data_dir = required_path("IIITD_DATA_DIR")
    run_id = os.environ.get("SLAKSHNA_PHASE8_RUN_ID", site_root.name)
    bootstrap_config = site_root / "bootstrap/resolved-config.yaml"
    verify_bundle(site_root / "global-0", site_root)
    state_path = site_root / "bridge-state.json"
    state: dict[str, Any] = {
        "invocations_completed": 0,
        "training_rounds_completed": 0,
        "finalized": False,
        "site": site,
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("site") != site:
        raise RuntimeError("bridge state belongs to another site")
    if state.get("node_id") not in (None, node_id):
        raise RuntimeError("local Slakshna node ID changed during the run")
    invocation = int(state.get("invocations_completed", 0)) + 1
    if invocation not in range(1, TRAINING_ROUNDS + 2):
        raise RuntimeError(f"Phase 8 bridge refuses invocation {invocation}")
    if invocation == 1 and neighbors:
        raise RuntimeError(f"first invocation unexpectedly received peer history: {neighbors}")
    if invocation > 1 and len(neighbors) != 1:
        raise RuntimeError(f"invocation {invocation} requires exactly one peer, got {neighbors}")
    prior_peer_id = state.get("peer_node_id")
    if prior_peer_id is not None and neighbors[0] != prior_peer_id:
        raise RuntimeError("remote Slakshna node ID changed during the run")
    peer_node_id = neighbors[0] if neighbors else None

    aggregation = None
    if invocation == 1:
        round_base, resume_path = global_paths(site_root, 0)
    else:
        round_base, resume_path, aggregation = aggregate_round(
            round_number=invocation - 1,
            site=site,
            peer_node_id=peer_node_id or "",
            site_root=site_root,
            data_dir=data_dir,
        )
    if invocation <= TRAINING_ROUNDS:
        output = train_round(
            round_number=invocation,
            site=site,
            node_id=node_id,
            peer_node_id=peer_node_id,
            site_root=site_root,
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
            "site": site,
            "node_id": node_id,
            "peer_node_id": peer_node_id,
        }
    else:
        prior = json.loads(
            (site_root / f"round-{TRAINING_ROUNDS}/ml-engine-output.json").read_text(
                encoding="utf-8"
            )
        )
        output = {
            **prior,
            "weights": {node_id: 0.5, peer_node_id: 0.5},
            "metadata": f"phase8 {site} aggregation-only finalizer; global=G{TRAINING_ROUNDS}",
        }
        write_json(site_root / "finalization-audit.json", {
            "schema_version": 1,
            "invocation": invocation,
            "training_performed": False,
            "final_global_state_sha256": state_sha256(read_state(round_base)),
            "aggregation": aggregation,
        })
        next_state = {
            "invocations_completed": invocation,
            "training_rounds_completed": TRAINING_ROUNDS,
            "finalized": True,
            "site": site,
            "node_id": node_id,
            "peer_node_id": peer_node_id,
        }
    write_json(state_path, next_state)
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
