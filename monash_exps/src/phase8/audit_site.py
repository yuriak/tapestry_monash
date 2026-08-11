#!/usr/bin/env python3
"""Validate one completed Phase 8 site and export a path-free audit document."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import TOLERANCE, read_state, state_sha256, write_json  # noqa: E402
from phase8.g0_bundle import contract_sha256, training_contract, verify_bundle  # noqa: E402
from phase8.protocol import FORMAT as ENVELOPE_FORMAT, VERSION as ENVELOPE_VERSION  # noqa: E402
from phase8.protocol import other_site, sha256_bytes  # noqa: E402


FORMAT = "slakshna-phase8-site-audit"
VERSION = 1
ROUNDS = range(1, 6)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.site_root.resolve()
    site_manifest = load_json(root / "site-manifest.json")
    site = site_manifest["site"]
    if site not in {"site-a", "site-b"}:
        raise RuntimeError(f"invalid site manifest: {site}")
    state = load_json(root / "bridge-state.json")
    if state.get("site") != site or state.get("invocations_completed") != 6:
        raise RuntimeError("bridge did not complete six invocations")
    if state.get("training_rounds_completed") != 5 or state.get("finalized") is not True:
        raise RuntimeError("bridge did not finalize five training rounds")
    node_id = state["node_id"]
    peer_node_id = state["peer_node_id"]
    if not node_id or not peer_node_id or node_id == peer_node_id:
        raise RuntimeError("invalid local/peer node identity in bridge state")

    for split in ("train", "validation"):
        shard = site_manifest["shard"][split]
        path = Path(shard["path"])
        if sha256_bytes(path.read_bytes()) != shard["sha256"]:
            raise RuntimeError(f"private {split} data hash mismatch")
        if len(path.read_text(encoding="utf-8").splitlines()) != shard["rows"]:
            raise RuntimeError(f"private {split} row count mismatch")
    contract = training_contract(root)
    contract_hash = contract_sha256(contract)
    g0 = verify_bundle(root / "global-0", root)
    if g0["training_contract_sha256"] != contract_hash:
        raise RuntimeError("G0/local contract mismatch")

    previous_hash = g0["adapter"]["state_sha256"]
    rounds: list[dict[str, Any]] = []
    for number in ROUNDS:
        round_dir = root / f"round-{number}"
        audit = load_json(round_dir / "ml-bridge-audit.json")
        outbound = load_json(round_dir / "outbound-manifest.json")
        config = yaml.safe_load(
            (round_dir / "local-training/resolved-config.yaml").read_text(encoding="utf-8")
        )
        summary = load_json(round_dir / "local-training/verification-summary.json")
        if audit["site"] != site or audit["node_id"] != node_id:
            raise RuntimeError(f"Round {number} local identity mismatch")
        if audit["training_round"] != number or audit["round_base_state_sha256"] != previous_hash:
            raise RuntimeError(f"Round {number} base/round mismatch")
        if audit["start_base_max_abs_error"] > TOLERANCE:
            raise RuntimeError(f"Round {number} did not load its global base")
        if config["training"]["num_epochs"] != 10 or config["training"]["max_steps"] != 720:
            raise RuntimeError(f"Round {number} training budget mismatch")
        if summary.get("status") != "PASS":
            raise RuntimeError(f"Round {number} training verification failed")
        losses = summary["checks"]["finite_metrics_and_loss_trend"]["detail"]
        if not all(math.isfinite(float(losses[key])) for key in (
            "initial_median_loss", "final_median_loss", "relative_loss_drop"
        )):
            raise RuntimeError(f"Round {number} loss is non-finite")
        delta_path = round_dir / "update/delta.safetensors"
        delta_state = read_state(delta_path)
        if outbound["format"] != ENVELOPE_FORMAT or outbound["version"] != ENVELOPE_VERSION:
            raise RuntimeError(f"Round {number} outbound envelope version mismatch")
        if outbound["sender_site"] != site or outbound["sender_node_id"] != node_id:
            raise RuntimeError(f"Round {number} outbound sender mismatch")
        if outbound["round"] != number or outbound["base_state_sha256"] != previous_hash:
            raise RuntimeError(f"Round {number} outbound provenance mismatch")
        if outbound["delta_file_sha256"] != sha256_bytes(delta_path.read_bytes()):
            raise RuntimeError(f"Round {number} outbound file mismatch")
        if outbound["delta_state_sha256"] != state_sha256(delta_state):
            raise RuntimeError(f"Round {number} outbound state mismatch")

        aggregation = load_json(root / f"global-{number}/aggregation-manifest.json")
        received = load_json(root / f"global-{number}/received-envelope-manifest.json")
        global_state = read_state(root / f"global-{number}/global_adapter.safetensors")
        global_hash = state_sha256(global_state)
        if aggregation["round"] != number or aggregation["base_state_sha256"] != previous_hash:
            raise RuntimeError(f"Round {number} aggregation provenance mismatch")
        if aggregation["own_delta_file_sha256"] != outbound["delta_file_sha256"]:
            raise RuntimeError(f"Round {number} aggregation used wrong local delta")
        if aggregation["peer_site"] != other_site(site) or aggregation["peer_node_id"] != peer_node_id:
            raise RuntimeError(f"Round {number} aggregation peer mismatch")
        if received["sender_site"] != other_site(site) or received["sender_node_id"] != peer_node_id:
            raise RuntimeError(f"Round {number} received sender mismatch")
        if received["round"] != number or received["base_state_sha256"] != previous_hash:
            raise RuntimeError(f"Round {number} received provenance mismatch")
        if aggregation["peer_delta_file_sha256"] != received["delta_file_sha256"]:
            raise RuntimeError(f"Round {number} received delta mismatch")
        if aggregation["global_state_sha256"] != global_hash:
            raise RuntimeError(f"Round {number} global state mismatch")
        rounds.append({
            "round": number,
            "base_state_sha256": previous_hash,
            "outbound": outbound,
            "received": received,
            "aggregation": {
                "algorithm": aggregation["algorithm"],
                "weights": aggregation["weights"],
                "aggregate_delta_state_sha256": aggregation["aggregate_delta_state_sha256"],
                "global_state_sha256": global_hash,
                "global_change_max_abs": aggregation["global_change_max_abs"],
                "client_delta_max_abs_difference": aggregation["client_delta_max_abs_difference"],
            },
            "training": {
                "local_epochs": 10,
                "optimizer_steps": 720,
                "initial_median_loss": losses["initial_median_loss"],
                "final_median_loss": losses["final_median_loss"],
                "relative_loss_drop": losses["relative_loss_drop"],
            },
        })
        previous_hash = global_hash

    shard = site_manifest["shard"]
    output = {
        "format": FORMAT,
        "version": VERSION,
        "status": "PASS",
        "site": site,
        "node_id": node_id,
        "peer_node_id": peer_node_id,
        "dataset": site_manifest["dataset"],
        "partition_type": site_manifest["partition_type"],
        "shard": {
            "categories": shard["categories"],
            "category_counts": shard["category_counts"],
            "train": {key: shard["train"][key] for key in ("rows", "source_indices", "sha256")},
            "validation": {
                key: shard["validation"][key] for key in ("rows", "source_indices", "sha256")
            },
        },
        "training_contract": contract,
        "training_contract_sha256": contract_hash,
        "g0": {
            "adapter_file_sha256": g0["adapter"]["file_sha256"],
            "adapter_state_sha256": g0["adapter"]["state_sha256"],
            "tensor_count": g0["adapter"]["tensor_count"],
            "parameter_count": g0["adapter"]["parameter_count"],
        },
        "rounds": rounds,
        "totals": {"training_rounds": 5, "local_epochs": 50, "optimizer_steps": 3600},
        "final_global_state_sha256": previous_hash,
    }
    write_json(args.output.resolve(), output)
    print(json.dumps({
        "status": "PASS", "site": site, "rounds": 5,
        "final_global_state_sha256": previous_hash,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
