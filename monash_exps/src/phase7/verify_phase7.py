#!/usr/bin/env python3
"""Fail-closed verifier for the minimal five-round Phase 7 experiment."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import TOLERANCE, max_abs_error, read_state, state_sha256, write_json  # noqa: E402


PEERS = ("peer-a", "peer-b")
ROUNDS = range(1, 6)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def check(name: str, action: Callable[[], Any]) -> Any:
        try:
            detail = action()
            checks[name] = {"status": "PASS", "detail": detail}
            return detail
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            checks[name] = {"status": "FAIL", "detail": message}
            failures.append(f"{name}: {message}")
            return None

    def verify_data() -> dict[str, Any]:
        manifest = load_json(root / "data-manifest.json")
        require(manifest["dataset"]["revision"] == "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "dataset revision mismatch")
        require(manifest["partition_type"] == "disjoint-category-non-iid", "partition type mismatch")
        require(manifest["train_rows_per_peer"] == 1152, "train row count mismatch")
        require(manifest["validation_rows_per_peer"] == 128, "validation row count mismatch")
        index_sets: dict[str, set[int]] = {}
        for peer in PEERS:
            shard = manifest["peers"][peer]
            train = shard["train"]
            validation = shard["validation"]
            require(sha256_file(Path(train["path"])) == train["sha256"], f"{peer} train hash mismatch")
            require(sha256_file(Path(validation["path"])) == validation["sha256"], f"{peer} validation hash mismatch")
            index_sets[peer] = set(train["source_indices"] + validation["source_indices"])
        require(index_sets["peer-a"].isdisjoint(index_sets["peer-b"]), "peer data overlaps")
        return {
            "dataset": manifest["dataset"],
            "partition_type": manifest["partition_type"],
            "train_rows_per_peer": 1152,
            "validation_rows_per_peer": 128,
            "categories": {peer: manifest["peers"][peer]["categories"] for peer in PEERS},
        }

    check("data_and_non_iid_partition", verify_data)

    def verify_topology() -> dict[str, Any]:
        topology = load_json(root / "cluster-topology.json")
        nodes = topology["nodes"]
        placement = topology.get("placement_mode", "two-node")
        addresses = [ipaddress.ip_address(nodes[peer]["ip"]) for peer in PEERS]
        if placement == "two-node":
            require(nodes["peer-a"]["hostname"] != nodes["peer-b"]["hostname"], "hosts are identical")
            require(addresses[0] != addresses[1], "node IPs are identical")
            require(not any(address.is_loopback for address in addresses), "loopback node IP")
        elif placement == "single-node-two-gpu":
            require(nodes["peer-a"]["hostname"] == nodes["peer-b"]["hostname"], "single-node hosts differ")
            require(all(address.is_loopback for address in addresses), "single-node transport is not loopback")
        else:
            raise RuntimeError(f"unknown placement mode: {placement}")
        expected_seed = (
            f"{nodes['peer-a']['endpoint_id']}@{nodes['peer-a']['ip']}:"
            f"{nodes['peer-a']['p2p_port']}"
        )
        require(topology["direct_seed"] == expected_seed, "direct seed mismatch")
        require(not any(topology["discovery"].values()), "public discovery or relay was enabled")
        gpu_ids: set[int] = set()
        for peer in PEERS:
            execution = load_json(root / peer / "node-execution.json")
            require(execution["hostname"] == nodes[peer]["hostname"], f"{peer} hostname mismatch")
            require(execution["gpu_count"] == 1, f"{peer} did not have one GPU")
            require("A100" in execution["gpu_name"].upper(), f"{peer} did not use an A100")
            require(execution["cuda_visible_devices"] == str(execution["gpu_id_configured"]), f"{peer} GPU id mismatch")
            gpu_ids.add(int(execution["gpu_id_configured"]))
        if placement == "single-node-two-gpu":
            require(gpu_ids == {0, 1}, f"single-node peers did not use distinct GPUs: {gpu_ids}")
        return {"placement_mode": placement, "nodes": nodes, "direct_seed": expected_seed, "discovery": topology["discovery"]}

    check("two_node_direct_transport", verify_topology)

    def verify_training() -> dict[str, Any]:
        detail: dict[str, Any] = {}
        for peer in PEERS:
            state = load_json(root / peer / "bridge-state.json")
            require(state["training_rounds_completed"] == 5, f"{peer} did not complete five rounds")
            require(state["invocations_completed"] == 6 and state["finalized"] is True, f"{peer} not finalized")
            rounds: list[dict[str, Any]] = []
            for round_number in ROUNDS:
                round_dir = root / peer / f"round-{round_number}"
                audit = load_json(round_dir / "ml-bridge-audit.json")
                summary = load_json(round_dir / "local-training/verification-summary.json")
                config = yaml.safe_load(
                    (round_dir / "local-training/resolved-config.yaml").read_text(encoding="utf-8")
                )
                require(summary["status"] == "PASS", f"{peer} Round {round_number} training failed")
                require(config["training"]["num_epochs"] == 10, "local epoch count mismatch")
                require(config["training"]["max_steps"] == 720, "optimizer step budget mismatch")
                checkpoint = summary["checks"]["checkpoint"]["detail"]
                require(checkpoint["step"] == 720, f"{peer} Round {round_number} checkpoint mismatch")
                require(audit["start_base_max_abs_error"] <= TOLERANCE, "round base load mismatch")
                require(math.isfinite(audit["initial_median_loss"]), "initial loss non-finite")
                require(math.isfinite(audit["final_median_loss"]), "final loss non-finite")
                require(audit["relative_loss_drop"] >= 0.0, "local loss did not decrease")
                rounds.append({
                    "round": round_number,
                    "initial_median_loss": audit["initial_median_loss"],
                    "final_median_loss": audit["final_median_loss"],
                    "relative_loss_drop": audit["relative_loss_drop"],
                    "optimizer_steps": 720,
                    "local_epochs": 10,
                })
            detail[peer] = {"total_local_epochs": 50, "total_optimizer_steps": 3600, "rounds": rounds}
        return detail

    check("five_round_fifty_epoch_training", verify_training)

    def verify_fedavg() -> dict[str, Any]:
        detail: list[dict[str, Any]] = []
        global0 = read_state(root / "global-0/global_adapter.safetensors")
        previous = global0
        for round_number in ROUNDS:
            paths = {
                peer: root / peer / f"global-{round_number}/global_adapter.safetensors"
                for peer in PEERS
            }
            states = {peer: read_state(path) for peer, path in paths.items()}
            peer_error = max_abs_error(states["peer-a"], states["peer-b"])
            require(peer_error <= TOLERANCE, f"G{round_number} differs between peers: {peer_error}")
            change = max_abs_error(previous, states["peer-a"])
            require(change > 0.0, f"G{round_number} did not change")
            manifests = {
                peer: load_json(root / peer / f"global-{round_number}/aggregation-manifest.json")
                for peer in PEERS
            }
            for peer in PEERS:
                other = "peer-b" if peer == "peer-a" else "peer-a"
                expected_hash = load_json(root / other / f"round-{round_number}/ml-engine-output.json")["model_hash"]
                require(manifests[peer]["peer_delta_hash"] == expected_hash, f"{peer} consumed wrong Round {round_number} update")
                require(manifests[peer]["base_state_sha256"] == state_sha256(previous), f"{peer} used wrong G{round_number - 1}")
            detail.append({
                "round": round_number,
                "peer_global_max_abs_error": peer_error,
                "global_change_max_abs": change,
                "global_state_sha256": state_sha256(states["peer-a"]),
            })
            previous = states["peer-a"]
        return {"algorithm": "0.5/0.5 dense FedAvg", "rounds": detail}

    check("per_round_fedavg_and_global_consistency", verify_fedavg)

    def verify_network_history() -> dict[str, Any]:
        canonical = None
        canonical_origins = None
        for peer in PEERS:
            updates = load_json(root / peer / "api-updates.json")["updates"]
            model_records = [record for record in updates if "ModelUpdate" in record["kind"]]
            review_records = [record for record in updates if "PeerReview" in record["kind"]]
            require(len(model_records) == 12, f"{peer} model record count is {len(model_records)}")
            require(len(review_records) == 20, f"{peer} review record count is {len(review_records)}")
            origins = {record["node_id"] for record in model_records}
            require(len(origins) == 2, f"{peer} model history does not contain two origins: {origins}")
            require(all(origin.startswith("slakshna1") for origin in origins), f"{peer} has invalid node IDs")
            model_counts = {
                origin: sum(record["node_id"] == origin for record in model_records)
                for origin in origins
            }
            review_counts = {
                origin: sum(record["node_id"] == origin for record in review_records)
                for origin in origins
            }
            review_targets = {
                record["kind"]["PeerReview"]["target_node"] for record in review_records
            }
            require(set(model_counts.values()) == {6}, f"{peer} model origin counts: {model_counts}")
            require(set(review_counts.values()) == {10}, f"{peer} review origin counts: {review_counts}")
            require(review_targets == origins, f"{peer} review targets do not cover both origins")
            hashes = sorted(record["hash"] for record in updates)
            if canonical is None:
                canonical = hashes
                canonical_origins = origins
            else:
                require(hashes == canonical, "peer history record sets did not converge")
                require(origins == canonical_origins, "peer history origin sets differ")
        return {
            "model_updates": 12,
            "peer_reviews": 20,
            "model_updates_per_origin": 6,
            "peer_reviews_per_origin": 10,
            "origins": sorted(canonical_origins or []),
            "histories_identical": True,
        }

    check("slakshna_update_exchange", verify_network_history)

    def verify_evaluation() -> dict[str, Any]:
        evaluations = [load_json(root / f"evaluation/global-{number}.json") for number in range(6)]
        for number, result in enumerate(evaluations):
            require(result["global_number"] == number, "evaluation global number mismatch")
            require(result["fresh_load_max_abs_error"] <= TOLERANCE, "fresh adapter load mismatch")
            require(math.isfinite(result["macro_negative_log_likelihood"]), "macro validation loss non-finite")
            for peer in PEERS:
                require(result["sites"][peer]["examples"] == 128, f"{peer} validation count mismatch")
                require(math.isfinite(result["sites"][peer]["negative_log_likelihood"]), f"{peer} validation loss non-finite")
        initial = evaluations[0]["macro_negative_log_likelihood"]
        final = evaluations[-1]["macro_negative_log_likelihood"]
        require(final < initial, f"final macro held-out loss {final} did not improve over base {initial}")
        inference = evaluations[-1].get("inference") or {}
        require(str(inference.get("generated_text", "")).strip(), "final fresh-process inference is empty")
        return {
            "global_macro_negative_log_likelihood": [result["macro_negative_log_likelihood"] for result in evaluations],
            "base_macro_negative_log_likelihood": initial,
            "final_macro_negative_log_likelihood": final,
            "relative_improvement": (initial - final) / initial,
            "final_site_metrics": evaluations[-1]["sites"],
            "fresh_process_inference": inference,
        }

    check("held_out_convergence_and_fresh_inference", verify_evaluation)

    def verify_cleanup() -> dict[str, Any]:
        for peer in PEERS:
            residue = load_json(root / peer / "process-residue.json")
            require(residue["residue"] == [], f"{peer} process residue: {residue['residue']}")
        require((root / "source-integrity.txt").read_text().splitlines()[-1] == "status=clean", "source integrity failed")
        return {"process_residue": {peer: [] for peer in PEERS}, "slakshna_source": "clean"}

    check("cleanup_and_source_integrity", verify_cleanup)

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "mode": "phase7",
        "checks": checks,
        "failures": failures,
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print("PHASE7 PASSED")


if __name__ == "__main__":
    main()
