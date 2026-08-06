#!/usr/bin/env python3
"""Fail-closed verifier for the two-peer, two-round Phase 5 lifecycle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.adapter_delta import TOLERANCE, max_abs_error, read_state, write_json  # noqa: E402


PEERS = ("peer-a", "peer-b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def kind(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = record.get("kind", {})
    return value.get(name) if isinstance(value, dict) else None


def record_hash(record: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(record["node_id"].encode())
    digest.update(record["prev_hash"].encode())
    model = kind(record, "ModelUpdate")
    review = kind(record, "PeerReview")
    if model is not None:
        digest.update(b"model_update")
        digest.update(model["delta_hash"].encode())
        digest.update(model["compressed_delta"].encode())
    elif review is not None:
        import struct

        digest.update(b"peer_review")
        digest.update(review["target_node"].encode())
        digest.update(review["update_hash"].encode())
        digest.update(struct.pack("<d", float(review["loss_drop"])))
        digest.update(struct.pack("<d", float(review["trust_score"])))
    else:
        raise RuntimeError(f"unknown Rust record kind: {record.get('kind')}")
    return digest.hexdigest()


def decode_payload(payload: str) -> tuple[bytes, dict[str, torch.Tensor]]:
    raw = zlib.decompress(base64.b64decode(payload, validate=True))
    state = {name: value.float().contiguous() for name, value in load(raw).items()}
    require(state, "transport payload decoded to an empty state")
    require(all(torch.isfinite(value).all() for value in state.values()), "non-finite transport delta")
    return raw, state


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="phase5")
    parser.add_argument("--expected-visible-a", default="0")
    parser.add_argument("--expected-visible-b", default="1")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    global0 = read_state(root / "global-0/global_adapter.safetensors")

    audits: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    outputs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    local_details: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    node_ids: dict[str, str] = {}
    for peer in PEERS:
        state = load_json(root / peer / "bridge-state.json")
        require(state.get("rounds_completed") == 2, f"{peer} did not complete exactly two rounds")
        for round_number in (1, 2):
            round_dir = root / peer / f"round-{round_number}"
            audit = load_json(round_dir / "ml-bridge-audit.json")
            output = load_json(round_dir / "ml-engine-output.json")
            summary = load_json(round_dir / "local-training/verification-summary.json")
            evidence = load_json(round_dir / "local-training/worker-evidence/rank-0-start.json")
            require(audit["round"] == round_number, f"{peer} round audit mismatch")
            require(summary.get("status") == "PASS", f"{peer} Round {round_number} training failed")
            require(
                summary["checks"]["checkpoint"]["detail"]["step"] == 2,
                f"{peer} Round {round_number} did not stop at step 2",
            )
            require(evidence.get("resume_step") == 0, f"{peer} Round {round_number} reused DCP state")
            require(evidence.get("resume_checkpoint") is None, f"{peer} found stale checkpoint state")
            require(audit["gpu_count"] == 1, f"{peer} did not see exactly one GPU")
            expected_visible = (
                args.expected_visible_a if peer == "peer-a" else args.expected_visible_b
            )
            require(
                audit["cuda_visible_devices"] == expected_visible,
                f"{peer} used CUDA_VISIBLE_DEVICES={audit['cuda_visible_devices']}",
            )
            expected_rows = [0, 2, 4, 6] if peer == "peer-a" else [1, 3, 5, 7]
            require(audit["row_indices"] == expected_rows, f"{peer} data shard mismatch")
            require(audit["start_base_max_abs_error"] <= TOLERANCE, f"{peer} start mismatch")
            raw, delta = decode_payload(output["compressed_delta"])
            require(hashlib.sha256(raw).hexdigest() == output["model_hash"], "bridge hash mismatch")
            local_delta = (round_dir / "update/delta.safetensors").read_bytes()
            require(raw == local_delta, f"{peer} Round {round_number} payload differs from local delta")
            require(any(torch.count_nonzero(value) for value in delta.values()), "zero local delta")
            audits[peer][round_number] = audit
            outputs[peer][round_number] = output
            node_ids[peer] = audit["node_id"]
            local_details[peer][round_number] = {
                "gpu": audit["gpu_name"],
                "cuda_visible_devices": audit["cuda_visible_devices"],
                "checkpoint_step": 2,
                "resume_step": 0,
                "final_median_loss": audit["final_median_loss"],
                "model_hash": output["model_hash"],
            }
    require(node_ids["peer-a"] != node_ids["peer-b"], "peer federation identities are identical")

    status = {peer: load_json(root / peer / "api-status.json") for peer in PEERS}
    peers_api = {peer: load_json(root / peer / "api-peers.json") for peer in PEERS}
    endpoints = {peer: status[peer].get("endpoint_id") for peer in PEERS}
    require(all(endpoints.values()), f"missing endpoint identity: {endpoints}")
    require(endpoints["peer-a"] != endpoints["peer-b"], "transport EndpointIds are identical")
    for peer, remote in (("peer-a", "peer-b"), ("peer-b", "peer-a")):
        require(
            endpoints[remote] in peers_api[peer].get("connected", []),
            f"{peer} API does not show {remote} as connected",
        )
        require(
            endpoints[remote] in peers_api[peer].get("known", []),
            f"{peer} did not persist {remote} in known peers",
        )

    api_records: dict[str, list[dict[str, Any]]] = {}
    for peer in PEERS:
        payload = load_json(root / peer / "api-updates.json")
        require(payload.get("success") is True, f"{peer} updates API failed")
        records = payload.get("updates", [])
        require(len(records) == 8, f"{peer} expected 8 history records, got {len(records)}")
        for record in records:
            require(record.get("hash") == record_hash(record), f"invalid record hash on {peer}")
        api_records[peer] = records
    require(
        sorted(record["hash"] for record in api_records["peer-a"])
        == sorted(record["hash"] for record in api_records["peer-b"]),
        "the two peer histories did not converge",
    )

    canonical_records = api_records["peer-a"]
    by_origin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical_records:
        by_origin[record["node_id"]].append(record)
    require(set(by_origin) == set(node_ids.values()), f"unexpected history origins: {set(by_origin)}")
    model_record_hashes: dict[str, list[str]] = {}
    for peer in PEERS:
        origin = node_ids[peer]
        records = by_origin[origin]
        require(len(records) == 4, f"{peer} origin has {len(records)} records")
        model_records = [record for record in records if kind(record, "ModelUpdate") is not None]
        review_records = [record for record in records if kind(record, "PeerReview") is not None]
        require(len(model_records) == 2 and len(review_records) == 2, f"{peer} history kinds mismatch")
        for index, model_record in enumerate(model_records, start=1):
            model = kind(model_record, "ModelUpdate")
            expected = outputs[peer][index]
            require(model["delta_hash"] != "error_hash", f"{peer} recorded error_hash")
            require(model["delta_hash"] == expected["model_hash"], f"{peer} Round {index} hash mismatch")
            require(
                model["compressed_delta"] == expected["compressed_delta"],
                f"{peer} Round {index} payload mismatch",
            )
        model_record_hashes[origin] = [record["hash"] for record in model_records]

    review_target_rounds: list[int] = []
    for record in canonical_records:
        review = kind(record, "PeerReview")
        if review is None:
            continue
        require(review["target_node"] in model_record_hashes, "review targets an unknown peer")
        require(
            review["update_hash"] in model_record_hashes[review["target_node"]],
            "review does not target a recorded update from the claimed peer",
        )
        review_target_rounds.append(
            model_record_hashes[review["target_node"]].index(review["update_hash"]) + 1
        )
        require(float(review["trust_score"]) == 0.5, "unexpected Phase 5 trust score")

    for peer, remote in (("peer-a", "peer-b"), ("peer-b", "peer-a")):
        aggregation = audits[peer][2]["aggregation"]
        require(aggregation is not None, f"{peer} has no Round 2 aggregation evidence")
        require(aggregation["peer_id"] == node_ids[remote], f"{peer} aggregated wrong peer")
        require(
            aggregation["peer_delta_hash"] == outputs[remote][1]["model_hash"],
            f"{peer} did not aggregate {remote}'s transported Round 1 delta",
        )
        received = root / peer / "round-2/received-peer-delta.safetensors"
        remote_delta = root / remote / "round-1/update/delta.safetensors"
        require(received.read_bytes() == remote_delta.read_bytes(), "staged Gossip delta byte mismatch")

    global1_a = read_state(root / "peer-a/round-2/aggregation/global_adapter.safetensors")
    global1_b = read_state(root / "peer-b/round-2/aggregation/global_adapter.safetensors")
    global1_error = max_abs_error(global1_a, global1_b)
    require(global1_error <= TOLERANCE, f"peer G1 states differ: {global1_error}")
    global1_change = max_abs_error(global0, global1_a)
    require(global1_change > 0.0, "G1 did not change from G0")
    for peer in PEERS:
        start = read_state(root / peer / "round-2/local-training/initial_adapter.safetensors")
        require(max_abs_error(start, global1_a) <= TOLERANCE, f"{peer} did not train Round 2 from G1")

    leaderboard: dict[str, dict[str, float]] = {}
    for peer in PEERS:
        response = load_json(root / peer / "api-leaderboard.json")
        scores = {row["node"]: float(row["trust_score"]) for row in response["leaderboard"]}
        require(set(scores) == set(node_ids.values()), f"{peer} leaderboard identities mismatch")
        require(all(abs(score - 1.0) <= 1e-12 for score in scores.values()), "leaderboard score mismatch")
        leaderboard[peer] = scores

    recovery_status = load_json(root / "peer-a/recovery-status.json")
    recovery_peers = load_json(root / "peer-a/recovery-peers.json")
    recovery_updates = load_json(root / "peer-a/recovery-updates.json")
    require(recovery_status.get("endpoint_id") == endpoints["peer-a"], "Peer A identity changed")
    require(
        endpoints["peer-b"] in recovery_peers.get("known", []),
        "Peer A did not recover Peer B from known-peer state",
    )
    require(recovery_updates.get("updates") == [], "in-memory update history unexpectedly survived restart")

    for peer in PEERS:
        rust_log = (root / peer / "rust-node.log").read_text(encoding="utf-8", errors="replace")
        require("Python ML Engine failed" not in rust_log, f"{peer} Rust log reports ML failure")
        require(rust_log.count("Local Training Complete") == 2, f"{peer} did not log two completions")

    summary = {
        "status": "PASS",
        "mode": args.mode,
        "checks": {
            "p2p_mesh": {
                "status": "PASS",
                "detail": {"endpoint_ids": endpoints, "bidirectional_connected": True},
            },
            "local_training": {"status": "PASS", "detail": local_details},
            "gossip_exchange": {
                "status": "PASS",
                "detail": {
                    "converged_history_records_per_peer": 8,
                    "model_updates_per_origin": 2,
                    "peer_reviews_per_origin": 2,
                },
            },
            "round_1_fedavg": {
                "status": "PASS",
                "detail": {
                    "global1_peer_max_abs_error": global1_error,
                    "global0_to_global1_max_abs_change": global1_change,
                    "weights": {"peer-a": 0.5, "peer-b": 0.5},
                },
            },
            "trust_reviews": {
                "status": "PASS",
                "detail": {
                    "leaderboards": leaderboard,
                    "review_target_rounds": review_target_rounds,
                },
            },
            "state_recovery": {
                "status": "PASS",
                "detail": {
                    "endpoint_id_preserved": True,
                    "known_peer_preserved": True,
                    "update_history_records_after_restart": 0,
                },
            },
        },
        "failures": [],
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"{args.mode.upper()} PASSED")


if __name__ == "__main__":
    main()
