#!/usr/bin/env python3
"""Cross-check two independently exported Phase 8 site audits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import write_json  # noqa: E402
from phase8.audit_site import FORMAT, VERSION  # noqa: E402


TOP_LEVEL_FIELDS = {
    "format", "version", "status", "site", "node_id", "peer_node_id", "dataset",
    "partition_type", "shard", "training_contract", "training_contract_sha256", "g0",
    "rounds", "totals", "final_global_state_sha256",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise RuntimeError(f"site audit has unexpected top-level fields: {path}")
    if value["format"] != FORMAT or value["version"] != VERSION or value["status"] != "PASS":
        raise RuntimeError(f"unsupported or failed site audit: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-a-audit", type=Path, required=True)
    parser.add_argument("--site-b-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = load(args.site_a_audit.resolve())
    b = load(args.site_b_audit.resolve())
    if a["site"] != "site-a" or b["site"] != "site-b":
        raise RuntimeError("audits were not supplied as site-a/site-b")
    if a["node_id"] != b["peer_node_id"] or b["node_id"] != a["peer_node_id"]:
        raise RuntimeError("site audits disagree on Slakshna endpoint identities")
    if a["dataset"] != b["dataset"] or a["partition_type"] != b["partition_type"]:
        raise RuntimeError("site audits disagree on the dataset contract")
    if a["training_contract_sha256"] != b["training_contract_sha256"]:
        raise RuntimeError("site training contracts differ")
    if a["training_contract"] != b["training_contract"]:
        raise RuntimeError("site training contract documents differ")
    if a["g0"] != b["g0"]:
        raise RuntimeError("sites did not start from an identical G0 bundle")
    indices_a = set(a["shard"]["train"]["source_indices"] + a["shard"]["validation"]["source_indices"])
    indices_b = set(b["shard"]["train"]["source_indices"] + b["shard"]["validation"]["source_indices"])
    if not indices_a.isdisjoint(indices_b):
        raise RuntimeError("site dataset shards overlap")
    if len(a["rounds"]) != 5 or len(b["rounds"]) != 5:
        raise RuntimeError("both sites must provide five rounds")

    rounds: list[dict[str, Any]] = []
    prior = a["g0"]["adapter_state_sha256"]
    for number, (round_a, round_b) in enumerate(zip(a["rounds"], b["rounds"]), 1):
        if round_a["round"] != number or round_b["round"] != number:
            raise RuntimeError(f"Round {number} numbering mismatch")
        if round_a["base_state_sha256"] != prior or round_b["base_state_sha256"] != prior:
            raise RuntimeError(f"Round {number} base chain mismatch")
        for sent, received, label in (
            (round_a["outbound"], round_b["received"], "site-a to site-b"),
            (round_b["outbound"], round_a["received"], "site-b to site-a"),
        ):
            for key in (
                "format", "version", "codec", "sender_site", "sender_node_id", "round",
                "base_state_sha256", "delta_file_sha256", "delta_state_sha256", "raw_bytes",
                "compressed_bytes", "tensor_count", "parameter_count",
            ):
                if sent[key] != received[key]:
                    raise RuntimeError(f"Round {number} {label} envelope mismatch: {key}")
        global_a = round_a["aggregation"]["global_state_sha256"]
        global_b = round_b["aggregation"]["global_state_sha256"]
        if global_a != global_b:
            raise RuntimeError(f"Round {number} global states diverged")
        prior = global_a
        rounds.append({
            "round": number,
            "base_state_sha256": round_a["base_state_sha256"],
            "site_a_delta_file_sha256": round_a["outbound"]["delta_file_sha256"],
            "site_b_delta_file_sha256": round_b["outbound"]["delta_file_sha256"],
            "global_state_sha256": global_a,
        })
    if a["final_global_state_sha256"] != prior or b["final_global_state_sha256"] != prior:
        raise RuntimeError("final global state does not match the round chain")
    expected_totals = {"training_rounds": 5, "local_epochs": 50, "optimizer_steps": 3600}
    if a["totals"] != expected_totals or b["totals"] != expected_totals:
        raise RuntimeError("Phase 8 training totals mismatch")
    result = {
        "format": "slakshna-phase8-paired-verification",
        "version": 1,
        "status": "PASS",
        "dataset": a["dataset"],
        "partition_type": a["partition_type"],
        "categories": {"site-a": a["shard"]["categories"], "site-b": b["shard"]["categories"]},
        "training_contract_sha256": a["training_contract_sha256"],
        "g0_state_sha256": a["g0"]["adapter_state_sha256"],
        "site_node_ids": {"site-a": a["node_id"], "site-b": b["node_id"]},
        "rounds": rounds,
        "totals_per_site": expected_totals,
        "final_global_state_sha256": prior,
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("PHASE8 PAIRED AUDIT PASSED")


if __name__ == "__main__":
    main()
