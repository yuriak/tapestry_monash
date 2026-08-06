#!/usr/bin/env python3
"""Fail-closed verifier for two-client, two-round local FedAvg."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.adapter_delta import (  # noqa: E402
    TOLERANCE,
    max_abs_error,
    read_state,
    state_sha256,
    write_json,
)


CLIENTS = {
    "client-a": [0, 2, 4, 6],
    "client-b": [1, 3, 5, 7],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def trained_path(client_dir: Path) -> Path:
    return client_dir / "checkpoints/step_0000002/adapter_model.safetensors"


def verify_client(
    client_dir: Path, client_name: str, expected_base: dict[str, torch.Tensor]
) -> dict[str, Any]:
    summary = json.loads((client_dir / "verification-summary.json").read_text(encoding="utf-8"))
    require(summary.get("status") == "PASS", f"{client_dir} training did not pass")
    require(
        summary["checks"]["checkpoint"]["detail"]["step"] == 2,
        f"{client_dir} did not stop at checkpoint step 2",
    )
    evidence = json.loads((client_dir / "worker-evidence/rank-0-start.json").read_text())
    require(evidence.get("resume_step") == 0, f"{client_dir} reused DCP state")
    require(evidence.get("resume_checkpoint") is None, f"{client_dir} discovered an old checkpoint")

    inputs = json.loads((client_dir / "input-manifest.json").read_text(encoding="utf-8"))
    indices = inputs["dataset"].get("selected_indices")
    require(indices == CLIENTS[client_name], f"{client_name} has unexpected data rows {indices}")
    require(inputs["dataset"].get("selected_rows") == 4, f"{client_name} sample count is not four")

    start = read_state(client_dir / "initial_adapter.safetensors")
    trained = read_state(trained_path(client_dir))
    start_error = max_abs_error(start, expected_base)
    require(start_error <= TOLERANCE, f"{client_dir} start differs from round base: {start_error}")
    delta = {name: trained[name] - start[name] for name in start}
    nonzero = sum(int(torch.count_nonzero(value)) for value in delta.values())
    require(nonzero > 0, f"{client_dir} produced an all-zero delta")
    return {
        "start": start,
        "trained": trained,
        "delta": delta,
        "start_error": start_error,
        "nonzero_delta_values": nonzero,
        "final_median_loss": summary["checks"]["finite_metrics_and_loss_trend"]["detail"][
            "final_median_loss"
        ],
        "selected_indices": indices,
    }


def verify_round(root: Path, round_number: int, base_path: Path) -> dict[str, Any]:
    round_dir = root / f"round-{round_number}"
    base = read_state(base_path)
    results = {
        name: verify_client(round_dir / name, name, base) for name in CLIENTS
    }
    require(
        set(results["client-a"]["selected_indices"]).isdisjoint(
            results["client-b"]["selected_indices"]
        ),
        f"round {round_number} client shards overlap",
    )
    require(
        sorted(results["client-a"]["selected_indices"] + results["client-b"]["selected_indices"])
        == list(range(8)),
        f"round {round_number} client shards do not cover the source set",
    )
    delta_difference = max_abs_error(results["client-a"]["delta"], results["client-b"]["delta"])
    require(delta_difference > 0.0, f"round {round_number} client deltas are identical")

    expected_delta = {
        name: (
            results["client-a"]["delta"][name] * 0.5
            + results["client-b"]["delta"][name] * 0.5
        ).float().contiguous()
        for name in base
    }
    expected_global = {
        name: (base[name] + expected_delta[name]).float().contiguous() for name in base
    }
    aggregation_dir = round_dir / "aggregation"
    actual_delta = read_state(aggregation_dir / "aggregate_delta.safetensors")
    actual_global = read_state(aggregation_dir / "global_adapter.safetensors")
    delta_error = max_abs_error(expected_delta, actual_delta)
    global_error = max_abs_error(expected_global, actual_global)
    require(delta_error <= TOLERANCE, f"round {round_number} FedAvg delta error {delta_error}")
    require(global_error <= TOLERANCE, f"round {round_number} global adapter error {global_error}")
    global_change = max_abs_error(base, actual_global)
    require(global_change > 0.0, f"round {round_number} global adapter did not change")

    manifest = json.loads((aggregation_dir / "aggregation-manifest.json").read_text())
    require(manifest.get("algorithm") == "sample_weighted_fedavg", "unexpected algorithm")
    require(manifest.get("total_samples") == 8, "FedAvg total sample count is not eight")
    weights = {client["name"]: client["weight"] for client in manifest["clients"]}
    require(weights == {"client-a": 0.5, "client-b": 0.5}, f"wrong weights: {weights}")
    require(
        manifest.get("global_state_sha256") == state_sha256(actual_global),
        f"round {round_number} manifest global hash mismatch",
    )

    return {
        "base": base,
        "global": actual_global,
        "global_path": aggregation_dir / "global_adapter.safetensors",
        "client_delta_max_abs_difference": delta_difference,
        "aggregate_delta_max_abs_error": delta_error,
        "global_adapter_max_abs_error": global_error,
        "global_change_max_abs": global_change,
        "clients": {
            name: {
                "start_base_max_abs_error": result["start_error"],
                "nonzero_delta_values": result["nonzero_delta_values"],
                "final_median_loss": result["final_median_loss"],
                "selected_indices": result["selected_indices"],
            }
            for name, result in results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()

    global0 = root / "global-0/global_adapter.safetensors"
    round1 = verify_round(root, 1, global0)
    round2 = verify_round(root, 2, round1["global_path"])
    transition_error = max_abs_error(round1["global"], round2["base"])
    require(transition_error <= TOLERANCE, f"round 2 did not consume round 1 global: {transition_error}")
    global0_global2_change = max_abs_error(round1["base"], round2["global"])
    require(global0_global2_change > 0.0, "two rounds left the global adapter unchanged")

    def public_round(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key not in {"base", "global", "global_path"}}

    summary = {
        "status": "PASS",
        "mode": "phase4",
        "checks": {
            "data_partition": {
                "status": "PASS",
                "detail": {"client-a": CLIENTS["client-a"], "client-b": CLIENTS["client-b"]},
            },
            "round_1": {"status": "PASS", "detail": public_round(round1)},
            "round_2": {"status": "PASS", "detail": public_round(round2)},
            "multi_round_transition": {
                "status": "PASS",
                "detail": {
                    "round1_global_to_round2_base_max_abs_error": transition_error,
                    "global0_to_global2_max_abs_change": global0_global2_change,
                    "fresh_optimizer_runs": 4,
                },
            },
        },
        "failures": [],
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PHASE4 PASSED")


if __name__ == "__main__":
    main()
