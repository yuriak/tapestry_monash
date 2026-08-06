#!/usr/bin/env python3
"""Dense FP32 LoRA FedAvg utilities for the minimal local Phase 4 simulation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.adapter_delta import (  # noqa: E402
    TOLERANCE,
    atomic_safetensors,
    max_abs_error,
    read_state,
    require_compatible,
    state_sha256,
    write_json,
)


def atomic_torch_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({name: value.cpu().contiguous() for name, value in state.items()}, temporary)
    temporary.replace(path)


def tensor_stats(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "nonzero_values": sum(int(torch.count_nonzero(value)) for value in state.values()),
        "l2_norm": math.sqrt(
            sum(float(torch.sum(value.double() ** 2)) for value in state.values())
        ),
        "state_sha256": state_sha256(state),
    }


def create_bridge(args: argparse.Namespace) -> None:
    state = read_state(args.adapter.resolve())
    atomic_torch_state(state, args.output.resolve())
    if args.canonical_output is not None:
        atomic_safetensors(state, args.canonical_output.resolve())
    print(json.dumps({"adapter": str(args.adapter.resolve()), **tensor_stats(state)}, indent=2))


def aggregate(args: argparse.Namespace) -> None:
    if len(args.client) != 2:
        raise RuntimeError(f"minimal Phase 4 requires exactly two clients, got {len(args.client)}")
    base_path = args.base.resolve()
    base = read_state(base_path)
    clients: list[dict[str, Any]] = []
    deltas: list[dict[str, torch.Tensor]] = []
    total_samples = 0

    for raw_name, raw_start, raw_trained, raw_samples in args.client:
        samples = int(raw_samples)
        if samples <= 0:
            raise RuntimeError(f"client {raw_name} has invalid sample count {samples}")
        start_path = Path(raw_start).resolve()
        trained_path = Path(raw_trained).resolve()
        start = read_state(start_path)
        trained = read_state(trained_path)
        require_compatible(base, start, f"base/{raw_name}-start")
        require_compatible(start, trained, f"{raw_name}-start/trained")
        start_error = max_abs_error(base, start)
        if start_error > TOLERANCE:
            raise RuntimeError(
                f"client {raw_name} did not start from the round base: {start_error}"
            )
        delta = {name: trained[name] - start[name] for name in base}
        stats = tensor_stats(delta)
        if stats["nonzero_values"] <= 0:
            raise RuntimeError(f"client {raw_name} produced an all-zero update")
        total_samples += samples
        deltas.append(delta)
        clients.append(
            {
                "name": raw_name,
                "samples": samples,
                "start_path": str(start_path),
                "trained_path": str(trained_path),
                "start_base_max_abs_error": start_error,
                "start_state_sha256": state_sha256(start),
                "trained_state_sha256": state_sha256(trained),
                "delta": stats,
            }
        )

    delta_difference = max_abs_error(deltas[0], deltas[1])
    if delta_difference == 0.0:
        raise RuntimeError("the two client updates are identical")

    weights = [client["samples"] / total_samples for client in clients]
    aggregate_delta = {
        name: sum(
            (delta[name] * weight for delta, weight in zip(deltas, weights)),
            torch.zeros_like(base[name]),
        ).float().contiguous()
        for name in base
    }
    global_state = {
        name: (base[name] + aggregate_delta[name]).float().contiguous() for name in base
    }
    aggregate_stats = tensor_stats(aggregate_delta)
    if aggregate_stats["nonzero_values"] <= 0:
        raise RuntimeError("FedAvg produced an all-zero aggregate update")
    global_change = max_abs_error(base, global_state)
    if global_change == 0.0:
        raise RuntimeError("FedAvg global adapter did not change")

    output_dir = args.output_dir.resolve()
    delta_path = output_dir / "aggregate_delta.safetensors"
    global_path = output_dir / "global_adapter.safetensors"
    resume_path = output_dir / "global_adapter.pth"
    atomic_safetensors(aggregate_delta, delta_path)
    atomic_safetensors(global_state, global_path)
    atomic_torch_state(global_state, resume_path)

    for client, weight in zip(clients, weights):
        client["weight"] = weight
    manifest = {
        "schema_version": 1,
        "round": args.round,
        "algorithm": "sample_weighted_fedavg",
        "dtype": "float32",
        "base_path": str(base_path),
        "base_state_sha256": state_sha256(base),
        "clients": clients,
        "total_samples": total_samples,
        "client_delta_max_abs_difference": delta_difference,
        "aggregate_delta": aggregate_stats,
        "global_state_sha256": state_sha256(global_state),
        "global_change_max_abs": global_change,
        "aggregate_delta_path": str(delta_path),
        "global_adapter_path": str(global_path),
        "resume_bridge_path": str(resume_path),
        "tolerance": TOLERANCE,
    }
    write_json(output_dir / "aggregation-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    bridge = subparsers.add_parser("bridge")
    bridge.add_argument("--adapter", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    bridge.add_argument("--canonical-output", type=Path)
    bridge.set_defaults(action=create_bridge)

    fedavg = subparsers.add_parser("aggregate")
    fedavg.add_argument("--round", type=int, required=True)
    fedavg.add_argument("--base", type=Path, required=True)
    fedavg.add_argument(
        "--client",
        nargs=4,
        action="append",
        metavar=("NAME", "START", "TRAINED", "SAMPLES"),
        required=True,
    )
    fedavg.add_argument("--output-dir", type=Path, required=True)
    fedavg.set_defaults(action=aggregate)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
