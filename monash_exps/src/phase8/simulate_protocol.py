#!/usr/bin/env python3
"""Exercise five Phase 8 rounds using two isolated directories and wire text only."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import max_abs_error, state_sha256, write_json  # noqa: E402
from phase8.protocol import aggregate_equal_weight, decode_delta_payload, encode_delta_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(20260811)
    base = {
        "layer.0.lora_A.weight": torch.randn(8, 16, generator=generator),
        "layer.0.lora_B.weight": torch.randn(16, 8, generator=generator),
    }
    rounds = []
    with tempfile.TemporaryDirectory(prefix="phase8-protocol-") as temporary:
        root = Path(temporary)
        for number in range(1, 6):
            base_hash = state_sha256(base)
            delta_a = {name: torch.randn(value.shape, generator=generator) * 0.001 for name, value in base.items()}
            delta_b = {name: torch.randn(value.shape, generator=generator) * 0.001 for name, value in base.items()}
            path_a = root / "site-a" / f"round-{number}.safetensors"
            path_b = root / "site-b" / f"round-{number}.safetensors"
            path_a.parent.mkdir(parents=True, exist_ok=True)
            path_b.parent.mkdir(parents=True, exist_ok=True)
            save_file(delta_a, str(path_a))
            save_file(delta_b, str(path_b))
            payload_a, manifest_a = encode_delta_file(
                path_a, sender_site="site-a", sender_node_id="endpoint-a",
                round_number=number, base_state_sha256=base_hash,
            )
            payload_b, manifest_b = encode_delta_file(
                path_b, sender_site="site-b", sender_node_id="endpoint-b",
                round_number=number, base_state_sha256=base_hash,
            )
            # The only cross-site objects here are payload_a and payload_b strings.
            received_by_a = decode_delta_payload(
                payload_b, expected_sender_site="site-b", expected_sender_node_id="endpoint-b",
                expected_round=number, expected_base_state_sha256=base_hash,
            )
            received_by_b = decode_delta_payload(
                payload_a, expected_sender_site="site-a", expected_sender_node_id="endpoint-a",
                expected_round=number, expected_base_state_sha256=base_hash,
            )
            _, global_a = aggregate_equal_weight(base, delta_a, received_by_a.state)
            _, global_b = aggregate_equal_weight(base, delta_b, received_by_b.state)
            error = max_abs_error(global_a, global_b)
            if error != 0.0:
                raise RuntimeError(f"Round {number} isolated aggregation diverged: {error}")
            global_hash = state_sha256(global_a)
            rounds.append({
                "round": number,
                "base_state_sha256": base_hash,
                "site_a_delta_file_sha256": manifest_a["delta_file_sha256"],
                "site_b_delta_file_sha256": manifest_b["delta_file_sha256"],
                "global_state_sha256": global_hash,
                "site_global_max_abs_error": error,
            })
            base = global_a
    result = {
        "format": "slakshna-phase8-local-protocol-simulation",
        "version": 1,
        "status": "PASS",
        "transport": "wire-envelope-only; no cross-site filesystem reads",
        "rounds": rounds,
        "final_global_state_sha256": state_sha256(base),
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("PHASE8 LOCAL PROTOCOL SIMULATION PASSED")


if __name__ == "__main__":
    main()
