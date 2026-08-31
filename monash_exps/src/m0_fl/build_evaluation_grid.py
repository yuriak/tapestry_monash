#!/usr/bin/env python3
"""Materialize the normalized AU/India FL trajectory as vLLM LoRA weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SITES = ("au", "india")
ROUNDS = (2, 4, 6, 8, 10)
EXPECTED_TENSORS = 128
EXPECTED_PARAMETERS = 8_388_608


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "dummy" in state or len(state) != EXPECTED_TENSORS:
        raise RuntimeError(f"invalid LoRA state: {path}")
    if sum(value.numel() for value in state.values()) != EXPECTED_PARAMETERS:
        raise RuntimeError(f"LoRA parameter count mismatch: {path}")
    if any(
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or not torch.isfinite(value).all().item()
        for value in state.values()
    ):
        raise RuntimeError(f"invalid LoRA tensor: {path}")
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def materialize(source: Path, destination: Path) -> dict[str, Any]:
    state = load_state(source)
    temporary = destination.with_suffix(".safetensors.tmp")
    save_file(state, temporary)
    temporary.replace(destination)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "weights": str(destination.name),
        "weights_sha256": sha256(destination),
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "evaluation_grid").resolve()
    weights_dir = output_dir / "weights"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text(), end="")
        print("M0 FL EVALUATION GRID ALREADY COMPLETE")
        return 0
    if not (run_root / "finalized_round10" / "manifest.json").is_file():
        raise RuntimeError("round-10 finalized adapters are missing")
    weights_dir.mkdir(parents=True, exist_ok=False)

    adapters = []
    for site in SITES:
        sync_root = next((run_root / site / "ml_models").glob("sync_ckpt_*"))
        for round_index in ROUNDS:
            name = f"fl_{site}_p{round_index * 10:03d}"
            if round_index == 10:
                source = (
                    run_root
                    / "finalized_round10"
                    / site
                    / "adapter_model.pth"
                )
            else:
                source = sync_root / f"sync_round_{round_index}.pth"
            destination = weights_dir / f"{name}.safetensors"
            details = materialize(source, destination)
            details["weights"] = str(Path("weights") / destination.name)
            adapters.append(
                {
                    "name": name,
                    "training_run": f"fl_{site}",
                    "site": site,
                    "variant": "finalized" if round_index == 10 else "trajectory",
                    "step": round_index,
                    "final_step": 10,
                    "target_fraction": round_index / 10,
                    "actual_fraction": round_index / 10,
                    "epoch_equivalent": round_index / 5,
                    **details,
                }
            )

        raw_name = f"fl_{site}_p100_raw"
        raw_source = sync_root / "sync_round_10.pth"
        raw_destination = weights_dir / f"{raw_name}.safetensors"
        raw_details = materialize(raw_source, raw_destination)
        raw_details["weights"] = str(Path("weights") / raw_destination.name)
        adapters.append(
            {
                "name": raw_name,
                "training_run": f"fl_{site}",
                "site": site,
                "variant": "pre-finalization-diagnostic",
                "step": 10,
                "final_step": 10,
                "target_fraction": 1.0,
                "actual_fraction": 1.0,
                "epoch_equivalent": 2.0,
                **raw_details,
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "m0-local-fl-normalized-checkpoint-grid",
        "run_root": str(run_root),
        "base_model": "allenai/OLMo-2-1124-7B-Instruct",
        "fractions": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "adapters": adapters,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    print("M0 FL EVALUATION GRID PREPARATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
