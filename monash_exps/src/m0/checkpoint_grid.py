#!/usr/bin/env python3
"""Resolve normalized M0 training-progress points to retained LoRA adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluate_culturalbench_vllm import RUN_SPECS, create_adapter_view, sha256

DEFAULT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
STEP_DIR_RE = re.compile(r"step_(\d{7})$")


def run_directory(runtime: Path, import_root: Path, run: str) -> Path:
    spec = RUN_SPECS[run]
    return runtime / run if spec["source"] == "m3" else import_root / run


def retained_steps(run_root: Path) -> list[int]:
    history = run_root / "adapter_history"
    steps = []
    if not history.is_dir():
        raise ValueError(f"Missing adapter history: {history}")
    for path in history.iterdir():
        match = STEP_DIR_RE.fullmatch(path.name)
        if (
            match
            and (path / ".complete").is_file()
            and (path / "adapter_model.safetensors").is_file()
        ):
            steps.append(int(match.group(1)))
    if not steps:
        raise ValueError(f"No complete adapters found in {history}")
    return sorted(steps)


def nearest_steps(
    available: list[int], final_step: int, fractions: Iterable[float]
) -> list[tuple[float, int]]:
    selected = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Checkpoint fractions must be in (0, 1], got {fraction}")
        target = final_step * fraction
        step = min(available, key=lambda value: (abs(value - target), value))
        selected.append((fraction, step))
    if len({step for _, step in selected}) != len(selected):
        raise ValueError(
            f"Checkpoint fractions map to duplicate retained steps: {selected}"
        )
    if selected[-1][0] == 1.0 and selected[-1][1] != final_step:
        raise ValueError(f"Final adapter step {final_step} is not retained")
    return selected


def resolve_checkpoint_grid(
    runtime: Path,
    import_root: Path,
    model: Path,
    view_root: Path,
    fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> dict[str, dict[str, Any]]:
    fractions = tuple(fractions)
    adapters: dict[str, dict[str, Any]] = {
        "base": {
            "view": None,
            "weights": None,
            "adapter_sha256": None,
            "training_run": "base",
            "step": 0,
            "final_step": 0,
            "target_fraction": 0.0,
            "actual_fraction": 0.0,
            "epoch_equivalent": 0.0,
        }
    }
    for training_run, spec in RUN_SPECS.items():
        root = run_directory(runtime, import_root, training_run)
        final_step = int(spec["step"])
        available = retained_steps(root)
        for target_fraction, step in nearest_steps(available, final_step, fractions):
            checkpoint = root / "adapter_history" / f"step_{step:07d}"
            weights = checkpoint / "adapter_model.safetensors"
            name = f"{training_run}_p{round(100 * target_fraction):03d}"
            digest = sha256(weights)
            view = create_adapter_view(view_root, name, weights, model)
            adapters[name] = {
                "view": str(view),
                "weights": str(weights.resolve()),
                "adapter_sha256": digest,
                "training_run": training_run,
                "step": step,
                "final_step": final_step,
                "target_fraction": target_fraction,
                "actual_fraction": step / final_step,
                "epoch_equivalent": 2.0 * step / final_step,
            }
    return adapters
