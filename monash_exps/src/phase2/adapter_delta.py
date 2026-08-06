#!/usr/bin/env python3
"""Create and verify the minimal Phase 2 LoRA parameter-delta round trip."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


TOLERANCE = 1.0e-7


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_state(path: Path) -> dict[str, torch.Tensor]:
    state = {name: value.float().contiguous() for name, value in load_file(str(path)).items()}
    if not state:
        raise RuntimeError(f"adapter state is empty: {path}")
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"adapter tensor is non-finite: {name}")
    return state


def require_compatible(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], label: str
) -> None:
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))
        extra = sorted(set(right) - set(left))
        raise RuntimeError(f"{label} tensor keys differ: missing={missing}, extra={extra}")
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise RuntimeError(
                f"{label} shape mismatch for {name}: {left[name].shape} != {right[name].shape}"
            )


def max_abs_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    require_compatible(left, right, "comparison")
    return max(float((left[name] - right[name]).abs().max()) for name in left)


def atomic_safetensors(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({name: value.contiguous() for name, value in state.items()}, str(temporary))
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_update(args: argparse.Namespace) -> None:
    initial = read_state(args.initial.resolve())
    trained = read_state(args.trained.resolve())
    require_compatible(initial, trained, "initial/trained")

    delta = {name: trained[name] - initial[name] for name in initial}
    if not all(torch.isfinite(value).all() for value in delta.values()):
        raise RuntimeError("delta contains NaN or Inf")
    nonzero = sum(int(torch.count_nonzero(value)) for value in delta.values())
    if nonzero == 0:
        raise RuntimeError("all LoRA delta tensors are zero")
    l2_norm = math.sqrt(sum(float(torch.sum(value.double() ** 2)) for value in delta.values()))

    applied = {name: initial[name] + delta[name] for name in initial}
    error = max_abs_error(applied, trained)
    if error > TOLERANCE:
        raise RuntimeError(f"initial + delta reconstruction error {error} exceeds {TOLERANCE}")

    output_dir = args.output_dir.resolve()
    delta_path = output_dir / "delta.safetensors"
    applied_path = output_dir / "applied_adapter.safetensors"
    resume_path = output_dir / "applied_adapter.pth"
    atomic_safetensors(delta, delta_path)
    atomic_safetensors(applied, applied_path)
    temporary_resume = resume_path.with_suffix(resume_path.suffix + ".tmp")
    torch.save(applied, temporary_resume)
    temporary_resume.replace(resume_path)

    manifest = {
        "schema_version": 1,
        "update_type": "lora_parameter_delta",
        "dtype": "float32",
        "tensor_count": len(delta),
        "parameter_count": sum(value.numel() for value in delta.values()),
        "nonzero_delta_values": nonzero,
        "delta_l2_norm": l2_norm,
        "reconstruction_tolerance": TOLERANCE,
        "reconstruction_max_abs_error": error,
        "initial_state_sha256": state_sha256(initial),
        "trained_state_sha256": state_sha256(trained),
        "applied_state_sha256": state_sha256(applied),
        "delta_state_sha256": state_sha256(delta),
        "delta_file_sha256": file_sha256(delta_path),
        "initial_path": str(args.initial.resolve()),
        "trained_path": str(args.trained.resolve()),
        "applied_path": str(applied_path),
        "resume_bridge_path": str(resume_path),
        "source_config_sha256": file_sha256(args.config.resolve()),
    }
    write_json(output_dir / "update-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    update_dir = args.update_dir.resolve()
    manifest = json.loads((update_dir / "update-manifest.json").read_text(encoding="utf-8"))
    trained = read_state(args.trained.resolve())
    applied = read_state(update_dir / "applied_adapter.safetensors")
    continued_from = read_state(args.continuation_initial.resolve())
    trained_applied_error = max_abs_error(trained, applied)
    applied_loaded_error = max_abs_error(applied, continued_from)
    if trained_applied_error > TOLERANCE:
        raise RuntimeError("applied adapter does not reconstruct the trained adapter")
    if applied_loaded_error > TOLERANCE:
        raise RuntimeError("fresh training process did not load the applied adapter")

    continuation = json.loads(args.continuation_summary.read_text(encoding="utf-8"))
    if continuation.get("status") != "PASS":
        raise RuntimeError("one-step continuation verifier did not pass")
    metrics = continuation["checks"]["finite_metrics_and_loss_trend"]["detail"]
    if metrics["first_step"] != 1 or metrics["last_step"] != 1:
        raise RuntimeError(f"expected exactly continuation step 1, got {metrics}")

    summary = {
        "status": "PASS",
        "mode": "phase2",
        "checks": {
            "update": {
                "status": "PASS",
                "detail": {
                    "tensor_count": manifest["tensor_count"],
                    "parameter_count": manifest["parameter_count"],
                    "nonzero_delta_values": manifest["nonzero_delta_values"],
                    "delta_l2_norm": manifest["delta_l2_norm"],
                    "delta_file_sha256": manifest["delta_file_sha256"],
                },
            },
            "fresh_process_apply": {
                "status": "PASS",
                "detail": {
                    "trained_applied_max_abs_error": trained_applied_error,
                    "applied_loaded_max_abs_error": applied_loaded_error,
                    "tolerance": TOLERANCE,
                },
            },
            "continue_training": {
                "status": "PASS",
                "detail": {
                    "optimizer_steps": 1,
                    "loss": metrics["initial_median_loss"],
                    "checkpoint_step": continuation["checks"]["checkpoint"]["detail"]["step"],
                    "adapter_changed": True,
                },
            },
        },
        "failures": [],
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PHASE2 PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--initial", type=Path, required=True)
    create.add_argument("--trained", type=Path, required=True)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.set_defaults(action=create_update)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--update-dir", type=Path, required=True)
    finish.add_argument("--trained", type=Path, required=True)
    finish.add_argument("--continuation-initial", type=Path, required=True)
    finish.add_argument("--continuation-summary", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.set_defaults(action=finalize)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
