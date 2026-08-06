#!/usr/bin/env python3
"""Fail-closed verifier for Phase 1A and combined Phase 1B/1C runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Callable

import yaml
from safetensors.torch import load_file


LOSS_RE = re.compile(r"\[step\s+(\d+)\]\s+loss=([0-9.eE+-]+).*grad_norm=([0-9.eE+-]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("phase1a", "phase1b", "phase1c"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--minimum-loss-drop", type=float, default=0.0)
    parser.add_argument("--expected-resume-step", type=int, default=0)
    parser.add_argument("--expected-final-step", type=int)
    parser.add_argument("--prior-run-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

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

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def verify_config() -> dict[str, Any]:
        config = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
        require(config["model"]["dtype"] == "bfloat16", "dtype is not BF16")
        require(config["training"]["distributed"]["strategy"] == "ddp", "not DDP")
        require(config["lora"]["enabled"] is True, "LoRA disabled")
        return {
            "model": config["model"]["name"],
            "max_steps": config["training"]["max_steps"],
            "num_epochs": config["training"]["num_epochs"],
            "checkpoint_dir": config["checkpoint"]["save_dir"],
        }

    config_detail = check("resolved_config", verify_config)

    def expected_final_step() -> int:
        require(config_detail is not None, "config unavailable")
        return (
            args.expected_final_step
            if args.expected_final_step is not None
            else int(config_detail["max_steps"])
        )

    def verify_launcher() -> dict[str, Any]:
        data = json.loads((run_dir / "launcher-environment.json").read_text())
        require(data["visible_gpus"] == args.expected_workers, "visible GPU count mismatch")
        require(data["ray_resources"].get("GPU") == args.expected_workers, "Ray GPU mismatch")
        names = data["gpu_names"]
        require(len(names) == args.expected_workers and all(str(name).strip() for name in names),
                f"missing GPU names: {names}")
        if args.mode != "phase1a":
            require(all("A100" in name.upper() for name in names), f"expected A100s, got {names}")
        return data

    check("launcher_resources", verify_launcher)

    final_evidence: list[dict[str, Any]] = []

    def verify_workers() -> dict[str, Any]:
        starts = []
        finals = []
        for rank in range(args.expected_workers):
            starts.append(json.loads((run_dir / "worker-evidence" / f"rank-{rank}-start.json").read_text()))
            finals.append(json.loads((run_dir / "worker-evidence" / f"rank-{rank}-final.json").read_text()))
        require({item["rank"] for item in starts} == set(range(args.expected_workers)), "rank set mismatch")
        require({item["local_rank"] for item in starts} == set(range(args.expected_workers)),
                "local rank set mismatch")
        require(all(item["world_size"] == args.expected_workers for item in starts),
                "world size mismatch")
        require(len({item["cuda_device"] for item in starts}) == args.expected_workers,
                "workers did not use distinct logical GPUs")
        require(all(item["resume_step"] == args.expected_resume_step for item in starts),
                f"expected resume step {args.expected_resume_step}")
        require(all(item["initial_lora"]["sha256"] != item["final_lora"]["sha256"] for item in finals),
                "LoRA parameters did not change")
        hashes = {item["final_lora"]["sha256"] for item in finals}
        require(len(hashes) == 1, f"rank LoRA hashes differ: {hashes}")
        if args.expected_resume_step > 0:
            require(all(item.get("resume_loaded_lora") is not None for item in finals),
                    "checkpoint was discovered but no in-memory LoRA state was restored")
            loaded = {item["resume_loaded_lora"]["sha256"] for item in finals}
            require(len(loaded) == 1, f"rank resume hashes differ: {loaded}")
            adapter_hashes = {item.get("resume_adapter_sha256") for item in finals}
            require(None not in adapter_hashes and len(adapter_hashes) == 1,
                    f"rank checkpoint adapter hashes differ or are missing: {adapter_hashes}")
            prior_run_dir = (
                args.prior_run_dir.resolve()
                if args.prior_run_dir is not None
                else run_dir.parent / "phase1b-ddp-overfit"
            )
            prior_path = prior_run_dir / "worker-evidence" / "rank-0-final.json"
            prior = json.loads(prior_path.read_text())
            expected_loaded = prior["final_lora"]["sha256"]
            require(next(iter(loaded)) == expected_loaded,
                    "in-memory resumed LoRA state does not equal Run 1 final state")
        final_evidence.extend(finals)
        return {
            "ranks": [item["rank"] for item in starts],
            "local_ranks": [item["local_rank"] for item in starts],
            "world_size": args.expected_workers,
            "final_lora_sha256": next(iter(hashes)),
            "resume_loaded_lora_sha256": (
                finals[0]["resume_loaded_lora"]["sha256"]
                if args.expected_resume_step > 0 else None
            ),
            "resume_step": args.expected_resume_step,
        }

    check("worker_world_and_lora", verify_workers)

    def verify_metrics() -> dict[str, Any]:
        text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        points = [(int(step), float(loss), float(grad)) for step, loss, grad in LOSS_RE.findall(text)]
        require(points, "no structured step metrics found")
        require(all(math.isfinite(loss) and math.isfinite(grad) for _, loss, grad in points),
                "non-finite loss or gradient norm")
        require(config_detail is not None, "config unavailable")
        expected_steps = list(range(
            args.expected_resume_step + 1,
            expected_final_step() + 1,
        ))
        observed_steps = [point[0] for point in points]
        require(observed_steps == expected_steps,
                f"expected metric steps {expected_steps}, got {observed_steps}")
        losses = [point[1] for point in points]
        window = min(3, len(losses))
        initial = statistics.median(losses[:window])
        final = statistics.median(losses[-window:])
        drop = (initial - final) / initial if initial else 0.0
        require(drop >= args.minimum_loss_drop,
                f"loss drop {drop:.3%} is below {args.minimum_loss_drop:.3%}")
        return {
            "metric_points": len(points),
            "first_step": points[0][0],
            "last_step": points[-1][0],
            "initial_median_loss": initial,
            "final_median_loss": final,
            "relative_loss_drop": drop,
        }

    check("finite_metrics_and_loss_trend", verify_metrics)

    def verify_checkpoint() -> dict[str, Any]:
        require(config_detail is not None, "config unavailable")
        root = Path(config_detail["checkpoint_dir"])
        checkpoints = sorted(
            path for path in root.glob("step_*") if path.is_dir() and (path / ".complete").is_file()
        )
        require(checkpoints, "no completed checkpoint")
        latest = checkpoints[-1]
        meta = json.loads((latest / "meta.json").read_text())
        require(int(meta["step"]) == expected_final_step(),
                f"latest checkpoint step {meta['step']} does not equal expected final step "
                f"{expected_final_step()}")
        adapter_path = latest / "adapter_model.safetensors"
        require((latest / "model" / ".metadata").is_file(), "model DCP metadata missing")
        require((latest / "optim" / ".metadata").is_file(), "optimizer DCP metadata missing")
        require(adapter_path.is_file(), "adapter safetensors missing")
        tensors = load_file(str(adapter_path))
        require(tensors, "adapter is empty")
        require(any(int(tensor.count_nonzero()) > 0 for tensor in tensors.values()),
                "all adapter tensors are zero")
        return {
            "path": str(latest),
            "step": int(meta["step"]),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_tensor_count": len(tensors),
            "cursor_samples": meta.get("eval_lifecycle/samples_consumed"),
            "cursor_tokens": meta.get("eval_lifecycle/tokens_consumed"),
        }

    check("checkpoint", verify_checkpoint)
    check("ray_result", lambda: json.loads((run_dir / "ray-result.json").read_text()))

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "checks": checks,
        "failures": failures,
    }
    (run_dir / "verification-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print(f"{args.mode.upper()} PASSED")


if __name__ == "__main__":
    main()
