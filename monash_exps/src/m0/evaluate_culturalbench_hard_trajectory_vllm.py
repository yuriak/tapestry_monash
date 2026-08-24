#!/usr/bin/env python3
"""Evaluate representative intermediate M0 adapters on CulturalBench-Hard."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from evaluate_culturalbench_hard_vllm import (
    SYSTEM_PROMPT,
    completed_ids,
    load_examples,
    parse_bool,
    summarize,
)
from evaluate_culturalbench_vllm import RUN_SPECS, create_adapter_view, sha256

CHECKPOINT_STEPS = {
    "local_south_asia": [480, 960, 1440, 1916],
    "local_variant_2": [1350, 2700, 4275, 5630],
    "central_variant_1": [750, 1500, 2250, 3082],
}


def run_dir(runtime: Path, import_root: Path, run: str) -> Path:
    return runtime / run if RUN_SPECS[run]["source"] == "m3" else import_root / run


def resolve_checkpoints(
    runtime: Path, import_root: Path, model: Path, view_root: Path
) -> dict[str, dict[str, Any]]:
    adapters: dict[str, dict[str, Any]] = {
        "base": {
            "view": None,
            "weights": None,
            "adapter_sha256": None,
            "training_run": "base",
            "step": 0,
            "final_step": 0,
            "training_fraction": 0.0,
            "epoch_equivalent": 0.0,
        }
    }
    for training_run, steps in CHECKPOINT_STEPS.items():
        root = run_dir(runtime, import_root, training_run)
        final_step = RUN_SPECS[training_run]["step"]
        if steps[-1] != final_step:
            raise ValueError(
                f"{training_run}: trajectory does not include the final step"
            )
        for step in steps:
            checkpoint = root / "adapter_history" / f"step_{step:07d}"
            weights = checkpoint / "adapter_model.safetensors"
            if not (checkpoint / ".complete").is_file() or not weights.is_file():
                raise ValueError(f"Incomplete adapter checkpoint: {checkpoint}")
            name = f"{training_run}_step_{step:07d}"
            digest = sha256(weights)
            view = create_adapter_view(view_root, name, weights, model)
            adapters[name] = {
                "view": str(view),
                "weights": str(weights.resolve()),
                "adapter_sha256": digest,
                "training_run": training_run,
                "step": step,
                "final_step": final_step,
                "training_fraction": step / final_step,
                "epoch_equivalent": 2.0 * step / final_step,
            }
    return adapters


def true_rate(path: Path) -> float:
    total = 0
    predicted_true = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            predicted_true += row["parsed_answer"] == "TRUE"
    if not total:
        raise ValueError(f"Empty result file: {path}")
    return predicted_true / total


def write_summary(
    output_dir: Path,
    adapters: dict[str, dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]],
) -> None:
    rows = []
    for name, values in summaries.items():
        overall = next(row for row in values if row["region_group"] == "overall")
        result_path = output_dir / "results" / f"{name}.jsonl"
        rows.append(
            {
                "checkpoint": name,
                "training_run": adapters[name]["training_run"],
                "step": adapters[name]["step"],
                "final_step": adapters[name]["final_step"],
                "training_fraction": adapters[name]["training_fraction"],
                "epoch_equivalent": adapters[name]["epoch_equivalent"],
                "binary_accuracy": overall["binary_accuracy"],
                "question_exact_match": overall["question_exact_match"],
                "reconstructed_mc_accuracy": overall["reconstructed_mc_accuracy"],
                "true_prediction_rate": true_rate(result_path),
                "invalid_judgments": overall["invalid_judgments"],
            }
        )
    fields = list(rows[0]) if rows else []
    with (output_dir / "trajectory_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# M0 CulturalBench-Hard Checkpoint Trajectory",
        "",
        (
            "The three selected runs are evaluated at approximately 0.5, 1.0, 1.5, "
            "and 2.0 data epochs. The base row is included as an inference reference."
        ),
        "",
        "| Run | Step | Approx. epoch | Binary accuracy | Question exact match | Reconstructed MC | TRUE rate | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = "Base" if row["checkpoint"] == "base" else row["training_run"]
        lines.append(
            f"| {label} | {row['step']:,} | {row['epoch_equivalent']:.2f} | "
            f"{100 * row['binary_accuracy']:.2f}% | "
            f"{100 * row['question_exact_match']:.2f}% | "
            f"{100 * row['reconstructed_mc_accuracy']:.2f}% | "
            f"{100 * row['true_prediction_rate']:.2f}% | "
            f"{row['invalid_judgments']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    dataset = args.dataset.resolve()
    runtime = args.runtime_root.resolve()
    import_root = args.import_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (model / "config.json").is_file() or not dataset.is_file():
        raise ValueError("Model or CulturalBench-Hard input is missing")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"
    view_root.mkdir(exist_ok=True)
    examples = load_examples(dataset)
    adapters = resolve_checkpoints(runtime, import_root, model, view_root)
    names = list(adapters)
    manifest = {
        "schema_version": 1,
        "benchmark": "CulturalBench-Hard-checkpoint-trajectory",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "example_count": len(examples),
        "system_prompt": SYSTEM_PROMPT,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "benchmark",
            "model",
            "dataset_sha256",
            "example_count",
            "system_prompt",
            "checkpoint_steps",
            "adapters",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest changed: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Prepared {len(examples)} judgments for {len(adapters) - 1} checkpoints plus base",
        flush=True,
    )
    print(f"Output directory: {output_dir}", flush=True)
    if args.prepare_only:
        print("M0 CULTURALBENCH-HARD TRAJECTORY PREPARATION PASSED")
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_cpu_loras=max(16, len(adapters) - 1),
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=4, stop=["\n"])
    expected = {row["example_id"] for row in examples}
    summaries: dict[str, list[dict[str, Any]]] = {}
    for adapter_id, name in enumerate(names, 1):
        path = results_dir / f"{name}.jsonl"
        done = completed_ids(path)
        if not done.issubset(expected):
            raise ValueError(f"{name}: unexpected IDs in result")
        pending = [row for row in examples if row["example_id"] not in done]
        print(f"[{name}] complete={len(done)} pending={len(pending)}", flush=True)
        request = (
            None
            if name == "base"
            else LoRARequest(name, adapter_id, adapters[name]["view"])
        )
        with path.open("a") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                generated = llm.chat(
                    [row["messages"] for row in batch],
                    sampling_params=sampling,
                    lora_request=request,
                    use_tqdm=True,
                )
                if len(generated) != len(batch):
                    raise RuntimeError("vLLM output count mismatch")
                for row, output in zip(batch, generated):
                    raw = output.outputs[0].text
                    parsed = parse_bool(raw)
                    record = {
                        key: row[key]
                        for key in (
                            "example_id",
                            "data_idx",
                            "question_idx",
                            "country",
                            "region_group",
                            "question",
                            "proposed_answer",
                            "gold_answer",
                        )
                    }
                    record.update(
                        checkpoint=name,
                        training_run=adapters[name]["training_run"],
                        step=adapters[name]["step"],
                        adapter_sha256=adapters[name]["adapter_sha256"],
                        raw_output=raw,
                        parsed_answer=parsed,
                        correct=parsed == row["gold_answer"],
                        finish_reason=output.outputs[0].finish_reason,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"[{name}] wrote {min(start + len(batch), len(pending))}/{len(pending)} pending",
                    flush=True,
                )
        if completed_ids(path) != expected:
            raise RuntimeError(f"{name}: incomplete result coverage")
        summaries[name] = summarize(path)
        write_summary(output_dir, adapters, summaries)

    manifest.update(
        status="completed",
        completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 CULTURALBENCH-HARD TRAJECTORY EVALUATION PASSED")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
