#!/usr/bin/env python3
"""Evaluate CulturalBench Easy and Hard across the normalized M0 checkpoint grid."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from checkpoint_grid import (
    DEFAULT_FRACTIONS,
    resolve_adapter_manifest,
    resolve_checkpoint_grid,
)
from evaluate_culturalbench_hard_vllm import load_examples as load_hard_examples
from evaluate_culturalbench_hard_vllm import parse_bool
from evaluate_culturalbench_vllm import load_examples as load_easy_examples
from evaluate_culturalbench_vllm import parse_answer, sha256

REGIONS = ("overall", "australia_nz", "india", "rest_of_world")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                result.add(json.loads(line)["evaluation_id"])
            except Exception as error:
                raise ValueError(f"Invalid result line {path}:{number}") from error
    return result


def summarize(path: Path) -> list[dict[str, Any]]:
    easy = []
    hard = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            (easy if row["benchmark_split"] == "easy" else hard).append(row)

    hard_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard:
        hard_by_question[row["question_idx"]].append(row)
    if any(len(group) != 4 for group in hard_by_question.values()):
        raise ValueError(
            f"Hard results do not have four judgments per question: {path}"
        )

    output = []
    for region in REGIONS:
        easy_rows = (
            easy
            if region == "overall"
            else [row for row in easy if row["region_group"] == region]
        )
        hard_rows = (
            hard
            if region == "overall"
            else [row for row in hard if row["region_group"] == region]
        )
        hard_groups = [
            group
            for group in hard_by_question.values()
            if region == "overall" or group[0]["region_group"] == region
        ]
        easy_correct = sum(row["correct"] for row in easy_rows)
        hard_binary_correct = sum(row["correct"] for row in hard_rows)
        hard_exact_correct = sum(
            all(row["correct"] for row in group) for group in hard_groups
        )
        reconstructed_correct = 0
        reconstructed_valid = 0
        for group in hard_groups:
            predicted_true = [row for row in group if row["parsed_answer"] == "TRUE"]
            gold_true = [row for row in group if row["gold_answer"] == "TRUE"]
            if len(predicted_true) == 1:
                reconstructed_valid += 1
                reconstructed_correct += (
                    len(gold_true) == 1
                    and predicted_true[0]["evaluation_id"]
                    == gold_true[0]["evaluation_id"]
                )
        decision_total = len(easy_rows) + len(hard_rows)
        question_total = len(easy_rows) + len(hard_groups)
        output.append(
            {
                "region_group": region,
                "easy_questions": len(easy_rows),
                "easy_correct": easy_correct,
                "easy_accuracy": easy_correct / len(easy_rows) if easy_rows else None,
                "hard_judgments": len(hard_rows),
                "hard_binary_correct": hard_binary_correct,
                "hard_binary_accuracy": (
                    hard_binary_correct / len(hard_rows) if hard_rows else None
                ),
                "hard_questions": len(hard_groups),
                "hard_question_exact_correct": hard_exact_correct,
                "hard_question_exact_match": (
                    hard_exact_correct / len(hard_groups) if hard_groups else None
                ),
                "hard_single_true_questions": reconstructed_valid,
                "hard_reconstructed_mc_accuracy": (
                    reconstructed_correct / len(hard_groups) if hard_groups else None
                ),
                "hard_true_prediction_rate": (
                    sum(row["parsed_answer"] == "TRUE" for row in hard_rows)
                    / len(hard_rows)
                    if hard_rows
                    else None
                ),
                "combined_decision_units": decision_total,
                "combined_decision_accuracy": (
                    (easy_correct + hard_binary_correct) / decision_total
                    if decision_total
                    else None
                ),
                "combined_questions": question_total,
                "combined_question_accuracy": (
                    (easy_correct + hard_exact_correct) / question_total
                    if question_total
                    else None
                ),
                "invalid_outputs": sum(
                    row["parsed_answer"] is None for row in easy_rows
                )
                + sum(row["parsed_answer"] is None for row in hard_rows),
            }
        )
    return output


def write_summary(
    output_dir: Path,
    adapters: dict[str, dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]],
) -> None:
    metric_fields = list(next(iter(summaries.values()))[0]) if summaries else []
    metadata_fields = [
        "checkpoint",
        "training_run",
        "step",
        "final_step",
        "target_fraction",
        "actual_fraction",
        "epoch_equivalent",
    ]
    rows = []
    for checkpoint, values in summaries.items():
        metadata = adapters[checkpoint]
        rows.extend(
            {
                "checkpoint": checkpoint,
                **{field: metadata[field] for field in metadata_fields[1:]},
                **value,
            }
            for value in values
        )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields + metric_fields)
        writer.writeheader()
        writer.writerows(rows)

    overall = {
        row["checkpoint"]: row for row in rows if row["region_group"] == "overall"
    }
    lines = [
        "# M0 CulturalBench Training Trajectories",
        "",
        (
            "Easy and Hard are reported separately. Combined decision accuracy treats one Easy "
            "question and one Hard TRUE/FALSE judgment as one unit. Combined question accuracy "
            "treats one Easy question and one four-judgment Hard exact match as one unit."
        ),
        "",
        "| Checkpoint | Progress | Easy | Hard binary | Hard exact | Hard reconstructed | Combined decisions | Combined questions | TRUE rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in summaries:
        row = overall[checkpoint]
        lines.append(
            f"| {checkpoint} | {100 * row['actual_fraction']:.1f}% | "
            f"{100 * row['easy_accuracy']:.2f}% | "
            f"{100 * row['hard_binary_accuracy']:.2f}% | "
            f"{100 * row['hard_question_exact_match']:.2f}% | "
            f"{100 * row['hard_reconstructed_mc_accuracy']:.2f}% | "
            f"{100 * row['combined_decision_accuracy']:.2f}% | "
            f"{100 * row['combined_question_accuracy']:.2f}% | "
            f"{100 * row['hard_true_prediction_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                "Regional Easy, Hard, and combined results are retained in `summary.csv` for "
                "Australia/NZ, India, and the rest of the world."
            ),
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--easy-dataset", type=Path, required=True)
    parser.add_argument("--hard-dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-manifest",
        type=Path,
        help="Explicit adapter grid; defaults to the seven baseline trajectories.",
    )
    parser.add_argument("--request-batch-size", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    easy_dataset = args.easy_dataset.resolve()
    hard_dataset = args.hard_dataset.resolve()
    runtime = args.runtime_root.resolve()
    import_root = args.import_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"Invalid model directory: {model}")
    if not easy_dataset.is_file() or not hard_dataset.is_file():
        raise ValueError("CulturalBench Easy or Hard input is missing")
    if args.request_batch_size < 4096:
        raise ValueError("--request-batch-size must be at least 4096")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"
    view_root.mkdir(exist_ok=True)
    easy_examples = load_easy_examples(easy_dataset, None)
    hard_examples = load_hard_examples(hard_dataset)
    examples = [
        {**row, "benchmark_split": "easy", "evaluation_id": row["example_id"]}
        for row in easy_examples
    ] + [
        {**row, "benchmark_split": "hard", "evaluation_id": row["example_id"]}
        for row in hard_examples
    ]
    adapters = (
        resolve_adapter_manifest(args.adapter_manifest, model, view_root)
        if args.adapter_manifest
        else resolve_checkpoint_grid(runtime, import_root, model, view_root)
    )
    checkpoints = list(adapters)
    manifest = {
        "schema_version": 1,
        "benchmark": "CulturalBench-Easy-Hard-training-grid",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model),
        "easy_dataset": str(easy_dataset),
        "easy_dataset_sha256": sha256(easy_dataset),
        "hard_dataset": str(hard_dataset),
        "hard_dataset_sha256": sha256(hard_dataset),
        "easy_examples": len(easy_examples),
        "hard_judgments": len(hard_examples),
        "checkpoint_fractions": (
            sorted({item["target_fraction"] for item in adapters.values()})
            if args.adapter_manifest
            else list(DEFAULT_FRACTIONS)
        ),
        "request_batch_size": args.request_batch_size,
        "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "benchmark",
            "model",
            "easy_dataset_sha256",
            "hard_dataset_sha256",
            "easy_examples",
            "hard_judgments",
            "checkpoint_fractions",
            "request_batch_size",
            "adapters",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest changed: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Prepared {len(easy_examples)} Easy questions + {len(hard_examples)} Hard "
        f"judgments for {len(checkpoints)} model states",
        flush=True,
    )
    print(f"Request pool per model: {len(examples)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    if args.prepare_only:
        print("M0 CULTURALBENCH GRID PREPARATION PASSED")
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
        max_cpu_loras=max(40, len(checkpoints) - 1),
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    expected = {row["evaluation_id"] for row in examples}
    summaries: dict[str, list[dict[str, Any]]] = {}
    for adapter_id, checkpoint in enumerate(checkpoints, 1):
        result_path = results_dir / f"{checkpoint}.jsonl"
        done = completed_ids(result_path)
        if not done.issubset(expected):
            raise ValueError(f"{checkpoint}: result contains unexpected IDs")
        pending = [row for row in examples if row["evaluation_id"] not in done]
        print(f"[{checkpoint}] complete={len(done)} pending={len(pending)}", flush=True)
        request = (
            None
            if checkpoint == "base"
            else LoRARequest(checkpoint, adapter_id, adapters[checkpoint]["view"])
        )
        with result_path.open("a") as handle:
            for start in range(0, len(pending), args.request_batch_size):
                batch = pending[start : start + args.request_batch_size]
                sampling = [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=8 if row["benchmark_split"] == "easy" else 4,
                        stop=["\n"],
                    )
                    for row in batch
                ]
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
                    parsed = (
                        parse_answer(raw)
                        if row["benchmark_split"] == "easy"
                        else parse_bool(raw)
                    )
                    record = {
                        "checkpoint": checkpoint,
                        "training_run": adapters[checkpoint]["training_run"],
                        "step": adapters[checkpoint]["step"],
                        "target_fraction": adapters[checkpoint]["target_fraction"],
                        "actual_fraction": adapters[checkpoint]["actual_fraction"],
                        "adapter_sha256": adapters[checkpoint]["adapter_sha256"],
                        "benchmark_split": row["benchmark_split"],
                        "evaluation_id": row["evaluation_id"],
                        "data_idx": row["data_idx"],
                        "question_idx": row["question_idx"],
                        "country": row["country"],
                        "region_group": row["region_group"],
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "raw_output": raw,
                        "parsed_answer": parsed,
                        "correct": parsed == row["gold_answer"],
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"[{checkpoint}] wrote {min(start + len(batch), len(pending))}/"
                    f"{len(pending)} pending",
                    flush=True,
                )
        if completed_ids(result_path) != expected:
            raise RuntimeError(f"{checkpoint}: incomplete result coverage")
        summaries[checkpoint] = summarize(result_path)
        write_summary(output_dir, adapters, summaries)

    manifest.update(
        status="completed",
        completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 CULTURALBENCH GRID EVALUATION PASSED")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
