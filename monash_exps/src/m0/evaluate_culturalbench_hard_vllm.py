#!/usr/bin/env python3
"""Evaluate the base model and final M0 LoRAs on CulturalBench-Hard."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_culturalbench_vllm import (
    MODEL_CHOICES,
    RUN_SPECS,
    create_adapter_view,
    group_for_country,
    resolve_adapter,
    sha256,
)

SYSTEM_PROMPT = (
    "Judge whether the proposed answer correctly answers the cultural knowledge "
    "question. Return only TRUE or FALSE."
)
BOOL_RE = re.compile(r"(?<![A-Z])(TRUE|FALSE)(?![A-Z])", re.IGNORECASE)


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"data_idx", "question_idx", "prompt_question", "prompt_option", "answer", "country"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CulturalBench-Hard schema: {reader.fieldnames}")
        for index, row in enumerate(reader):
            answer = row["answer"].strip().upper()
            if answer not in {"TRUE", "FALSE"}:
                raise ValueError(f"Invalid answer at row {index}: {answer!r}")
            rows.append({
                "example_id": f"hard-{row['data_idx']}-{row['question_idx']}",
                "data_idx": row["data_idx"],
                "question_idx": row["question_idx"],
                "country": row["country"].strip(),
                "region_group": group_for_country(row["country"]),
                "question": row["prompt_question"].strip(),
                "proposed_answer": row["prompt_option"].strip(),
                "gold_answer": answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {row['prompt_question'].strip()}\nProposed answer: {row['prompt_option'].strip()}\nJudgment:"},
                ],
            })
    if len(rows) != 4908:
        raise ValueError(f"Expected 4,908 CulturalBench-Hard rows, found {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["question_idx"]].append(row)
    if len(groups) != 1227 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("CulturalBench-Hard must contain exactly four judgments for each of 1,227 questions")
    return rows


def parse_bool(text: str) -> str | None:
    match = BOOL_RE.search(text.strip().upper())
    return match.group(1).upper() if match else None


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                result.add(json.loads(line)["example_id"])
            except Exception as error:
                raise ValueError(f"Invalid result line {path}:{number}") from error
    return result


def summarize(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open()]
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["question_idx"]].append(row)
    output = []
    for region in ("overall", "australia_nz", "india", "rest_of_world"):
        judgments = rows if region == "overall" else [row for row in rows if row["region_group"] == region]
        question_groups = [
            group for group in by_question.values()
            if region == "overall" or group[0]["region_group"] == region
        ]
        exact = sum(all(item["correct"] for item in group) for group in question_groups)
        reconstructed_correct = 0
        reconstructed_valid = 0
        for group in question_groups:
            predicted_true = [item for item in group if item["parsed_answer"] == "TRUE"]
            gold_true = [item for item in group if item["gold_answer"] == "TRUE"]
            if len(predicted_true) == 1:
                reconstructed_valid += 1
                reconstructed_correct += len(gold_true) == 1 and predicted_true[0]["example_id"] == gold_true[0]["example_id"]
        output.append({
            "region_group": region,
            "judgments": len(judgments),
            "binary_correct": sum(item["correct"] for item in judgments),
            "binary_accuracy": sum(item["correct"] for item in judgments) / len(judgments) if judgments else None,
            "invalid_judgments": sum(item["parsed_answer"] is None for item in judgments),
            "questions": len(question_groups),
            "question_exact_match": exact / len(question_groups) if question_groups else None,
            "single_true_questions": reconstructed_valid,
            "reconstructed_mc_accuracy": reconstructed_correct / len(question_groups) if question_groups else None,
        })
    return output


def write_summary(output_dir: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    fields = ["run", "region_group", "judgments", "binary_correct", "binary_accuracy", "invalid_judgments", "questions", "question_exact_match", "single_true_questions", "reconstructed_mc_accuracy"]
    rows = [{"run": run, **row} for run, values in summaries.items() for row in values]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    indexed = {(row["run"], row["region_group"]): row for row in rows}
    lines = ["# M0 CulturalBench-Hard Results", "", "| Run | Binary accuracy | Question exact match | Reconstructed MC accuracy | Invalid judgments |", "|---|---:|---:|---:|---:|"]
    for run in summaries:
        row = indexed[(run, "overall")]
        lines.append(f"| {run} | {100*row['binary_accuracy']:.2f}% | {100*row['question_exact_match']:.2f}% | {100*row['reconstructed_mc_accuracy']:.2f}% | {row['invalid_judgments']} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, dataset = args.model.resolve(), args.dataset.resolve()
    runtime, import_root, output_dir = args.runtime_root.resolve(), args.import_root.resolve(), args.output_dir.resolve()
    if not (model / "config.json").is_file() or not dataset.is_file():
        raise ValueError("Model or CulturalBench-Hard input is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"; results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"; view_root.mkdir(exist_ok=True)
    examples = load_examples(dataset)
    adapters: dict[str, dict[str, Any]] = {"base": {"view": None, "adapter_sha256": None}}
    for run in args.runs:
        if run == "base": continue
        weights, completed = resolve_adapter(runtime, import_root, run)
        view = create_adapter_view(view_root, run, weights, model)
        adapters[run] = {"view": str(view), "adapter_sha256": completed["adapter_sha256"]}
    manifest = {
        "schema_version": 1, "benchmark": "CulturalBench-Hard",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model), "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "runs": args.runs, "example_count": len(examples), "system_prompt": SYSTEM_PROMPT,
        "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("benchmark", "model", "dataset_sha256", "runs", "example_count", "system_prompt", "adapters"):
            if previous.get(key) != manifest.get(key): raise ValueError(f"Cannot resume: manifest changed: {key}")
    else: manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    print(f"Prepared {len(examples)} judgments for {len(args.runs)} models", flush=True)
    llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=1, trust_remote_code=False,
              enable_lora=True, max_lora_rank=16, max_loras=1, max_cpu_loras=8,
              max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=4, stop=["\n"])
    expected = {row["example_id"] for row in examples}; summaries = {}
    for model_index, run in enumerate(args.runs, 1):
        path = results_dir / f"{run}.jsonl"; done = completed_ids(path)
        if not done.issubset(expected): raise ValueError(f"{run}: unexpected IDs in result")
        pending = [row for row in examples if row["example_id"] not in done]
        print(f"[{run}] complete={len(done)} pending={len(pending)}", flush=True)
        request = None if run == "base" else LoRARequest(run, model_index, adapters[run]["view"])
        with path.open("a") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start+args.batch_size]
                generated = llm.chat([row["messages"] for row in batch], sampling_params=sampling, lora_request=request, use_tqdm=True)
                if len(generated) != len(batch): raise RuntimeError("vLLM output count mismatch")
                for row, output in zip(batch, generated):
                    raw = output.outputs[0].text; parsed = parse_bool(raw)
                    record = {key: row[key] for key in ("example_id", "data_idx", "question_idx", "country", "region_group", "question", "proposed_answer", "gold_answer")}
                    record.update(run=run, adapter_sha256=adapters[run]["adapter_sha256"], raw_output=raw,
                                  parsed_answer=parsed, correct=parsed == row["gold_answer"], finish_reason=output.outputs[0].finish_reason)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(f"[{run}] wrote {min(start+len(batch),len(pending))}/{len(pending)} pending", flush=True)
        if completed_ids(path) != expected: raise RuntimeError(f"{run}: incomplete result coverage")
        summaries[run] = summarize(path); write_summary(output_dir, summaries)
    manifest.update(status="completed", completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 CULTURALBENCH-HARD VLLM EVALUATION PASSED")
    print(f"Report: {output_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr); raise
