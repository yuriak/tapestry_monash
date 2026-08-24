#!/usr/bin/env python3
"""Score GlobalOpinionQA answer distributions with base and final M0 LoRAs."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_culturalbench_vllm import MODEL_CHOICES, create_adapter_view, resolve_adapter, sha256

SYSTEM_PROMPT = (
    "Answer the survey question by choosing exactly one of the listed options. "
    "Return only the option letter."
)
LABELS = tuple(chr(ord("A") + index) for index in range(18))
# Request vLLM's default maximum first. Labels outside that full-vocabulary
# top-k are recovered below with a one-token forced request; vLLM returns the
# sampled token's raw log probability even when ``logprobs=0``.
OPTION_LOGPROBS = 20


def parse_selections(text: str) -> dict[str, list[float]]:
    cleaned = re.sub(r"^defaultdict\(<class 'list'>, ", "", text)
    if cleaned != text and cleaned.endswith(")"):
        cleaned = cleaned[:-1]
    value = ast.literal_eval(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Selections field is not a dictionary")
    return value


def normalize(values: list[float]) -> list[float] | None:
    numbers = [max(0.0, float(value)) for value in values]
    total = sum(numbers)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("Invalid probability distribution")
    if total <= 0:
        return None
    return [value / total for value in numbers]


def country_group(country: str) -> str:
    name = country.strip().casefold()
    if name in {"australia", "new zealand"}:
        return "australia_nz"
    if name.startswith("india"):
        return "india"
    return "rest_of_world"


def js_distance(p: list[float], q: list[float]) -> float:
    midpoint = [(left + right) / 2 for left, right in zip(p, q)]
    def kl(left: list[float], right: list[float]) -> float:
        return sum(value * math.log2(value / target) for value, target in zip(left, right) if value > 0)
    return math.sqrt((kl(p, midpoint) + kl(q, midpoint)) / 2)


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["question", "selections", "options", "source"]:
            raise ValueError(f"Unexpected GlobalOpinionQA schema: {reader.fieldnames}")
        for index, row in enumerate(reader):
            options = ast.literal_eval(row["options"])
            selections = parse_selections(row["selections"])
            if not 2 <= len(options) <= len(LABELS):
                raise ValueError(f"Question {index} has unsupported option count {len(options)}")
            human = {}
            excluded_zero_distributions = 0
            for country, values in selections.items():
                if len(values) != len(options):
                    raise ValueError(f"Question {index}/{country}: distribution length mismatch")
                distribution = normalize(values)
                if distribution is None:
                    excluded_zero_distributions += 1
                else:
                    human[country] = distribution
            option_lines = [f"{LABELS[i]}. {option}" for i, option in enumerate(options)]
            questions.append({
                "question_id": f"goqa-{index:04d}", "row_index": index,
                "question": row["question"].strip(), "options": options,
                "source": row["source"].strip(), "human_distributions": human,
                "excluded_zero_distributions": excluded_zero_distributions,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join([row["question"].strip(), "", *option_lines, "", "Answer:"])},
                ],
            })
    if len(questions) != 2556:
        raise ValueError(f"Expected 2,556 GlobalOpinionQA questions, found {len(questions)}")
    return questions


def completed_ids(path: Path) -> set[str]:
    if not path.exists(): return set()
    ids = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try: ids.add(json.loads(line)["question_id"])
            except Exception as error: raise ValueError(f"Invalid result {path}:{number}") from error
    return ids


def softmax_logprobs(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def summarize(path: Path) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            for metric in row["country_metrics"]:
                for source in ("all", row["source"]):
                    for group in ("all", metric["region_group"]):
                        buckets[(source, group)].append((metric["country"], metric["js_distance"]))
    output = []
    for source in ("all", "GAS", "WVS"):
        for group in ("all", "australia_nz", "india", "rest_of_world"):
            values = buckets.get((source, group), [])
            by_country: dict[str, list[float]] = defaultdict(list)
            for country, value in values: by_country[country].append(value)
            output.append({
                "source": source, "region_group": group,
                "country_question_pairs": len(values), "countries": len(by_country),
                "pair_mean_js_distance": statistics.fmean(value for _, value in values) if values else None,
                "country_macro_mean_js_distance": statistics.fmean(statistics.fmean(group_values) for group_values in by_country.values()) if by_country else None,
            })
    return output


def write_summary(output_dir: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    fields = ["run", "source", "region_group", "country_question_pairs", "countries", "pair_mean_js_distance", "country_macro_mean_js_distance"]
    rows = [{"run": run, **row} for run, values in summaries.items() for row in values]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    indexed = {(row["run"], row["source"], row["region_group"]): row for row in rows}
    lines = ["# M0 GlobalOpinionQA Results", "", "Lower Jensen–Shannon distance is better. Values below are country-macro means across both survey sources.", "", "| Run | Overall | Australia/NZ | India | Rest of world |", "|---|---:|---:|---:|---:|"]
    for run in summaries:
        values = [indexed[(run, "all", group)]["country_macro_mean_js_distance"] for group in ("all", "australia_nz", "india", "rest_of_world")]
        lines.append("| " + run + " | " + " | ".join("n/a" if value is None else f"{value:.4f}" for value in values) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True); parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); model, dataset = args.model.resolve(), args.dataset.resolve()
    runtime, import_root, output_dir = args.runtime_root.resolve(), args.import_root.resolve(), args.output_dir.resolve()
    if not (model / "config.json").is_file() or not dataset.is_file(): raise ValueError("Model or GlobalOpinionQA input is missing")
    output_dir.mkdir(parents=True, exist_ok=True); results_dir = output_dir / "results"; results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"; view_root.mkdir(exist_ok=True)
    questions = load_questions(dataset)
    adapters: dict[str, dict[str, Any]] = {"base": {"view": None, "adapter_sha256": None}}
    for run in args.runs:
        if run == "base": continue
        weights, completed = resolve_adapter(runtime, import_root, run)
        adapters[run] = {"view": str(create_adapter_view(view_root, run, weights, model)), "adapter_sha256": completed["adapter_sha256"]}
    manifest = {
        "schema_version": 1, "benchmark": "GlobalOpinionQA",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model), "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "runs": args.runs, "question_count": len(questions), "system_prompt": SYSTEM_PROMPT,
        "zero_human_distributions_excluded": sum(row["excluded_zero_distributions"] for row in questions),
        "distribution_method": "softmax over constrained first-token option-label log probabilities",
        "option_logprobs_requested": OPTION_LOGPROBS,
        "js_log_base": 2, "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("benchmark", "model", "dataset_sha256", "runs", "question_count", "system_prompt", "distribution_method", "adapters"):
            if previous.get(key) != manifest.get(key): raise ValueError(f"Cannot resume: manifest changed: {key}")
    else: manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    print(f"Prepared {len(questions)} questions for {len(args.runs)} models", flush=True)
    llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=1, trust_remote_code=False,
              enable_lora=True, max_lora_rank=16, max_loras=1, max_cpu_loras=8,
              max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    tokenizer = llm.get_tokenizer(); label_ids = []
    for label in LABELS:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1: raise ValueError(f"Option label {label!r} is not one token: {tokens}")
        label_ids.append(tokens[0])
    if len(set(label_ids)) != len(label_ids): raise ValueError("Option labels do not map to unique tokens")
    expected = {row["question_id"] for row in questions}; summaries = {}
    for model_index, run in enumerate(args.runs, 1):
        path = results_dir / f"{run}.jsonl"; done = completed_ids(path)
        if not done.issubset(expected): raise ValueError(f"{run}: unexpected IDs in result")
        pending = [row for row in questions if row["question_id"] not in done]
        print(f"[{run}] complete={len(done)} pending={len(pending)}", flush=True)
        request = None if run == "base" else LoRARequest(run, model_index, adapters[run]["view"])
        with path.open("a") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start+args.batch_size]
                sampling = [SamplingParams(temperature=0.0, max_tokens=1, logprobs=OPTION_LOGPROBS, allowed_token_ids=label_ids[:len(row["options"])]) for row in batch]
                generated = llm.chat([row["messages"] for row in batch], sampling_params=sampling, lora_request=request, use_tqdm=True)
                if len(generated) != len(batch): raise RuntimeError("vLLM output count mismatch")
                positions = [dict(output.outputs[0].logprobs[0]) for output in generated]
                fallback_messages, fallback_sampling, fallback_targets = [], [], []
                for batch_index, (row, position) in enumerate(zip(batch, positions)):
                    for token in label_ids[:len(row["options"])]:
                        if token not in position:
                            fallback_messages.append(row["messages"])
                            fallback_sampling.append(SamplingParams(
                                temperature=0.0, max_tokens=1, logprobs=0,
                                allowed_token_ids=[token],
                            ))
                            fallback_targets.append((batch_index, token, row["question_id"]))
                if fallback_messages:
                    recovered = llm.chat(
                        fallback_messages, sampling_params=fallback_sampling,
                        lora_request=request, use_tqdm=False,
                    )
                    if len(recovered) != len(fallback_targets):
                        raise RuntimeError("vLLM fallback output count mismatch")
                    for output, (batch_index, token, question_id) in zip(recovered, fallback_targets):
                        position = output.outputs[0].logprobs[0]
                        if token not in position:
                            raise RuntimeError(
                                f"{run}/{question_id}: forced label logprob missing: {token}"
                            )
                        positions[batch_index][token] = position[token]
                    print(f"[{run}] recovered {len(fallback_targets)} low-ranked option logprobs", flush=True)
                for row, output, position in zip(batch, generated, positions):
                    count = len(row["options"])
                    probabilities = softmax_logprobs([position[token].logprob for token in label_ids[:count]])
                    country_metrics = []
                    for country, human in row["human_distributions"].items():
                        country_metrics.append({"country": country, "region_group": country_group(country), "human_distribution": human, "js_distance": js_distance(probabilities, human)})
                    record = {
                        "run": run, "adapter_sha256": adapters[run]["adapter_sha256"],
                        "question_id": row["question_id"], "row_index": row["row_index"], "source": row["source"],
                        "question": row["question"], "options": row["options"],
                        "option_labels": list(LABELS[:count]), "model_distribution": probabilities,
                        "selected_label": output.outputs[0].text.strip(), "country_metrics": country_metrics,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(f"[{run}] wrote {min(start+len(batch),len(pending))}/{len(pending)} pending", flush=True)
        if completed_ids(path) != expected: raise RuntimeError(f"{run}: incomplete result coverage")
        summaries[run] = summarize(path); write_summary(output_dir, summaries)
    manifest.update(status="completed", completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"), option_label_token_ids=dict(zip(LABELS,label_ids)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 GLOBALOPINIONQA VLLM EVALUATION PASSED")
    print(f"Report: {output_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr); raise
