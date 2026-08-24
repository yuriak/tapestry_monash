#!/usr/bin/env python3
"""Evaluate GlobalOpinionQA with a five-prompt option-order ensemble."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from checkpoint_grid import DEFAULT_FRACTIONS, resolve_checkpoint_grid
from evaluate_culturalbench_vllm import (
    MODEL_CHOICES,
    create_adapter_view,
    resolve_adapter,
    sha256,
)
from evaluate_global_opinion_qa_vllm import (
    LABELS,
    OPTION_LOGPROBS,
    SYSTEM_PROMPT,
    country_group,
    js_distance,
    load_questions,
    softmax_logprobs,
)

PROMPT_VARIANTS = 5
PAPER_COUNTRIES = {
    "United States": "United States",
    "Japan": "Japan",
    "Germany": "Germany",
    "China": "China",
}


def option_order(question_id: str, option_count: int, variant: int) -> list[int]:
    """Return a stable option permutation; variant zero preserves source order."""
    order = list(range(option_count))
    if variant == 0:
        return order
    seed_material = f"goqa-five-prompt-v1:{question_id}:{variant}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(seed).shuffle(order)
    return order


def build_prompt(question: dict[str, Any], order: list[int]) -> list[dict[str, str]]:
    option_lines = [
        f"({LABELS[position]}) {question['options'][source_index]}"
        for position, source_index in enumerate(order)
    ]
    user = "\n".join(
        [
            question["question"],
            "",
            "Here are the options:",
            *option_lines,
            "",
            "If you had to select one of the options, return only its letter.",
            "Answer:",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                result.add(json.loads(line)["question_id"])
            except Exception as error:
                raise ValueError(f"Invalid result line {path}:{number}") from error
    return result


def summarize(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    countries: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            for metric in row["country_metrics"]:
                countries[(row["source"], metric["country"])].append(
                    metric["js_distance"]
                )
                countries[("all", metric["country"])].append(metric["js_distance"])
                for source in ("all", row["source"]):
                    for group in ("all", metric["region_group"]):
                        buckets[(source, group)].append(
                            (metric["country"], metric["js_distance"])
                        )

    regional_rows = []
    for source in ("all", "GAS", "WVS"):
        for group in ("all", "australia_nz", "india", "rest_of_world"):
            values = buckets.get((source, group), [])
            by_country: dict[str, list[float]] = defaultdict(list)
            for country, value in values:
                by_country[country].append(value)
            regional_rows.append(
                {
                    "source": source,
                    "region_group": group,
                    "country_question_pairs": len(values),
                    "countries": len(by_country),
                    "pair_mean_js_distance": (
                        statistics.fmean(value for _, value in values)
                        if values
                        else None
                    ),
                    "country_macro_mean_js_distance": (
                        statistics.fmean(
                            statistics.fmean(group_values)
                            for group_values in by_country.values()
                        )
                        if by_country
                        else None
                    ),
                }
            )

    country_rows = []
    for (source, country), values in sorted(countries.items()):
        country_rows.append(
            {
                "source": source,
                "country": country,
                "questions": len(values),
                "mean_js_distance": statistics.fmean(values),
            }
        )
    return regional_rows, country_rows


def write_summaries(
    output_dir: Path,
    regional: dict[str, list[dict[str, Any]]],
    countries: dict[str, list[dict[str, Any]]],
) -> None:
    regional_fields = [
        "run",
        "source",
        "region_group",
        "country_question_pairs",
        "countries",
        "pair_mean_js_distance",
        "country_macro_mean_js_distance",
    ]
    regional_rows = [
        {"run": run, **row} for run, values in regional.items() for row in values
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=regional_fields)
        writer.writeheader()
        writer.writerows(regional_rows)

    country_fields = ["run", "source", "country", "questions", "mean_js_distance"]
    country_rows = [
        {"run": run, **row} for run, values in countries.items() for row in values
    ]
    with (output_dir / "country_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=country_fields)
        writer.writeheader()
        writer.writerows(country_rows)

    regional_index = {
        (row["run"], row["source"], row["region_group"]): row for row in regional_rows
    }
    country_index = {
        (row["run"], row["source"], row["country"]): row for row in country_rows
    }
    lines = [
        "# M0 GlobalOpinionQA Five-Prompt Results",
        "",
        (
            "Each question is scored under five deterministic option-order prompts. Candidate "
            "probabilities are mapped back to the source option order and averaged before "
            "Jensen–Shannon distance is calculated."
        ),
        "",
        "## Main regional view",
        "",
        "| Run | Overall | Australia/NZ | India | Rest of world |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in regional:
        values = [
            regional_index[(run, "all", group)]["country_macro_mean_js_distance"]
            for group in ("all", "australia_nz", "india", "rest_of_world")
        ]
        lines.append(
            "| "
            + run
            + " | "
            + " | ".join("n/a" if value is None else f"{value:.4f}" for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Countries reported by CultureInstruct",
            "",
            "| Run | United States | Japan | Germany | China |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run in countries:
        values = []
        for dataset_name in PAPER_COUNTRIES.values():
            row = country_index.get((run, "all", dataset_name))
            values.append("n/a" if row is None else f"{row['mean_js_distance']:.4f}")
        lines.append(f"| {run} | " + " | ".join(values) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=MODEL_CHOICES)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=16384,
        help="Maximum prompt variants submitted to one vLLM call.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Legacy question-batch override; each question expands to five requests.",
    )
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
        raise ValueError("Model or GlobalOpinionQA input is missing")
    if args.all_checkpoints and args.runs:
        raise ValueError("--all-checkpoints and --runs are mutually exclusive")
    if args.request_batch_size < 4096:
        raise ValueError("--request-batch-size must be at least 4096")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"
    view_root.mkdir(exist_ok=True)
    questions = load_questions(dataset)
    if args.all_checkpoints:
        adapters = resolve_checkpoint_grid(runtime, import_root, model, view_root)
    else:
        selected_runs = args.runs or MODEL_CHOICES
        adapters = {"base": {"view": None, "adapter_sha256": None}}
        for run in selected_runs:
            if run == "base":
                continue
            weights, completed = resolve_adapter(runtime, import_root, run)
            adapters[run] = {
                "view": str(create_adapter_view(view_root, run, weights, model)),
                "adapter_sha256": completed["adapter_sha256"],
            }
    model_names = list(adapters)
    question_batch_size = args.batch_size or max(
        1, args.request_batch_size // PROMPT_VARIANTS
    )

    manifest = {
        "schema_version": 1,
        "benchmark": "GlobalOpinionQA-five-prompt-option-order-ensemble",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "runs": model_names,
        "question_count": len(questions),
        "system_prompt": SYSTEM_PROMPT,
        "prompt_variants": PROMPT_VARIANTS,
        "checkpoint_fractions": list(DEFAULT_FRACTIONS) if args.all_checkpoints else None,
        "request_batch_size": args.request_batch_size,
        "question_batch_size": question_batch_size,
        "prompt_method": "identity plus four SHA-256-seeded option-order permutations",
        "aggregation": "mean option probability after mapping each permutation to source order",
        "distribution_method": "softmax over constrained first-token option-label log probabilities",
        "zero_human_distributions_excluded": sum(
            row["excluded_zero_distributions"] for row in questions
        ),
        "js_log_base": 2,
        "paper_country_mapping": PAPER_COUNTRIES,
        "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "benchmark",
            "model",
            "dataset_sha256",
            "runs",
            "question_count",
            "system_prompt",
            "prompt_variants",
            "checkpoint_fractions",
            "request_batch_size",
            "question_batch_size",
            "prompt_method",
            "aggregation",
            "adapters",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest changed: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Prepared {len(questions)} questions x {PROMPT_VARIANTS} prompts "
        f"for {len(model_names)} models",
        flush=True,
    )
    print(
        f"Prompt request pool per model: {len(questions) * PROMPT_VARIANTS}; "
        f"maximum per vLLM call: {question_batch_size * PROMPT_VARIANTS}",
        flush=True,
    )
    print(f"Output directory: {output_dir}", flush=True)
    if args.prepare_only:
        print("M0 GLOBALOPINIONQA FIVE-PROMPT PREPARATION PASSED")
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
        max_cpu_loras=max(40, len(adapters) - int("base" in adapters)),
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    label_ids = []
    for label in LABELS:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Option label {label!r} is not one token: {tokens}")
        label_ids.append(tokens[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Option labels do not map to unique tokens")

    expected = {row["question_id"] for row in questions}
    regional_summaries: dict[str, list[dict[str, Any]]] = {}
    country_summaries: dict[str, list[dict[str, Any]]] = {}
    for model_index, run in enumerate(model_names, 1):
        path = results_dir / f"{run}.jsonl"
        done = completed_ids(path)
        if not done.issubset(expected):
            raise ValueError(f"{run}: unexpected IDs in result")
        pending = [row for row in questions if row["question_id"] not in done]
        print(f"[{run}] complete={len(done)} pending={len(pending)}", flush=True)
        request = (
            None
            if run == "base"
            else LoRARequest(run, model_index, adapters[run]["view"])
        )
        with path.open("a") as handle:
            for start in range(0, len(pending), question_batch_size):
                batch = pending[start : start + question_batch_size]
                requests: list[dict[str, Any]] = []
                messages = []
                sampling = []
                for batch_index, row in enumerate(batch):
                    count = len(row["options"])
                    for variant in range(PROMPT_VARIANTS):
                        order = option_order(row["question_id"], count, variant)
                        requests.append(
                            {
                                "batch_index": batch_index,
                                "variant": variant,
                                "order": order,
                                "count": count,
                            }
                        )
                        messages.append(build_prompt(row, order))
                        sampling.append(
                            SamplingParams(
                                temperature=0.0,
                                max_tokens=1,
                                logprobs=OPTION_LOGPROBS,
                                allowed_token_ids=label_ids[:count],
                            )
                        )
                generated = llm.chat(
                    messages,
                    sampling_params=sampling,
                    lora_request=request,
                    use_tqdm=True,
                )
                if len(generated) != len(requests):
                    raise RuntimeError("vLLM output count mismatch")
                positions = [
                    dict(output.outputs[0].logprobs[0]) for output in generated
                ]

                fallback_messages = []
                fallback_sampling = []
                fallback_targets = []
                for request_index, (request_info, position) in enumerate(
                    zip(requests, positions)
                ):
                    for token in label_ids[: request_info["count"]]:
                        if token not in position:
                            fallback_messages.append(messages[request_index])
                            fallback_sampling.append(
                                SamplingParams(
                                    temperature=0.0,
                                    max_tokens=1,
                                    logprobs=0,
                                    allowed_token_ids=[token],
                                )
                            )
                            fallback_targets.append((request_index, token))
                if fallback_messages:
                    recovered = llm.chat(
                        fallback_messages,
                        sampling_params=fallback_sampling,
                        lora_request=request,
                        use_tqdm=False,
                    )
                    if len(recovered) != len(fallback_targets):
                        raise RuntimeError("vLLM fallback output count mismatch")
                    for output, (request_index, token) in zip(
                        recovered, fallback_targets
                    ):
                        position = output.outputs[0].logprobs[0]
                        if token not in position:
                            raise RuntimeError(
                                "Forced label log probability is missing"
                            )
                        positions[request_index][token] = position[token]
                    print(
                        f"[{run}] recovered {len(fallback_targets)} low-ranked option logprobs",
                        flush=True,
                    )

                variants_by_question: list[list[dict[str, Any]]] = [[] for _ in batch]
                for request_info, output, position in zip(
                    requests, generated, positions
                ):
                    count = request_info["count"]
                    displayed = softmax_logprobs(
                        [position[token].logprob for token in label_ids[:count]]
                    )
                    source_order = [0.0] * count
                    for display_index, source_index in enumerate(request_info["order"]):
                        source_order[source_index] = displayed[display_index]
                    variants_by_question[request_info["batch_index"]].append(
                        {
                            "variant": request_info["variant"],
                            "display_to_source_order": request_info["order"],
                            "display_distribution": displayed,
                            "source_order_distribution": source_order,
                            "selected_label": output.outputs[0].text.strip(),
                        }
                    )

                for row, variants in zip(batch, variants_by_question):
                    count = len(row["options"])
                    averaged = [
                        statistics.fmean(
                            variant["source_order_distribution"][index]
                            for variant in variants
                        )
                        for index in range(count)
                    ]
                    if not math.isclose(sum(averaged), 1.0, abs_tol=1e-8):
                        raise RuntimeError(
                            f"{run}/{row['question_id']}: invalid averaged distribution"
                        )
                    country_metrics = [
                        {
                            "country": country,
                            "region_group": country_group(country),
                            "human_distribution": human,
                            "js_distance": js_distance(averaged, human),
                        }
                        for country, human in row["human_distributions"].items()
                    ]
                    record = {
                        "run": run,
                        "adapter_sha256": adapters[run]["adapter_sha256"],
                        "question_id": row["question_id"],
                        "row_index": row["row_index"],
                        "source": row["source"],
                        "question": row["question"],
                        "options": row["options"],
                        "option_labels_in_source_order": list(LABELS[:count]),
                        "prompt_variants": variants,
                        "model_distribution": averaged,
                        "country_metrics": country_metrics,
                    }
                    if args.all_checkpoints:
                        record.update(
                            training_run=adapters[run]["training_run"],
                            step=adapters[run]["step"],
                            target_fraction=adapters[run]["target_fraction"],
                            actual_fraction=adapters[run]["actual_fraction"],
                        )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"[{run}] wrote {min(start + len(batch), len(pending))}/{len(pending)} pending",
                    flush=True,
                )
        if completed_ids(path) != expected:
            raise RuntimeError(f"{run}: incomplete result coverage")
        regional_summaries[run], country_summaries[run] = summarize(path)
        write_summaries(output_dir, regional_summaries, country_summaries)

    manifest.update(
        status="completed",
        completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        option_label_token_ids=dict(zip(LABELS, label_ids)),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 GLOBALOPINIONQA FIVE-PROMPT EVALUATION PASSED")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
