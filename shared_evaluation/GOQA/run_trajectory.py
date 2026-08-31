#!/usr/bin/env python3
"""Evaluate a base model and multiple unmerged LoRA adapters with one vLLM engine."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from goqa_common import (
    LABELS,
    PROMPT_VARIANTS,
    SYSTEM_PROMPT,
    build_messages,
    load_dataset,
    option_order,
    sha256_file,
    softmax,
)

OPTION_LOGPROBS = len(LABELS)
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_adapter(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not SAFE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError("adapter must use NAME=/path/to/adapter")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"adapter directory does not exist: {path}")
    return name, path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument(
        "--adapter", action="append", default=[], type=parse_adapter,
        metavar="NAME=PATH", help="Adapter to evaluate; repeat for a trajectory.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=here / "data/goqa_au_nz_india.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-batch-size", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                question_id = json.loads(line)["question_id"]
            except Exception as error:
                raise ValueError(f"Invalid existing result at {path}:{number}") from error
            if question_id in completed:
                raise ValueError(f"Duplicate existing result: {question_id}")
            completed.add(question_id)
    return completed


def prediction_manifest(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def main() -> int:
    args = parse_args()
    if not args.include_base and not args.adapter:
        raise ValueError("Select --include-base and/or at least one --adapter")
    names = (["base"] if args.include_base else []) + [name for name, _ in args.adapter]
    if len(names) != len(set(names)):
        raise ValueError("Trajectory model names must be unique")
    if args.request_batch_size < PROMPT_VARIANTS:
        raise ValueError("request-batch-size must be at least five")
    if args.max_model_len < 2:
        raise ValueError("max-model-len must be at least two")
    if args.max_num_batched_tokens < args.max_model_len:
        raise ValueError("max-num-batched-tokens must cover max-model-len")

    dataset = args.dataset.resolve()
    questions = load_dataset(dataset)
    expected = {row["question_id"] for row in questions}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = str(Path(args.model).expanduser().resolve())
    tokenizer_path = str(Path(args.tokenizer).expanduser().resolve()) if args.tokenizer else model
    models: list[dict[str, Any]] = []
    if args.include_base:
        models.append({"name": "base", "adapter": None})
    models.extend({"name": name, "adapter": str(path)} for name, path in args.adapter)

    engine_config = {
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_lora_rank": args.max_lora_rank,
        "request_batch_size": args.request_batch_size,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "logprobs_mode": "processed_logprobs",
    }
    trajectory_manifest = {
        "schema_version": 1,
        "benchmark": "GOQA-AU-NZ-India-five-prompt-trajectory",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "tokenizer": tokenizer_path,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "question_count": len(questions),
        "prompt_variants": PROMPT_VARIANTS,
        "prompt_method": "identity plus four SHA-256-seeded option permutations",
        "system_prompt": SYSTEM_PROMPT,
        "aggregation": "mean model probability after mapping to source option order",
        "distribution_method": "processed log probabilities over valid first-token labels",
        "models": models,
        "engine": engine_config,
    }
    trajectory_path = output_dir / "trajectory_manifest.json"
    if trajectory_path.exists():
        previous = json.loads(trajectory_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark", "model", "tokenizer", "dataset_sha256", "question_count",
            "prompt_variants", "prompt_method", "system_prompt", "aggregation",
            "distribution_method", "models", "engine",
        ):
            if previous.get(key) != trajectory_manifest.get(key):
                raise ValueError(f"Cannot resume because trajectory manifest changed: {key}")
    else:
        trajectory_path.write_text(json.dumps(trajectory_manifest, indent=2) + "\n")

    pending_by_model: dict[str, list[dict[str, Any]]] = {}
    for item in models:
        output = output_dir / item["name"] / "predictions.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = read_completed(output)
        if not completed.issubset(expected):
            raise ValueError(f"{item['name']}: output contains IDs outside this dataset")
        pending_by_model[item["name"]] = [
            row for row in questions if row["question_id"] not in completed
        ]
        print(
            f"[{item['name']}] complete={len(completed)} "
            f"pending={len(pending_by_model[item['name']])}", flush=True,
        )
    if not any(pending_by_model.values()):
        print(f"Trajectory inference already complete: {output_dir}")
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_options: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer_path,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "enable_lora": bool(args.adapter),
        "logprobs_mode": "processed_logprobs",
    }
    if args.adapter:
        llm_options.update(
            max_lora_rank=args.max_lora_rank,
            max_loras=1,
            max_cpu_loras=max(1, len(args.adapter)),
        )
    engine_started = time.monotonic()
    llm = LLM(**llm_options)
    engine_initialization_seconds = time.monotonic() - engine_started
    print(
        f"vLLM engine initialized in {engine_initialization_seconds:.1f} seconds",
        flush=True,
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

    question_batch_size = max(1, args.request_batch_size // PROMPT_VARIANTS)
    model_runtime_seconds: dict[str, float] = {}
    for model_index, item in enumerate(models, 1):
        name = item["name"]
        adapter = item["adapter"]
        pending = pending_by_model[name]
        if not pending:
            continue
        model_started = time.monotonic()
        output = output_dir / name / "predictions.jsonl"
        metadata = {
            **trajectory_manifest,
            "benchmark": "GOQA-AU-NZ-India-five-prompt",
            "run_name": name,
            "adapter": adapter,
        }
        metadata_path = prediction_manifest(output)
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        request = LoRARequest(name, model_index, adapter) if adapter else None
        with output.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), question_batch_size):
                batch = pending[start : start + question_batch_size]
                request_info: list[dict[str, Any]] = []
                messages = []
                sampling = []
                for batch_index, question in enumerate(batch):
                    count = len(question["options"])
                    for variant in range(PROMPT_VARIANTS):
                        order = option_order(question["question_id"], count, variant)
                        request_info.append(
                            {"batch_index": batch_index, "variant": variant,
                             "order": order, "count": count}
                        )
                        messages.append(build_messages(question, order))
                        sampling.append(
                            SamplingParams(
                                temperature=0.0,
                                max_tokens=1,
                                logprobs=OPTION_LOGPROBS,
                                allowed_token_ids=label_ids[:count],
                            )
                        )
                generated = llm.chat(
                    messages, sampling_params=sampling, lora_request=request, use_tqdm=True
                )
                if len(generated) != len(request_info):
                    raise RuntimeError("vLLM output count mismatch")
                positions = [dict(value.outputs[0].logprobs[0]) for value in generated]
                by_question: list[list[dict[str, Any]]] = [[] for _ in batch]
                for request_index, (info, value, position) in enumerate(
                    zip(request_info, generated, positions)
                ):
                    valid_ids = label_ids[: info["count"]]
                    missing = [token_id for token_id in valid_ids if token_id not in position]
                    if missing:
                        raise RuntimeError(
                            f"{name}: processed log probabilities omitted labels for "
                            f"request {request_index}: {missing}"
                        )
                    displayed = softmax([position[token_id].logprob for token_id in valid_ids])
                    source_order = [0.0] * info["count"]
                    for display_index, source_index in enumerate(info["order"]):
                        source_order[source_index] = displayed[display_index]
                    by_question[info["batch_index"]].append(
                        {
                            "variant": info["variant"],
                            "display_to_source_order": info["order"],
                            "source_order_distribution": source_order,
                            "selected_label": value.outputs[0].text.strip(),
                        }
                    )
                for question, variants in zip(batch, by_question):
                    count = len(question["options"])
                    averaged = [
                        statistics.fmean(
                            variant["source_order_distribution"][index]
                            for variant in variants
                        )
                        for index in range(count)
                    ]
                    if not math.isclose(sum(averaged), 1.0, abs_tol=1e-8):
                        raise RuntimeError(f"{name}/{question['question_id']}: invalid distribution")
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "run_name": name,
                                "question_id": question["question_id"],
                                "model_distribution": averaged,
                                "prompt_variants": variants,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"[{name}] wrote {min(start + len(batch), len(pending))}/"
                    f"{len(pending)} pending questions", flush=True,
                )
        if read_completed(output) != expected:
            raise RuntimeError(f"{name}: inference output is incomplete")
        completed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed_metadata.update(
            status="completed",
            completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            option_label_token_ids=dict(zip(LABELS, label_ids)),
            inference_seconds=time.monotonic() - model_started,
        )
        metadata_path.write_text(json.dumps(completed_metadata, indent=2) + "\n")
        model_runtime_seconds[name] = time.monotonic() - model_started
        print(
            f"[{name}] inference passed in {model_runtime_seconds[name]:.1f} seconds",
            flush=True,
        )

    trajectory_manifest.update(
        status="completed",
        completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        option_label_token_ids=dict(zip(LABELS, label_ids)),
        engine_initialization_seconds=engine_initialization_seconds,
        model_inference_seconds=model_runtime_seconds,
    )
    trajectory_path.write_text(json.dumps(trajectory_manifest, indent=2) + "\n")
    print(f"GOQA trajectory inference passed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
