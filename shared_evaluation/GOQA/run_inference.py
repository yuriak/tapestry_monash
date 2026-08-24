#!/usr/bin/env python3
"""Run standalone five-prompt GOQA inference with vLLM and optional LoRA."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
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

OPTION_LOGPROBS = 20


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path")
    parser.add_argument("--adapter", help="Optional Hugging Face-format LoRA directory")
    parser.add_argument("--run-name", default="model")
    parser.add_argument(
        "--dataset", type=Path, default=here / "data/goqa_au_nz_india.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=8192,
        help="Maximum number of prompt variants submitted per vLLM call.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
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


def manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.adapter and not Path(args.adapter).expanduser().is_dir():
        raise FileNotFoundError(args.adapter)
    if args.request_batch_size < PROMPT_VARIANTS:
        raise ValueError("request-batch-size must be at least five")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")

    questions = load_dataset(dataset)
    expected = {row["question_id"] for row in questions}
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = str(Path(args.adapter).expanduser().resolve()) if args.adapter else None
    manifest = {
        "schema_version": 1,
        "benchmark": "GOQA-AU-NZ-India-five-prompt",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "model": args.model,
        "adapter": adapter,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "question_count": len(questions),
        "prompt_variants": PROMPT_VARIANTS,
        "prompt_method": "identity plus four SHA-256-seeded option permutations",
        "system_prompt": SYSTEM_PROMPT,
        "aggregation": "mean model probability after mapping to source option order",
        "distribution_method": "softmax over constrained first-token option-label log probabilities",
        "inference": {
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_lora_rank": args.max_lora_rank if adapter else None,
            "request_batch_size": args.request_batch_size,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
        },
    }
    metadata_path = manifest_path(output)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark",
            "run_name",
            "model",
            "adapter",
            "dataset_sha256",
            "question_count",
            "prompt_variants",
            "prompt_method",
            "system_prompt",
            "aggregation",
            "distribution_method",
            "inference",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume because manifest changed: {key}")
    else:
        metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")

    completed = read_completed(output)
    if not completed.issubset(expected):
        raise ValueError("Existing output contains question IDs outside this dataset")
    pending = [row for row in questions if row["question_id"] not in completed]
    print(
        f"Dataset questions={len(questions)} complete={len(completed)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        print(f"Inference already complete: {output}")
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_options: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_lora": bool(adapter),
    }
    if adapter:
        llm_options.update(max_lora_rank=args.max_lora_rank, max_loras=1)
    llm = LLM(**llm_options)
    tokenizer = llm.get_tokenizer()
    label_ids = []
    for label in LABELS:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Option label {label!r} is not one token: {tokens}")
        label_ids.append(tokens[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Option labels do not map to unique tokens")

    lora_request = LoRARequest(args.run_name, 1, adapter) if adapter else None
    question_batch_size = max(1, args.request_batch_size // PROMPT_VARIANTS)
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
                        {
                            "batch_index": batch_index,
                            "variant": variant,
                            "order": order,
                            "count": count,
                        }
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
                messages,
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=True,
            )
            if len(generated) != len(request_info):
                raise RuntimeError("vLLM output count mismatch")
            positions = [dict(item.outputs[0].logprobs[0]) for item in generated]

            recovery_messages = []
            recovery_sampling = []
            recovery_targets = []
            for request_index, (info, position) in enumerate(zip(request_info, positions)):
                for token_id in label_ids[: info["count"]]:
                    if token_id not in position:
                        recovery_messages.append(messages[request_index])
                        recovery_sampling.append(
                            SamplingParams(
                                temperature=0.0,
                                max_tokens=1,
                                logprobs=0,
                                allowed_token_ids=[token_id],
                            )
                        )
                        recovery_targets.append((request_index, token_id))
            if recovery_messages:
                recovered = llm.chat(
                    recovery_messages,
                    sampling_params=recovery_sampling,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
                if len(recovered) != len(recovery_targets):
                    raise RuntimeError("vLLM fallback output count mismatch")
                for item, (request_index, token_id) in zip(recovered, recovery_targets):
                    position = item.outputs[0].logprobs[0]
                    if token_id not in position:
                        raise RuntimeError("Forced option-label log probability is missing")
                    positions[request_index][token_id] = position[token_id]

            by_question: list[list[dict[str, Any]]] = [[] for _ in batch]
            for info, item, position in zip(request_info, generated, positions):
                count = info["count"]
                displayed = softmax(
                    [position[token_id].logprob for token_id in label_ids[:count]]
                )
                source_order = [0.0] * count
                for display_index, source_index in enumerate(info["order"]):
                    source_order[source_index] = displayed[display_index]
                by_question[info["batch_index"]].append(
                    {
                        "variant": info["variant"],
                        "display_to_source_order": info["order"],
                        "source_order_distribution": source_order,
                        "selected_label": item.outputs[0].text.strip(),
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
                    raise RuntimeError(f"{question['question_id']}: invalid distribution")
                handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_name": args.run_name,
                            "question_id": question["question_id"],
                            "model_distribution": averaged,
                            "prompt_variants": variants,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"Wrote {min(start + len(batch), len(pending))}/{len(pending)} "
                "pending questions",
                flush=True,
            )

    if read_completed(output) != expected:
        raise RuntimeError("Inference output is incomplete")
    manifest.update(
        status="completed",
        completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        option_label_token_ids=dict(zip(LABELS, label_ids)),
    )
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"GOQA inference passed: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        raise
