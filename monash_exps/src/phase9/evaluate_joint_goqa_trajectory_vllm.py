#!/usr/bin/env python3
"""Evaluate one shard of FL LoRA adapters with the shared five-prompt GOQA protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

OPTION_LOGPROBS = 20


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                question_id = json.loads(line)["question_id"]
            except Exception as error:
                raise ValueError(f"Invalid result at {path}:{number}") from error
            if question_id in ids:
                raise ValueError(f"Duplicate question ID in {path}: {question_id}")
            ids.add(question_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--goqa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--request-batch-size", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    goqa_dir = args.goqa_dir.resolve()
    sys.path.insert(0, str(goqa_dir))
    from goqa_common import (  # type: ignore
        LABELS,
        PROMPT_VARIANTS,
        SYSTEM_PROMPT,
        build_messages,
        load_dataset,
        option_order,
        sha256_file,
        softmax,
    )

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    base_model = Path(manifest["base_model"]).resolve()
    tokenizer_path = args.tokenizer.resolve()
    dataset = goqa_dir / "data" / "goqa_au_nz_india.jsonl"
    questions = load_dataset(dataset)
    expected = {row["question_id"] for row in questions}
    models: list[dict[str, Any]] = []
    if manifest.get("include_base"):
        models.append({"name": "base", "round": 0, "adapter": None, "delta_id": None})
    models.extend(manifest["adapters"])
    if not models:
        raise ValueError("Evaluation shard contains no models")
    if args.request_batch_size < PROMPT_VARIANTS:
        raise ValueError("Request batch size must be at least five")
    question_batch_size = max(1, args.request_batch_size // PROMPT_VARIANTS)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "benchmark": "GOQA-AU-NZ-India-five-prompt-trajectory",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path),
        "base_model": str(base_model),
        "tokenizer": str(tokenizer_path),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "question_count": len(questions),
        "prompt_variants": PROMPT_VARIANTS,
        "system_prompt": SYSTEM_PROMPT,
        "models": [item["name"] for item in models],
        "request_batch_size": args.request_batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
    }
    run_manifest_path = output_dir / f"gpu{manifest['shard']}_inference_manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text())
        for key in (
            "benchmark",
            "source_manifest",
            "base_model",
            "tokenizer",
            "dataset_sha256",
            "models",
            "request_batch_size",
            "max_num_batched_tokens",
            "max_num_seqs",
            "enforce_eager",
        ):
            if previous.get(key) != run_manifest.get(key):
                raise ValueError(f"Cannot resume: manifest changed: {key}")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    for item in models:
        if item.get("adapter") is not None:
            adapter = Path(item["adapter"])
            if not (adapter / "adapter_model.safetensors").is_file() or not (
                adapter / "adapter_config.json"
            ).is_file():
                raise FileNotFoundError(f"Incomplete adapter view: {adapter}")
    print(
        f"Prepared shard {manifest['shard']}: models={len(models)} "
        f"questions={len(questions)} prompts_per_model={len(questions) * PROMPT_VARIANTS}",
        flush=True,
    )
    if args.prepare_only:
        print(f"GOQA TRAJECTORY SHARD {manifest['shard']} PREPARATION PASSED")
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(base_model),
        tokenizer=str(tokenizer_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_cpu_loras=max(16, len(models)),
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
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

    for model_index, item in enumerate(models, 1):
        name = item["name"]
        result_dir = output_dir / name
        result_dir.mkdir(exist_ok=True)
        output = result_dir / "predictions.jsonl"
        done = completed_ids(output)
        if not done.issubset(expected):
            raise ValueError(f"{name}: unexpected question IDs")
        pending = [row for row in questions if row["question_id"] not in done]
        print(f"[{name}] complete={len(done)} pending={len(pending)}", flush=True)
        request = None if item.get("adapter") is None else LoRARequest(name, model_index, item["adapter"])
        with output.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), question_batch_size):
                batch = pending[start : start + question_batch_size]
                request_info = []
                messages = []
                sampling = []
                for batch_index, question in enumerate(batch):
                    count = len(question["options"])
                    for variant in range(PROMPT_VARIANTS):
                        order = option_order(question["question_id"], count, variant)
                        request_info.append({"batch_index": batch_index, "variant": variant, "order": order, "count": count})
                        messages.append(build_messages(question, order))
                        sampling.append(SamplingParams(temperature=0.0, max_tokens=1, logprobs=OPTION_LOGPROBS, allowed_token_ids=label_ids[:count]))
                generated = llm.chat(messages, sampling_params=sampling, lora_request=request, use_tqdm=True)
                positions = [dict(value.outputs[0].logprobs[0]) for value in generated]
                recovery_messages = []
                recovery_sampling = []
                recovery_targets = []
                for request_index, (info, position) in enumerate(zip(request_info, positions)):
                    for token_id in label_ids[: info["count"]]:
                        if token_id not in position:
                            recovery_messages.append(messages[request_index])
                            recovery_sampling.append(SamplingParams(temperature=0.0, max_tokens=1, logprobs=0, allowed_token_ids=[token_id]))
                            recovery_targets.append((request_index, token_id))
                if recovery_messages:
                    recovered = llm.chat(recovery_messages, sampling_params=recovery_sampling, lora_request=request, use_tqdm=False)
                    for value, (request_index, token_id) in zip(recovered, recovery_targets):
                        positions[request_index][token_id] = value.outputs[0].logprobs[0][token_id]
                by_question: list[list[dict[str, Any]]] = [[] for _ in batch]
                for info, value, position in zip(request_info, generated, positions):
                    displayed = softmax([position[token_id].logprob for token_id in label_ids[: info["count"]]])
                    source_order = [0.0] * info["count"]
                    for display_index, source_index in enumerate(info["order"]):
                        source_order[source_index] = displayed[display_index]
                    by_question[info["batch_index"]].append({"variant": info["variant"], "display_to_source_order": info["order"], "source_order_distribution": source_order, "selected_label": value.outputs[0].text.strip()})
                for question, variants in zip(batch, by_question):
                    count = len(question["options"])
                    averaged = [statistics.fmean(variant["source_order_distribution"][index] for variant in variants) for index in range(count)]
                    if not math.isclose(sum(averaged), 1.0, abs_tol=1e-8):
                        raise RuntimeError(f"{name}/{question['question_id']}: invalid distribution")
                    handle.write(json.dumps({"schema_version": 1, "run_name": name, "round": item["round"], "delta_id": item.get("delta_id"), "question_id": question["question_id"], "model_distribution": averaged, "prompt_variants": variants}, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(f"[{name}] wrote {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
        if completed_ids(output) != expected:
            raise RuntimeError(f"{name}: incomplete prediction coverage")

    run_manifest.update(status="completed", completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"), option_label_token_ids=dict(zip(LABELS, label_ids)))
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"GOQA TRAJECTORY SHARD {manifest['shard']} PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
