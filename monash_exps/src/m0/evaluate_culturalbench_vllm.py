#!/usr/bin/env python3
"""Run CulturalBench-Easy on final M0 LoRA adapters with one vLLM engine."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


RUN_SPECS = {
    "local_south_asia": {"step": 1916, "source": "m3"},
    "local_variant_1": {"step": 1166, "source": "m3"},
    "local_variant_2": {"step": 5630, "source": "spartan"},
    "local_variant_3": {"step": 11238, "source": "m3"},
    "central_variant_1": {"step": 3082, "source": "spartan"},
    "central_variant_2": {"step": 7546, "source": "spartan"},
    "central_variant_3": {"step": 13154, "source": "m3"},
}
MODEL_CHOICES = ["base", *RUN_SPECS]
LETTERS = ("A", "B", "C", "D")
ANSWER_RE = re.compile(r"(?<![A-Z])([ABCD])(?![A-Z])", re.IGNORECASE)
SYSTEM_PROMPT = (
    "Answer the multiple-choice cultural knowledge question. "
    "Return only the single letter A, B, C, or D."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def group_for_country(country: str) -> str:
    normalized = country.strip().casefold()
    if normalized in {"australia", "new zealand"}:
        return "australia_nz"
    if normalized == "india":
        return "india"
    return "rest_of_world"


def load_examples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    examples = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "data_idx", "question_idx", "prompt_question", "prompt_option_a",
            "prompt_option_b", "prompt_option_c", "prompt_option_d", "answer", "country",
        }
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CulturalBench-Easy schema: {reader.fieldnames}")
        for row_number, row in enumerate(reader):
            answer = row["answer"].strip().upper()
            if answer not in LETTERS:
                raise ValueError(f"Invalid answer at row {row_number}: {answer!r}")
            question = row["prompt_question"].strip()
            options = {letter: row[f"prompt_option_{letter.lower()}"].strip() for letter in LETTERS}
            user_prompt = "\n".join([
                question,
                "",
                *(f"{letter}. {options[letter]}" for letter in LETTERS),
                "",
                "Answer:",
            ])
            examples.append({
                "example_id": f"easy-{row['data_idx']}-{row['question_idx']}",
                "data_idx": row["data_idx"],
                "question_idx": row["question_idx"],
                "country": row["country"].strip(),
                "region_group": group_for_country(row["country"]),
                "question": question,
                "options": options,
                "gold_answer": answer,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            })
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("CulturalBench input produced zero examples")
    return examples


def resolve_adapter(runtime: Path, import_root: Path, run: str) -> tuple[Path, dict[str, str]]:
    spec = RUN_SPECS[run]
    run_dir = runtime / run if spec["source"] == "m3" else import_root / run
    completed_path = run_dir / "COMPLETED"
    if not completed_path.is_file():
        raise ValueError(f"{run}: missing COMPLETED at {run_dir}")
    completed = read_kv(completed_path)
    if (
        completed.get("status") != "completed"
        or completed.get("run") != run
        or int(completed.get("step", -1)) != spec["step"]
    ):
        raise ValueError(f"{run}: invalid completion record")
    adapter_dir = run_dir / "adapter_history" / f"step_{spec['step']:07d}"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not (adapter_dir / ".complete").is_file() or not adapter_path.is_file():
        raise ValueError(f"{run}: final adapter is incomplete")
    digest = sha256(adapter_path)
    if digest != completed.get("adapter_sha256"):
        raise ValueError(f"{run}: final adapter SHA-256 mismatch")
    return adapter_path, completed


def create_adapter_view(view_root: Path, run: str, weights: Path, model: Path) -> Path:
    view = view_root / run
    view.mkdir(parents=True, exist_ok=True)
    link = view / "adapter_model.safetensors"
    if link.is_symlink() or link.exists():
        if link.resolve() != weights.resolve():
            raise ValueError(f"Existing adapter view points to the wrong weights: {link}")
    else:
        link.symlink_to(weights.resolve())
    config = {
        "base_model_name_or_path": str(model.resolve()),
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 64,
        "lora_dropout": 0.03,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
    }
    config_path = view / "adapter_config.json"
    rendered = json.dumps(config, indent=2) + "\n"
    if config_path.exists() and config_path.read_text() != rendered:
        raise ValueError(f"Existing adapter view has a conflicting config: {config_path}")
    config_path.write_text(rendered)
    return view


def parse_answer(text: str) -> str | None:
    cleaned = text.strip().upper()
    if cleaned in LETTERS:
        return cleaned
    match = ANSWER_RE.search(cleaned)
    return match.group(1).upper() if match else None


def load_completed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid resume JSONL at {path}:{number}") from error
            ids.add(item["example_id"])
    return ids


def summarize_run(path: Path) -> list[dict[str, Any]]:
    counts: dict[str, list[int]] = {group: [0, 0, 0] for group in ("overall", "australia_nz", "india", "rest_of_world")}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            for group in ("overall", row["region_group"]):
                counts[group][0] += 1
                counts[group][1] += int(row["correct"])
                counts[group][2] += int(row["parsed_answer"] is None)
    rows = []
    for group, (total, correct, invalid) in counts.items():
        rows.append({
            "region_group": group,
            "examples": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "invalid_outputs": invalid,
        })
    return rows


def write_summary(output_dir: Path, run_summaries: dict[str, list[dict[str, Any]]]) -> None:
    rows = []
    for run, summaries in run_summaries.items():
        rows.extend({"run": run, **summary} for summary in summaries)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "region_group", "examples", "correct", "accuracy", "invalid_outputs"])
        writer.writeheader()
        writer.writerows(rows)
    overall = {row["run"]: row for row in rows if row["region_group"] == "overall"}
    lines = [
        "# M0 CulturalBench-Easy Results",
        "",
        "| Run | Overall | Australia/NZ | India | Rest of world | Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    indexed = {(row["run"], row["region_group"]): row for row in rows}
    for run in run_summaries:
        def accuracy(group: str) -> str:
            value = indexed[(run, group)]["accuracy"]
            return "n/a" if value is None else f"{100 * value:.2f}%"
        lines.append(
            f"| {run} | {accuracy('overall')} | {accuracy('australia_nz')} | "
            f"{accuracy('india')} | {accuracy('rest_of_world')} | {overall[run]['invalid_outputs']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=MODEL_CHOICES, default=list(RUN_SPECS))
    parser.add_argument("--limit", type=int)
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
    if not (model / "config.json").is_file():
        raise ValueError(f"Invalid local model directory: {model}")
    if not dataset.is_file():
        raise ValueError(f"Missing CulturalBench dataset: {dataset}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    view_root = output_dir / ".adapter_views"
    view_root.mkdir(exist_ok=True)
    examples = load_examples(dataset, args.limit)
    adapters: dict[str, dict[str, Any]] = {}
    for run in args.runs:
        if run == "base":
            adapters[run] = {
                "view": None,
                "weights": None,
                "adapter_sha256": None,
                "step": 0,
                "source_cluster": "base",
            }
            continue
        weights, completed = resolve_adapter(runtime, import_root, run)
        view = create_adapter_view(view_root, run, weights, model)
        adapters[run] = {
            "view": str(view),
            "weights": str(weights),
            "adapter_sha256": completed["adapter_sha256"],
            "step": RUN_SPECS[run]["step"],
            "source_cluster": RUN_SPECS[run]["source"],
        }

    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark": "CulturalBench-Easy",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "model": str(model),
        "runs": args.runs,
        "limit": args.limit,
        "example_count": len(examples),
        "system_prompt": SYSTEM_PROMPT,
        "generation": {"temperature": 0.0, "max_tokens": 8},
        "adapters": adapters,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("benchmark", "dataset_sha256", "model", "runs", "limit", "example_count", "system_prompt", "generation", "adapters"):
            if old.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest field changed: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Prepared {len(examples)} examples and {len(adapters)} final adapters")
    print(f"Output directory: {output_dir}")
    if args.prepare_only:
        print("M0 CULTURALBENCH PREPARATION PASSED")
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("Loading one OLMo 2 7B vLLM engine...", flush=True)
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_cpu_loras=max(8, len(adapters) - int("base" in adapters)),
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=8, stop=["\n"])
    run_summaries: dict[str, list[dict[str, Any]]] = {}
    expected_ids = {item["example_id"] for item in examples}
    for adapter_id, run in enumerate(args.runs, 1):
        result_path = results_dir / f"{run}.jsonl"
        completed_ids = load_completed_ids(result_path)
        if not completed_ids.issubset(expected_ids):
            raise ValueError(f"{run}: result JSONL contains unexpected example IDs")
        pending = [item for item in examples if item["example_id"] not in completed_ids]
        print(f"[{run}] complete={len(completed_ids)} pending={len(pending)}", flush=True)
        request = None if run == "base" else LoRARequest(run, adapter_id, adapters[run]["view"])
        with result_path.open("a") as output_handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                outputs = llm.chat(
                    [item["messages"] for item in batch],
                    sampling_params=sampling,
                    use_tqdm=True,
                    lora_request=request,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(f"{run}: vLLM output count mismatch")
                for item, generated in zip(batch, outputs):
                    text = generated.outputs[0].text
                    parsed = parse_answer(text)
                    record = {
                        "run": run,
                        "adapter_sha256": adapters[run]["adapter_sha256"],
                        "example_id": item["example_id"],
                        "data_idx": item["data_idx"],
                        "question_idx": item["question_idx"],
                        "country": item["country"],
                        "region_group": item["region_group"],
                        "question": item["question"],
                        "options": item["options"],
                        "gold_answer": item["gold_answer"],
                        "raw_output": text,
                        "parsed_answer": parsed,
                        "correct": parsed == item["gold_answer"],
                        "finish_reason": generated.outputs[0].finish_reason,
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                print(f"[{run}] wrote {min(start + len(batch), len(pending))}/{len(pending)} pending examples", flush=True)
        final_ids = load_completed_ids(result_path)
        if final_ids != expected_ids:
            raise RuntimeError(f"{run}: final result coverage mismatch")
        run_summaries[run] = summarize_run(result_path)
        write_summary(output_dir, run_summaries)

    manifest["completed_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("M0 CULTURALBENCH VLLM EVALUATION PASSED")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
