#!/usr/bin/env python3
"""Audit OLMo 2 tokenizers and measure M0 chat-token length distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from bhaskera.data.formats import render_with_format
from transformers import AutoTokenizer

SCHEMA_VERSION = 1
AUDIT_VERSION = "m0-olmo2-chatml-lengths-v1"
DEFAULT_CANDIDATES = (512, 1024, 2048, 4096)
SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a concise cultural assistant."},
    {"role": "user", "content": "Name one Australian cultural practice."},
    {
        "role": "assistant",
        "content": "A common example is acknowledging Country at formal events.",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def tokenizer_identity(model_path: Path, workspace_root: Path) -> tuple[Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        SAMPLE_MESSAGES,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Bhaskera's chatml path renders first and then calls tokenizer() with its
    # default add_special_tokens=True behaviour.
    bhaskera_ids = tokenizer(
        rendered,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    direct_ids = tokenizer.apply_chat_template(
        SAMPLE_MESSAGES,
        tokenize=True,
        add_generation_prompt=False,
    )
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ):
        path = model_path / name
        if path.is_file():
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    identity = {
        "path": portable(model_path, workspace_root),
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_length": len(tokenizer),
        "vocab_sha256": sha256_json(tokenizer.get_vocab()),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
        "chat_template": tokenizer.chat_template,
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest(),
        "sample_messages": SAMPLE_MESSAGES,
        "sample_rendered": rendered,
        "sample_bhaskera_ids": bhaskera_ids,
        "sample_direct_template_ids": direct_ids,
        "sample_paths_equivalent": bhaskera_ids == direct_ids,
        "files": files,
    }
    return tokenizer, identity


def percentile(lengths: np.ndarray, quantile: float) -> int:
    return int(np.quantile(lengths, quantile, method="nearest"))


def candidate_metrics(lengths: np.ndarray, candidate: int) -> dict[str, Any]:
    clipped = np.minimum(lengths, candidate)
    retained = int(clipped.sum(dtype=np.int64))
    original = int(lengths.sum(dtype=np.int64))
    capacity = int(lengths.size * candidate)
    truncated_rows = int(np.count_nonzero(lengths > candidate))
    return {
        "seq_len": candidate,
        "truncated_rows": truncated_rows,
        "truncation_rate": truncated_rows / int(lengths.size),
        "retained_tokens": retained,
        "retained_token_fraction": retained / original,
        "discarded_tokens": original - retained,
        "padding_tokens": capacity - retained,
        "padding_fraction": (capacity - retained) / capacity,
        "utilized_fraction": retained / capacity,
    }


def tokenize_text_batch(tokenizer: Any, texts: list[str]) -> list[int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    return [int(value) for value in encoded["length"]]


def analyze_view(
    path: Path,
    tokenizer: Any,
    candidates: tuple[int, ...],
    batch_size: int,
    expected_rows: int | None = None,
    progress_every: int = 25_000,
) -> dict[str, Any]:
    lengths: list[int] = []
    texts: list[str] = []
    started = time.monotonic()

    def flush() -> None:
        if not texts:
            return
        lengths.extend(tokenize_text_batch(tokenizer, texts))
        texts.clear()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            row = json.loads(line)
            rendered = render_with_format(
                "chatml",
                row,
                tokenizer,
                {"messages_field": "messages"},
            )
            if not rendered:
                raise ValueError(f"{path}:{line_number}: empty rendered chat")
            texts.append(rendered)
            if len(texts) >= batch_size:
                flush()
            if progress_every and line_number % progress_every == 0:
                print(f"    {line_number} rows", flush=True)
    flush()

    if not lengths:
        raise RuntimeError(f"view is empty: {path}")
    if expected_rows is not None and len(lengths) != expected_rows:
        raise RuntimeError(
            f"row mismatch for {path}: tokenized={len(lengths)}, expected={expected_rows}"
        )

    values = np.asarray(lengths, dtype=np.int64)
    histogram = Counter(int(value) for value in values)
    return {
        "rows": int(values.size),
        "total_untruncated_tokens": int(values.sum(dtype=np.int64)),
        "mean": float(values.mean()),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": int(values.max()),
        "histogram": {str(key): count for key, count in sorted(histogram.items())},
        "candidates": {
            str(candidate): candidate_metrics(values, candidate)
            for candidate in candidates
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def equivalent_tokenizers(first: dict[str, Any], second: dict[str, Any]) -> bool:
    fields = (
        "class",
        "vocab_size",
        "tokenizer_length",
        "vocab_sha256",
        "bos_token",
        "bos_token_id",
        "eos_token",
        "eos_token_id",
        "pad_token",
        "pad_token_id",
        "unk_token",
        "unk_token_id",
        "padding_side",
        "truncation_side",
        "chat_template_sha256",
        "sample_rendered",
        "sample_bhaskera_ids",
        "sample_direct_template_ids",
    )
    return all(first[field] == second[field] for field in fields)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# M0 OLMo 2 Tokenizer and Length Audit",
        "",
        f"Generated: `{report['created_at']}`",
        "",
        (
            "The measurements reproduce Bhaskera's `chatml` renderer and its SFT "
            "tokenization path: the tokenizer chat template is rendered without a "
            "generation prompt, then tokenized with right truncation and fixed-length "
            "padding. This report measures untruncated lengths; each candidate column "
            "models the later truncation and padding exactly."
        ),
        "",
        "## Tokenizer equivalence",
        "",
        (
            f"OLMo 2 7B and 1B tokenizer-equivalent: "
            f"**{str(report['tokenizers']['equivalent']).lower()}**."
        ),
        "",
        "## Length distributions",
        "",
        "| View | Rows | Mean | p50 | p90 | p95 | p99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(report["views"].items()):
        lines.append(
            f"| `{name}` | {stats['rows']:,} | {stats['mean']:.1f} | "
            f"{stats['p50']} | {stats['p90']} | {stats['p95']} | "
            f"{stats['p99']} | {stats['max']} |"
        )

    for candidate in report["candidates"]:
        lines += [
            "",
            f"## Candidate sequence length: {candidate}",
            "",
            "| View | Truncated rows | Truncation | Retained tokens | Padding | Utilized |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, stats in sorted(report["views"].items()):
            metric = stats["candidates"][str(candidate)]
            lines.append(
                f"| `{name}` | {metric['truncated_rows']:,} | "
                f"{metric['truncation_rate']:.2%} | "
                f"{metric['retained_token_fraction']:.2%} | "
                f"{metric['padding_fraction']:.2%} | "
                f"{metric['utilized_fraction']:.2%} |"
            )
    lines += [
        "",
        "## Decision status",
        "",
        (
            "This audit records measurements only. The reviewed selection is stored "
            "separately in `sequence-length-decision.json`; no production token cache "
            "was generated by this audit."
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    inferred_workspace = Path(__file__).resolve().parents[3]
    runtime = inferred_workspace / "monash_exps" / ".runtime"
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=runtime / "data" / "m0" / "prepared",
    )
    parser.add_argument(
        "--model-7b",
        type=Path,
        default=runtime / "models" / "m0" / "OLMo-2-1124-7B-Instruct",
    )
    parser.add_argument(
        "--model-1b",
        type=Path,
        default=runtime / "models" / "m0" / "OLMo-2-0425-1B-Instruct-metadata",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=runtime / "manifests" / "m0" / "tokenizer-length-audit.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=runtime / "manifests" / "m0" / "tokenizer-length-audit.md",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    candidates = tuple(sorted(set(args.candidates)))
    if not candidates or candidates[0] <= 0:
        parser.error("--candidates must contain positive lengths")

    workspace_root = args.workspace_root.resolve()
    prepared_root = args.prepared_root.resolve()
    prepared_manifest_path = prepared_root / "manifests" / "prepared-data.json"
    if not prepared_manifest_path.is_file():
        parser.error(f"missing prepared manifest: {prepared_manifest_path}")
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest_sha256 = sha256_file(prepared_manifest_path)

    print("Loading and comparing local OLMo 2 tokenizers...", flush=True)
    tokenizer_7b, identity_7b = tokenizer_identity(args.model_7b.resolve(), workspace_root)
    _, identity_1b = tokenizer_identity(args.model_1b.resolve(), workspace_root)
    tokenizers_equivalent = equivalent_tokenizers(identity_7b, identity_1b)
    if not tokenizers_equivalent:
        raise RuntimeError("OLMo 2 7B and 1B tokenizers are not runtime-equivalent")
    if not identity_7b["sample_paths_equivalent"]:
        raise RuntimeError("Bhaskera render/tokenize path differs from direct chat template")
    print("  runtime tokenizer equivalence passed", flush=True)

    views = {}
    for name, view_manifest in sorted(prepared_manifest["views"].items()):
        path = workspace_root / view_manifest["path"]
        if not path.is_file() or sha256_file(path) != view_manifest["sha256"]:
            raise RuntimeError(f"prepared view hash mismatch: {name}")
        print(f"Measuring {name} ({view_manifest['rows']} rows)...", flush=True)
        stats = analyze_view(
            path,
            tokenizer_7b,
            candidates,
            args.batch_size,
            expected_rows=view_manifest["rows"],
        )
        stats["path"] = view_manifest["path"]
        stats["prepared_sha256"] = view_manifest["sha256"]
        views[name] = stats

    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prepared_manifest": portable(prepared_manifest_path, workspace_root),
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "rendering": {
            "bhaskera_format": "chatml",
            "format_options": {"messages_field": "messages"},
            "add_generation_prompt": False,
            "tokenizer_add_special_tokens": True,
            "padding": "max_length during later cache generation",
            "truncation_side": tokenizer_7b.truncation_side,
            "labels": "input_ids with padding positions masked to -100",
            "pack_sequences": False,
        },
        "tokenizers": {
            "equivalent": tokenizers_equivalent,
            "olmo2_7b": identity_7b,
            "olmo2_1b": identity_1b,
        },
        "percentile_method": "nearest observed token length",
        "candidates": list(candidates),
        "views": views,
        "decision": {
            "status": "pending_length_and_throughput_review",
            "selected_seq_len": None,
        },
    }
    atomic_json(report, args.output_json.resolve())
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, args.output_markdown.resolve())
    print(f"JSON report: {args.output_json.resolve()}")
    print(f"Markdown report: {args.output_markdown.resolve()}")
    print("M0 TOKENIZER LENGTH AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
