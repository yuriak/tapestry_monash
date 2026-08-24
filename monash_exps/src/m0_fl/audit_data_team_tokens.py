#!/usr/bin/env python3
"""Verify that data-team token-string JSONL decodes to the selected M0 views."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode_view(
    path: Path,
    tokenizer: Any,
    countries: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    country_counts: Counter[str] = Counter()
    token_count = 0
    for row_number, row in enumerate(load_jsonl(path), start=1):
        if countries is not None and row.get("country") not in countries:
            continue
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError(f"{path}: row {row_number} has no messages list")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list) or not all(isinstance(token, str) for token in content):
                raise RuntimeError(
                    f"{path}: row {row_number} content is not a token-string list"
                )
            token_count += len(content)
            message["content"] = tokenizer.convert_tokens_to_string(content)
        country_counts[row["country"]] += 1
        records.append(row)
    return records, {
        "source_rows_selected": len(records),
        "token_strings": token_count,
        "countries": dict(sorted(country_counts.items())),
    }


def compare(
    name: str,
    data_team_path: Path,
    prepared_path: Path,
    tokenizer: Any,
    countries: set[str] | None,
) -> dict[str, Any]:
    decoded, details = decode_view(data_team_path, tokenizer, countries)
    prepared = load_jsonl(prepared_path)
    exact_rows = sum(left == right for left, right in zip(decoded, prepared, strict=False))
    exact = decoded == prepared
    if not exact:
        raise RuntimeError(
            f"{name}: decoded data-team records differ from the prepared view "
            f"({exact_rows}/{len(prepared)} rows exact)"
        )
    return {
        "view": name,
        "data_team_path": str(data_team_path),
        "data_team_sha256": sha256_file(data_team_path),
        "prepared_path": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "decoded_rows": len(decoded),
        "prepared_rows": len(prepared),
        "exact_rows_in_order": exact_rows,
        "decoded_content_exact": exact,
        **details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    results = [
        compare(
            "australia_nz_v1",
            args.incoming_root / "Oceania.jsonl",
            args.prepared_root / "australia_nz/train.jsonl",
            tokenizer,
            {"Australia", "New Zealand"},
        ),
        compare(
            "south_asia",
            args.incoming_root / "South_Asia.jsonl",
            args.prepared_root / "south_asia/train.jsonl",
            tokenizer,
            None,
        ),
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "result": "PASS",
        "decision": (
            "Use the existing Bhaskera numeric parquet caches. The data-team files "
            "contain OLMo token strings rather than Bhaskera input_ids/labels parquet, "
            "and decode exactly to the already prepared training views."
        ),
        "views": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Data-team token-string audit",
        "",
        "The audit passed. The files use OLMo token strings inside each message, not "
        "Bhaskera numeric parquet. After deterministic detokenization, both selected "
        "training views match the existing prepared JSONL exactly and in order.",
        "",
        "| View | Selected rows | Exact rows | Exact |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['view']} | {result['decoded_rows']} | "
            f"{result['exact_rows_in_order']} | {result['decoded_content_exact']} |"
        )
    lines.extend(["", f"Decision: {payload['decision']}", ""])
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_markdown}")
    print("Data-team token-string audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
