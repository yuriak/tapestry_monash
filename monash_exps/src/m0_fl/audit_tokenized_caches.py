#!/usr/bin/env python3
"""Compare data-team token caches with the locally generated Bhaskera caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("input_ids", "attention_mask", "labels")


def portable(path: Path, workspace: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_cache_groups(root: Path) -> list[tuple[Path, list[Path]]]:
    grouped: dict[Path, list[Path]] = {}
    for parquet in sorted(root.rglob("*.parquet")):
        grouped.setdefault(parquet.parent, []).append(parquet)
    return sorted(grouped.items(), key=lambda item: str(item[0]))


def audit_group(group: Path, parquets: list[Path], workspace: Path) -> dict[str, Any]:
    logical_digest = hashlib.sha256()
    lengths: Counter[int] = Counter()
    rows = 0
    active_tokens = 0
    supervised_tokens = 0
    token_min: int | None = None
    token_max: int | None = None
    schemas: set[str] = set()
    compatible = True
    problems: list[str] = []
    file_records = []

    for parquet in parquets:
        parquet_file = pq.ParquetFile(parquet)
        schema = parquet_file.schema_arrow
        schemas.add(str(schema))
        missing = sorted(set(REQUIRED_COLUMNS) - set(schema.names))
        if missing:
            compatible = False
            problems.append(f"{parquet.name}: missing columns {missing}")
            continue

        file_records.append(
            {
                "path": portable(parquet, workspace),
                "bytes": parquet.stat().st_size,
                "rows": parquet_file.metadata.num_rows,
                "sha256": sha256_file(parquet),
            }
        )
        for batch in parquet_file.iter_batches(batch_size=256, columns=list(REQUIRED_COLUMNS)):
            columns = [batch.column(index).to_pylist() for index in range(3)]
            for input_ids_raw, attention_raw, labels_raw in zip(*columns, strict=True):
                input_ids = np.asarray(input_ids_raw, dtype="<i8")
                attention = np.asarray(attention_raw, dtype="<i8")
                labels = np.asarray(labels_raw, dtype="<i8")
                row_length = len(input_ids)
                lengths[row_length] += 1
                rows += 1
                if len(attention) != row_length or len(labels) != row_length:
                    compatible = False
                    problems.append(f"row {rows}: tensor lengths differ")
                    continue
                unique_mask = set(np.unique(attention).tolist())
                if not unique_mask.issubset({0, 1}):
                    compatible = False
                    problems.append(f"row {rows}: attention mask has values {sorted(unique_mask)}")
                active_tokens += int(attention.sum())
                supervised_tokens += int(np.count_nonzero(labels != -100))
                if input_ids.size:
                    current_min = int(input_ids.min())
                    current_max = int(input_ids.max())
                    token_min = current_min if token_min is None else min(token_min, current_min)
                    token_max = current_max if token_max is None else max(token_max, current_max)
                logical_digest.update(struct.pack("<Q", row_length))
                logical_digest.update(input_ids.tobytes())
                logical_digest.update(attention.tobytes())
                logical_digest.update(labels.tobytes())

    if len(schemas) != 1:
        compatible = False
        problems.append(f"cache has {len(schemas)} distinct parquet schemas")
    if not rows:
        compatible = False
        problems.append("cache has no readable rows")

    metadata = []
    for path in sorted(group.glob("*.json")):
        metadata.append(
            {
                "path": portable(path, workspace),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "path": portable(group, workspace),
        "compatible_bhaskera_parquet": compatible,
        "problems": problems[:50],
        "rows": rows,
        "sequence_lengths": {str(key): value for key, value in sorted(lengths.items())},
        "active_tokens": active_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_fraction_of_active": (
            supervised_tokens / active_tokens if active_tokens else None
        ),
        "token_id_min": token_min,
        "token_id_max": token_max,
        "logical_content_sha256": logical_digest.hexdigest(),
        "parquet_files": file_records,
        "metadata_files": metadata,
        "schema": next(iter(schemas), ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument(
        "--reference-view",
        action="append",
        default=[],
        help="Restrict references to one or more immediate subdirectories.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]
    incoming_groups = discover_cache_groups(args.incoming_root)
    reference_groups = discover_cache_groups(args.reference_root)
    if args.reference_view:
        requested = set(args.reference_view)
        reference_groups = [
            item
            for item in reference_groups
            if item[0].relative_to(args.reference_root).parts[0] in requested
        ]
    if not incoming_groups:
        raise RuntimeError(f"no parquet files found under {args.incoming_root}")
    if not reference_groups:
        raise RuntimeError(f"no reference parquet files found under {args.reference_root}")

    print(f"Auditing {len(incoming_groups)} incoming cache group(s)...", flush=True)
    incoming = [audit_group(path, files, workspace) for path, files in incoming_groups]
    print(f"Auditing {len(reference_groups)} local reference cache group(s)...", flush=True)
    reference = [audit_group(path, files, workspace) for path, files in reference_groups]

    comparisons = []
    for candidate in incoming:
        exact = [
            item["path"]
            for item in reference
            if item["logical_content_sha256"] == candidate["logical_content_sha256"]
            and item["rows"] == candidate["rows"]
        ]
        same_rows = [item["path"] for item in reference if item["rows"] == candidate["rows"]]
        comparisons.append(
            {
                "incoming": candidate["path"],
                "exact_logical_matches": exact,
                "same_row_count_references": same_rows,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "incoming_root": portable(args.incoming_root, workspace),
        "reference_root": portable(args.reference_root, workspace),
        "reference_views": args.reference_view,
        "incoming": incoming,
        "reference": reference,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Data-team token cache compatibility audit",
        "",
        "| Source | Rows | Lengths | Bhaskera-compatible | Exact local match |",
        "|---|---:|---|---|---|",
    ]
    comparison_by_path = {item["incoming"]: item for item in comparisons}
    for item in incoming:
        comparison = comparison_by_path[item["path"]]
        exact = "<br>".join(comparison["exact_logical_matches"]) or "none"
        lengths = ", ".join(f"{key}: {value}" for key, value in item["sequence_lengths"].items())
        lines.append(
            f"| `{item['path']}` | {item['rows']} | {lengths} | "
            f"{item['compatible_bhaskera_parquet']} | {exact} |"
        )
    lines.extend(["", "## Local references", ""])
    for item in reference:
        lines.append(
            f"- `{item['path']}`: {item['rows']} rows; "
            f"logical SHA-256 `{item['logical_content_sha256']}`"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
