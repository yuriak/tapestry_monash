#!/usr/bin/env python3
"""Build the self-contained Australia/New Zealand/India GOQA subset."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from goqa_common import (
    SCHEMA_VERSION,
    TARGET_GROUPS,
    normalize,
    parse_mapping,
    sha256_file,
)

SOURCE_REPOSITORY = "Anthropic/llm_global_opinions"
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=here / "data/goqa_au_nz_india.jsonl"
    )
    parser.add_argument("--manifest", type=Path, default=here / "data/manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_csv.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    source_counts: Counter[str] = Counter()
    group_pairs: Counter[str] = Counter()
    excluded_zero: Counter[str] = Counter()
    source_rows = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = ["question", "selections", "options", "source"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected source schema: {reader.fieldnames}")
        for index, row in enumerate(reader):
            source_rows += 1
            options = ast.literal_eval(row["options"])
            if not isinstance(options, list) or not 2 <= len(options) <= 18:
                raise ValueError(f"Source row {index}: unsupported options")
            selections = parse_mapping(row["selections"])
            target = {}
            for group in TARGET_GROUPS:
                if group not in selections:
                    continue
                if len(selections[group]) != len(options):
                    raise ValueError(f"Source row {index}/{group}: length mismatch")
                distribution = normalize(selections[group])
                if distribution is None:
                    excluded_zero[group] += 1
                    continue
                target[group] = distribution
                group_pairs[group] += 1
            if not target:
                continue
            source_name = row["source"].strip()
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "question_id": f"goqa-{index:04d}",
                    "original_row_index": index,
                    "source": source_name,
                    "question": row["question"].strip(),
                    "options": options,
                    "human_distributions": target,
                }
            )
            source_counts[source_name] += 1

    if source_rows != 2556:
        raise ValueError(f"Expected 2,556 source rows, found {source_rows}")

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    au_nz_ids = {
        row["question_id"]
        for row in records
        if any(group in row["human_distributions"] for group in TARGET_GROUPS[:2])
    }
    india_ids = {
        row["question_id"]
        for row in records
        if any(group in row["human_distributions"] for group in TARGET_GROUPS[2:])
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "license": SOURCE_LICENSE,
            "csv_sha256": sha256_file(source),
            "rows": source_rows,
        },
        "dataset_file": output.name,
        "dataset_sha256": sha256_file(output),
        "filter": (
            "Keep questions with at least one non-zero Australia, New Zealand, "
            "or India human distribution; remove all non-target distributions."
        ),
        "question_count": len(records),
        "source_question_counts": dict(sorted(source_counts.items())),
        "target_pair_counts": {group: group_pairs[group] for group in TARGET_GROUPS},
        "excluded_zero_distributions": {
            group: excluded_zero[group] for group in TARGET_GROUPS
        },
        "australia_nz_unique_questions": len(au_nz_ids),
        "india_unique_questions": len(india_ids),
        "cross_region_question_overlap": len(au_nz_ids & india_ids),
        "target_groups": list(TARGET_GROUPS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} questions to {output}")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
