#!/usr/bin/env python3
"""Validate the standalone GOQA package, dataset, and optional predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from goqa_common import TARGET_GROUPS, load_dataset, sha256_file, validate_model_distribution

EXPECTED_QUESTION_COUNT = 1106
EXPECTED_PAIR_COUNTS = {
    "Australia": 626,
    "New Zealand": 273,
    "India (Current national sample)": 470,
    "India (Non-national sample)": 340,
    "India (Old national sample)": 122,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=here / "data/goqa_au_nz_india.jsonl"
    )
    parser.add_argument("--manifest", type=Path, default=here / "data/manifest.json")
    parser.add_argument("--predictions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("Dataset SHA-256 does not match the manifest")
    questions = load_dataset(dataset)
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(questions)}")
    if manifest.get("question_count") != len(questions):
        raise ValueError("Manifest question count does not match the dataset")
    pair_counts = Counter(
        group for row in questions for group in row["human_distributions"]
    )
    if dict(pair_counts) != EXPECTED_PAIR_COUNTS:
        raise ValueError(f"Unexpected target pair counts: {dict(pair_counts)}")
    if manifest.get("target_pair_counts") != EXPECTED_PAIR_COUNTS:
        raise ValueError("Manifest target pair counts are inconsistent")
    indices = [row["original_row_index"] for row in questions]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Original row indices are duplicated or out of order")
    for row in questions:
        expected_id = f"goqa-{row['original_row_index']:04d}"
        if row["question_id"] != expected_id:
            raise ValueError(f"Stable question ID mismatch: {row['question_id']}")

    if args.predictions:
        expected = {row["question_id"]: row for row in questions}
        observed = set()
        with args.predictions.resolve().open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                record = json.loads(line)
                question_id = record.get("question_id")
                if question_id not in expected or question_id in observed:
                    raise ValueError(
                        f"Unexpected or duplicate prediction at line {number}: {question_id}"
                    )
                observed.add(question_id)
                validate_model_distribution(
                    question_id,
                    record.get("model_distribution"),
                    len(expected[question_id]["options"]),
                )
        if observed != set(expected):
            raise ValueError(
                f"Prediction coverage is incomplete: {len(observed)}/{len(expected)}"
            )
        print(f"Prediction coverage passed: {len(observed)}/{len(expected)}")

    print(f"Dataset hash passed: {manifest['dataset_sha256']}")
    print(f"Questions passed: {len(questions)}")
    print(
        "Target pairs passed: "
        + ", ".join(f"{group}={pair_counts[group]}" for group in TARGET_GROUPS)
    )
    print("GOQA PACKAGE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
