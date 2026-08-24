#!/usr/bin/env python3
"""Score standalone GOQA predictions for Australia/NZ and India."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from goqa_common import (
    AUSTRALIA,
    AU_NZ_GROUPS,
    INDIA_CURRENT,
    INDIA_GROUPS,
    INDIA_NON_NATIONAL,
    INDIA_OLD,
    NEW_ZEALAND,
    TARGET_GROUPS,
    js_distance,
    load_dataset,
    sha256_file,
    validate_model_distribution,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=here / "data/goqa_au_nz_india.jsonl"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path, questions: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    predictions: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except Exception as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            question_id = row.get("question_id")
            if question_id not in questions:
                raise ValueError(f"Unexpected question ID in predictions: {question_id}")
            if question_id in predictions:
                raise ValueError(f"Duplicate prediction: {question_id}")
            predictions[question_id] = validate_model_distribution(
                question_id,
                row.get("model_distribution"),
                len(questions[question_id]["options"]),
            )
    missing = set(questions) - set(predictions)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Predictions are missing {len(missing)} questions: {preview}")
    return predictions


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric group")
    return statistics.fmean(values)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    questions_list = load_dataset(dataset)
    questions = {row["question_id"]: row for row in questions_list}
    predictions = read_predictions(predictions_path, questions)

    distances: dict[str, list[float]] = defaultdict(list)
    pair_rows = []
    for question in questions_list:
        question_id = question["question_id"]
        model = predictions[question_id]
        for group, human in question["human_distributions"].items():
            distance = js_distance(model, human)
            distances[group].append(distance)
            pair_rows.append(
                {
                    "question_id": question_id,
                    "source": question["source"],
                    "group": group,
                    "js_distance": distance,
                }
            )

    missing_groups = [group for group in TARGET_GROUPS if not distances[group]]
    if missing_groups:
        raise ValueError(f"No scored pairs for: {missing_groups}")
    group_means = {group: mean(distances[group]) for group in TARGET_GROUPS}
    au_nz_macro = mean([group_means[group] for group in AU_NZ_GROUPS])
    india_macro = mean([group_means[group] for group in INDIA_GROUPS])
    au_nz_pairs = [value for group in AU_NZ_GROUPS for value in distances[group]]
    india_pairs = [value for group in INDIA_GROUPS for value in distances[group]]
    metrics = {
        "schema_version": 1,
        "metric": "base-2 Jensen-Shannon distance (sqrt of JS divergence)",
        "lower_is_better": True,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_sha256": sha256_file(dataset),
        "predictions_sha256": sha256_file(predictions_path),
        "questions": len(questions),
        "country_question_pairs": len(pair_rows),
        "primary": {
            "australia_nz_macro_js_distance": au_nz_macro,
            "india_sample_frame_macro_js_distance": india_macro,
            "two_region_macro_js_distance": mean([au_nz_macro, india_macro]),
        },
        "secondary": {
            "australia_nz_pair_weighted_js_distance": mean(au_nz_pairs),
            "india_pair_weighted_js_distance": mean(india_pairs),
        },
        "groups": {
            group: {
                "country_question_pairs": len(distances[group]),
                "mean_js_distance": group_means[group],
            }
            for group in TARGET_GROUPS
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "group_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["group", "country_question_pairs", "mean_js_distance"]
        )
        writer.writeheader()
        for group in TARGET_GROUPS:
            writer.writerow({"group": group, **metrics["groups"][group]})
    with (output_dir / "regional_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["region", "aggregation", "country_question_pairs", "js_distance"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "region": "Australia/New Zealand",
                    "aggregation": "macro over Australia and New Zealand",
                    "country_question_pairs": len(au_nz_pairs),
                    "js_distance": au_nz_macro,
                },
                {
                    "region": "India",
                    "aggregation": "macro over three India sample frames",
                    "country_question_pairs": len(india_pairs),
                    "js_distance": india_macro,
                },
                {
                    "region": "Australia/New Zealand + India",
                    "aggregation": "macro over the two regional metrics",
                    "country_question_pairs": len(pair_rows),
                    "js_distance": mean([au_nz_macro, india_macro]),
                },
            ]
        )
    with (output_dir / "pair_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["question_id", "source", "group", "js_distance"]
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    labels = {
        AUSTRALIA: "Australia",
        NEW_ZEALAND: "New Zealand",
        INDIA_CURRENT: "India — current national",
        INDIA_NON_NATIONAL: "India — non-national",
        INDIA_OLD: "India — old national",
    }
    lines = [
        "# GOQA Australia/New Zealand and India Results",
        "",
        "Lower Jensen–Shannon distance is better.",
        "",
        "## Primary regional metrics",
        "",
        "| Region | Aggregation | Pairs | JS distance |",
        "|---|---|---:|---:|",
        f"| Australia/New Zealand | Equal macro over two countries | {len(au_nz_pairs)} | {au_nz_macro:.6f} |",
        f"| India | Equal macro over three survey sample frames | {len(india_pairs)} | {india_macro:.6f} |",
        f"| Two-region summary | Equal macro over the two rows above | {len(pair_rows)} | {mean([au_nz_macro, india_macro]):.6f} |",
        "",
        "## Disaggregated metrics",
        "",
        "| Group | Pairs | Mean JS distance |",
        "|---|---:|---:|",
    ]
    for group in TARGET_GROUPS:
        lines.append(
            f"| {labels[group]} | {len(distances[group])} | {group_means[group]:.6f} |"
        )
    lines.extend(
        [
            "",
            "The model is not prompted with a country identity. Each score compares the same "
            "question-level model distribution with the human distribution available for the "
            "named country or survey sample frame.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Scored {len(questions)} questions and {len(pair_rows)} target pairs")
    print(f"Australia/NZ macro JS distance: {au_nz_macro:.6f}")
    print(f"India sample-frame macro JS distance: {india_macro:.6f}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
