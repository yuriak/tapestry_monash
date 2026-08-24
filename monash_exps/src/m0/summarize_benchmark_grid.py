#!/usr/bin/env python3
"""Combine CulturalBench and GlobalOpinionQA checkpoint-grid results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def build_rows(cultural_dir: Path, goqa_dir: Path) -> list[dict[str, Any]]:
    cultural = {
        row["checkpoint"]: row
        for row in read_csv(cultural_dir / "summary.csv")
        if row["region_group"] == "overall"
    }
    goqa = {
        row["run"]: row
        for row in read_csv(goqa_dir / "summary.csv")
        if row["source"] == "all" and row["region_group"] == "all"
    }
    manifest = json.loads((cultural_dir / "manifest.json").read_text())
    adapters = manifest["adapters"]
    if set(cultural) != set(goqa) or set(cultural) != set(adapters):
        raise ValueError(
            "CulturalBench, GlobalOpinionQA, and manifest checkpoints differ"
        )

    rows = []
    for checkpoint, metadata in adapters.items():
        culture = cultural[checkpoint]
        opinion = goqa[checkpoint]
        rows.append(
            {
                "checkpoint": checkpoint,
                "training_run": metadata["training_run"],
                "step": metadata["step"],
                "final_step": metadata["final_step"],
                "target_fraction": metadata["target_fraction"],
                "actual_fraction": metadata["actual_fraction"],
                "epoch_equivalent": metadata["epoch_equivalent"],
                "culturalbench_easy_accuracy": as_float(culture["easy_accuracy"]),
                "culturalbench_hard_binary_accuracy": as_float(
                    culture["hard_binary_accuracy"]
                ),
                "culturalbench_hard_question_exact_match": as_float(
                    culture["hard_question_exact_match"]
                ),
                "culturalbench_hard_reconstructed_mc_accuracy": as_float(
                    culture["hard_reconstructed_mc_accuracy"]
                ),
                "culturalbench_combined_decision_accuracy": as_float(
                    culture["combined_decision_accuracy"]
                ),
                "culturalbench_combined_question_accuracy": as_float(
                    culture["combined_question_accuracy"]
                ),
                "culturalbench_hard_true_prediction_rate": as_float(
                    culture["hard_true_prediction_rate"]
                ),
                "goqa_country_macro_js_distance": as_float(
                    opinion["country_macro_mean_js_distance"]
                ),
                "goqa_country_question_pairs": int(opinion["country_question_pairs"]),
                "goqa_countries": int(opinion["countries"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = next(row for row in rows if row["checkpoint"] == "base")
    runs = []
    for row in rows:
        run = row["training_run"]
        if run != "base" and run not in runs:
            runs.append(run)
    metrics = [
        ("goqa_country_macro_js_distance", "GOQA country-macro JSD", False),
        ("culturalbench_easy_accuracy", "CulturalBench Easy accuracy", True),
        ("culturalbench_hard_binary_accuracy", "Hard binary accuracy", True),
        (
            "culturalbench_hard_question_exact_match",
            "Hard question exact match",
            True,
        ),
        (
            "culturalbench_combined_question_accuracy",
            "Combined question accuracy",
            True,
        ),
        (
            "culturalbench_hard_true_prediction_rate",
            "Hard TRUE prediction rate",
            True,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, (metric, title, percentage) in zip(axes.flat, metrics):
        for run in runs:
            values = [base] + sorted(
                [row for row in rows if row["training_run"] == run],
                key=lambda row: row["target_fraction"],
            )
            x = [100 * row["target_fraction"] for row in values]
            scale = 100 if percentage else 1
            y = [scale * row[metric] for row in values]
            axis.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, label=run)
        axis.set_title(title)
        axis.set_xlabel("Training progress (%)")
        axis.set_ylabel("Percent" if percentage else "JSD (lower is better)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    base = next(row for row in rows if row["checkpoint"] == "base")
    runs = []
    for row in rows:
        run = row["training_run"]
        if run != "base" and run not in runs:
            runs.append(run)
    lines = [
        "# M0 Benchmark Training Trajectories",
        "",
        "![Benchmark trajectories](trajectory_comparison.png)",
        "",
        (
            "GlobalOpinionQA uses all available countries and five deterministic option-order "
            "prompts per question. CulturalBench Easy and Hard are evaluated together in one "
            "request pool but retain separate and combined metrics."
        ),
        "",
        "## Base",
        "",
        "| GOQA JSD | Easy | Hard binary | Hard exact | Combined questions | Hard TRUE rate |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {base['goqa_country_macro_js_distance']:.4f} | "
            f"{100 * base['culturalbench_easy_accuracy']:.2f}% | "
            f"{100 * base['culturalbench_hard_binary_accuracy']:.2f}% | "
            f"{100 * base['culturalbench_hard_question_exact_match']:.2f}% | "
            f"{100 * base['culturalbench_combined_question_accuracy']:.2f}% | "
            f"{100 * base['culturalbench_hard_true_prediction_rate']:.2f}% |"
        ),
    ]
    for run in runs:
        lines.extend(
            [
                "",
                f"## {run}",
                "",
                "| Target progress | Actual progress | Step | GOQA JSD | Easy | Hard binary | Hard exact | Combined questions | TRUE rate |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        values = sorted(
            [row for row in rows if row["training_run"] == run],
            key=lambda row: row["target_fraction"],
        )
        for row in values:
            lines.append(
                f"| {100 * row['target_fraction']:.0f}% | "
                f"{100 * row['actual_fraction']:.1f}% | {row['step']:,} | "
                f"{row['goqa_country_macro_js_distance']:.4f} | "
                f"{100 * row['culturalbench_easy_accuracy']:.2f}% | "
                f"{100 * row['culturalbench_hard_binary_accuracy']:.2f}% | "
                f"{100 * row['culturalbench_hard_question_exact_match']:.2f}% | "
                f"{100 * row['culturalbench_combined_question_accuracy']:.2f}% | "
                f"{100 * row['culturalbench_hard_true_prediction_rate']:.2f}% |"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cultural-dir", type=Path, required=True)
    parser.add_argument("--goqa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cultural_dir = args.cultural_dir.resolve()
    goqa_dir = args.goqa_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(cultural_dir, goqa_dir)
    write_csv(output_dir / "combined_summary.csv", rows)
    plot(output_dir / "trajectory_comparison.png", rows)
    write_report(output_dir / "report.md", rows)
    print("M0 BENCHMARK GRID SUMMARY PASSED")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
