#!/usr/bin/env python3
"""Summarize shared GOQA scores across the cross-country FL trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    evaluation_dir = args.evaluation_dir.resolve()
    entries = [{"name": "base", "round": 0, "delta_id": None}] + manifest["adapters"]
    rows = []
    for entry in entries:
        metrics_path = evaluation_dir / entry["name"] / "scores" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text())
        primary = metrics["primary"]
        groups = metrics["groups"]
        rows.append(
            {
                "model": entry["name"],
                "round": entry["round"],
                "delta_id": entry.get("delta_id") or "none",
                "australia_nz_macro_js_distance": primary["australia_nz_macro_js_distance"],
                "india_sample_frame_macro_js_distance": primary["india_sample_frame_macro_js_distance"],
                "two_region_macro_js_distance": primary["two_region_macro_js_distance"],
                "australia_js_distance": groups["Australia"]["mean_js_distance"],
                "new_zealand_js_distance": groups["New Zealand"]["mean_js_distance"],
                "india_current_js_distance": groups["India (Current national sample)"]["mean_js_distance"],
                "india_non_national_js_distance": groups["India (Non-national sample)"]["mean_js_distance"],
                "india_old_js_distance": groups["India (Old national sample)"]["mean_js_distance"],
            }
        )

    output_csv = evaluation_dir / "trajectory_summary.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rounds = [row["round"] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rounds, [row["australia_nz_macro_js_distance"] for row in rows], marker="o", label="Australia/New Zealand macro")
    ax.plot(rounds, [row["india_sample_frame_macro_js_distance"] for row in rows], marker="o", label="India sample-frame macro")
    ax.plot(rounds, [row["two_region_macro_js_distance"] for row in rows], marker="o", linewidth=2.5, label="Two-region macro")
    ax.axvline(20.5, color="crimson", linestyle="--", linewidth=1.5, label="India peer offline (during round 21)")
    for row in rows[1:]:
        ax.annotate(row["delta_id"], (row["round"], row["two_region_macro_js_distance"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set_title("Cross-country FL GOQA trajectory through peer disconnection")
    ax.set_xlabel("Local federated round (0 = base model)")
    ax.set_ylabel("Jensen–Shannon distance (lower is better)")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    plot_path = evaluation_dir / "trajectory_goqa_jsd.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    trained = rows[1:]
    best = min(trained, key=lambda row: row["two_region_macro_js_distance"])
    report = [
        "# Cross-country FL GOQA trajectory",
        "",
        "The evaluation uses the shared AU/NZ/India GOQA subset and its five-prompt Jensen–Shannon distance protocol. Lower values are better.",
        "",
        "The peer disconnected during local round 21. Round 20 is the last checkpoint saved before disconnection; round 21 is the first saved afterward and includes the final observed peer delta.",
        "",
        f"Best pre-cutoff checkpoint: **{best['model']}** (delta {best['delta_id']}), two-region macro JSD **{best['two_region_macro_js_distance']:.6f}**.",
        "",
        "| Model | Delta | AU/NZ macro | India macro | Two-region macro |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['delta_id']} | {row['australia_nz_macro_js_distance']:.6f} | {row['india_sample_frame_macro_js_distance']:.6f} | {row['two_region_macro_js_distance']:.6f} |"
        )
    (evaluation_dir / "trajectory_report.md").write_text("\n".join(report) + "\n")
    print(f"Summary: {output_csv}")
    print(f"Plot: {plot_path}")
    print(f"Report: {evaluation_dir / 'trajectory_report.md'}")
    print("JOINT GOQA TRAJECTORY SUMMARY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
