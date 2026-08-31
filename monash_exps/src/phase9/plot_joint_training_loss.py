#!/usr/bin/env python3
"""Plot completed joint federated rounds and their observed peer-delta merges."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

WORKSPACE = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--loss-csv",
    type=Path,
    default=WORKSPACE / "Slakshna/logs/epoch_loss_tracking_muon.csv",
)
parser.add_argument(
    "--runtime-log",
    type=Path,
    default=WORKSPACE / "Slakshna/logs/runtime_comm.log",
)
parser.add_argument(
    "--joint-log",
    type=Path,
    default=WORKSPACE / "Slakshna/monash_joint_20260826_retry.log",
)
parser.add_argument(
    "--output",
    type=Path,
    default=WORKSPACE
    / "monash_exps/.runtime/analysis/cross_country_fl/joint_loss_with_delta_merges.png",
)
parser.add_argument(
    "--node-id",
    default="slakshna1vxwxlxaznw253ucsx4djr422y9wp2ej5kuavu0",
)
parser.add_argument("--expected-steps", type=int, default=50)
parser.add_argument("--plot-step", type=int, default=5)
args = parser.parse_args()

CSV_PATH = args.loss_csv.resolve()
RUNTIME_PATH = args.runtime_log.resolve()
JOINT_LOG_PATH = args.joint_log.resolve()
OUTPUT_PATH = args.output.resolve()
NODE_ID = args.node_id
EXPECTED_STEPS = args.expected_steps
PLOT_STEP = args.plot_step


columns = ["timestamp", "node_id", "epoch", "step", "loss", "perplexity"]
df = pd.read_csv(CSV_PATH, names=columns, header=None)
df = df[df["node_id"].eq(NODE_ID)].copy()

for column in ("epoch", "step", "loss"):
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=["epoch", "step", "loss"])
df[["epoch", "step"]] = df[["epoch", "step"]].astype(int)
df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

# A CSV epoch is counted as a real round only after all 50 local steps exist.
complete_epochs = (
    df.groupby("epoch")["step"].max().loc[lambda values: values >= EXPECTED_STEPS].index
)
plot_df = df[df["epoch"].isin(complete_epochs) & df["step"].eq(PLOT_STEP)].copy()
plot_df = plot_df.sort_values("epoch").drop_duplicates("epoch", keep="last")
plot_df["round"] = range(1, len(plot_df) + 1)

# Reconstruct the observed peer-delta sequence from the joint-run network log.
# A large inbound payload is a model delta; the small ~500-byte payloads are
# reviews/metadata. Each extraction stages the latest received model delta.
received_deltas = []
extracted_deltas = []
if JOINT_LOG_PATH.exists():
    timestamp_re = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
    size_re = re.compile(r"Network Payload Size: (\d+) bytes")
    for line in JOINT_LOG_PATH.read_text(errors="replace").splitlines():
        timestamp_match = timestamp_re.search(line)
        if not timestamp_match:
            continue
        timestamp = (
            pd.to_datetime(timestamp_match.group(1), utc=True)
            .tz_convert("Australia/Melbourne")
            .tz_localize(None)
        )
        if "Gossiped model update received" in line:
            size_match = size_re.search(line)
            if size_match and int(size_match.group(1)) > 1_000_000:
                received_deltas.append(
                    {"id": f"D{len(received_deltas) + 1}", "received": timestamp}
                )
        elif "Network Delta Extracted" in line:
            candidates = [item for item in received_deltas if item["received"] <= timestamp]
            if candidates:
                extracted_deltas.append({"extracted": timestamp, **candidates[-1]})

# Map each successful peer_delta_loaded event to the local CSV epoch that had
# just completed and to the most recently staged network delta.
merge_times = []
if RUNTIME_PATH.exists():
    with RUNTIME_PATH.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and row[1] == NODE_ID and row[3] == "peer_delta_loaded":
                timestamp = pd.to_datetime(row[0], errors="coerce")
                if not pd.isna(timestamp):
                    merge_times.append(timestamp)

epoch_end = df.groupby("epoch")["timestamp_dt"].max().dropna().sort_values()
merged_epochs = set()
epoch_delta = {}
for merge_time in merge_times:
    candidates = epoch_end[epoch_end <= merge_time]
    if not candidates.empty:
        epoch = int(candidates.index[-1])
        merged_epochs.add(epoch)
        staged = [item for item in extracted_deltas if item["extracted"] <= merge_time]
        if staged:
            epoch_delta[epoch] = staged[-1]
plot_df["merged_peer_delta"] = plot_df["epoch"].isin(merged_epochs)
plot_df["delta_id"] = plot_df["epoch"].map(
    lambda epoch: epoch_delta.get(epoch, {}).get("id", "")
)
plot_df["delta_received"] = plot_df["epoch"].map(
    lambda epoch: epoch_delta.get(epoch, {}).get("received", pd.NaT)
)

if plot_df.empty:
    raise SystemExit("No completed 50-step Monash rounds are available to plot.")

fig, (ax, delta_ax) = plt.subplots(
    2,
    1,
    figsize=(16, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [5, 0.9], "hspace": 0.08},
)
ax.plot(
    plot_df["round"],
    plot_df["loss"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=8,
    label=NODE_ID,
)
merged = plot_df[plot_df["merged_peer_delta"]]
if not merged.empty:
    seen_delta_ids = set()
    for index, (_, row) in enumerate(merged.iterrows()):
        delta_id = row["delta_id"] or "peer delta"
        reused = delta_id in seen_delta_ids
        seen_delta_ids.add(delta_id)
        color = "darkorange" if reused else "crimson"
        ax.scatter(
            [row["round"]],
            [row["loss"]],
            marker="*",
            s=260,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=("Stale delta reused" if reused else "New peer delta merged"),
        )

ax.set_title("Global Model Loss at Step 1 of Each Epoch", fontsize=16)
ax.set_ylabel("Loss (Step 1)", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="upper right")

# A compact timeline makes the exact peer delta used in every local round
# readable even after the run grows to dozens of rounds.
delta_ids = [item for item in plot_df["delta_id"].unique() if item]
palette = plt.get_cmap("tab20")
delta_colors = {delta_id: palette(i % 20) for i, delta_id in enumerate(delta_ids)}
for _, row in plot_df.iterrows():
    round_number = int(row["round"])
    delta_id = row["delta_id"]
    if delta_id:
        delta_ax.add_patch(
            Rectangle(
                (round_number - 0.48, 0.05),
                0.96,
                0.9,
                facecolor=delta_colors[delta_id],
                edgecolor="white",
                linewidth=1,
            )
        )
        delta_ax.text(
            round_number,
            0.5,
            delta_id,
            ha="center",
            va="center",
            fontsize=8,
            rotation=90 if len(plot_df) > 20 else 0,
            color="black",
        )
    else:
        delta_ax.text(round_number, 0.5, "--", ha="center", va="center", color="gray")

delta_ax.set_ylim(0, 1)
delta_ax.set_xlim(plot_df["round"].min() - 0.5, plot_df["round"].max() + 0.5)
delta_ax.set_yticks([0.5], ["Peer delta\nused"])
delta_ax.set_xlabel("Local federated round", fontsize=14)
delta_ax.set_xticks(plot_df["round"])
delta_ax.tick_params(axis="x", labelsize=8)
delta_ax.grid(False)

fig.text(
    0.5,
    0.005,
    "D1, D2, ... denote peer model deltas in observed arrival order; '--' means no peer delta was successfully loaded.",
    ha="center",
    fontsize=9,
    color="dimgray",
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(
    plot_df[
        ["epoch", "round", "loss", "merged_peer_delta", "delta_id", "delta_received"]
    ].to_string(index=False)
)
print(f"Plot saved to {OUTPUT_PATH}")
