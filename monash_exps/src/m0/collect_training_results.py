#!/usr/bin/env python3
"""Validate and summarize the seven formal M0/T5 training runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import yaml


RUN_SPECS = {
    "local_south_asia": {"steps": 1916, "interval": 80, "source": "m3"},
    "local_variant_1": {"steps": 1166, "interval": 50, "source": "m3"},
    "local_variant_2": {"steps": 5630, "interval": 225, "source": "spartan"},
    "local_variant_3": {"steps": 11238, "interval": 450, "source": "m3"},
    "central_variant_1": {"steps": 3082, "interval": 125, "source": "spartan"},
    "central_variant_2": {"steps": 7546, "interval": 300, "source": "spartan"},
    "central_variant_3": {"steps": 13154, "interval": 525, "source": "m3"},
}

ANSI_RE = re.compile(r"\x1b(?:[@-_][0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
STEP_RE = re.compile(
    r"\[epoch (?P<epoch>\d+)\]\[step (?P<step>\d+)\] "
    r"loss=(?P<loss>[-+0-9.eE]+) lr=(?P<lr>[-+0-9.eE]+) "
    r"grad_norm=(?P<grad>[-+0-9.eE]+) tok/s=(?P<tps>[-+0-9.eE]+) "
    r"MFU=(?P<mfu>[-+0-9.eE]+)%"
)
EPOCH_RE = re.compile(r"\[epoch (?P<epoch>\d+)\] avg_loss=(?P<loss>[-+0-9.eE]+)")


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_safetensors_header(path: Path) -> tuple[int, int, str]:
    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header_bytes = handle.read(header_len)
    header = json.loads(header_bytes)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    parameter_count = 0
    schema = []
    for key, value in sorted(tensors.items()):
        shape = value["shape"]
        parameter_count += math.prod(shape)
        schema.append({"name": key, "dtype": value["dtype"], "shape": shape})
    schema_sha = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return len(tensors), parameter_count, schema_sha


def expected_adapter_steps(max_steps: int, interval: int) -> list[int]:
    values = list(range(interval, max_steps + 1, interval))
    if not values or values[-1] != max_steps:
        values.append(max_steps)
    return values


def parse_training_log(run: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    epochs: dict[int, float] = {}
    window: deque[float] = deque(maxlen=100)
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw.replace("\r", ""))
            match = STEP_RE.search(line)
            if match:
                row = {
                    "run": run,
                    "epoch": int(match.group("epoch")),
                    "step": int(match.group("step")),
                    "loss": float(match.group("loss")),
                    "lr": float(match.group("lr")),
                    "grad_norm": float(match.group("grad")),
                    "tokens_per_second": float(match.group("tps")),
                    "mfu_percent": float(match.group("mfu")),
                }
                window.append(row["loss"])
                row["loss_rolling_100"] = statistics.fmean(window)
                records.append(row)
                continue
            match = EPOCH_RE.search(line)
            if match:
                epochs[int(match.group("epoch"))] = float(match.group("loss"))
    epoch_rows = [
        {"run": run, "epoch": epoch, "avg_loss": loss}
        for epoch, loss in sorted(epochs.items())
    ]
    return records, epoch_rows


def parse_timestamp(text: str) -> dt.datetime:
    return dt.datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f")


def parse_telemetry(run: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_gpu: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"time": [], "util": [], "memory": [], "memory_total": [], "power": []}
    )
    with path.open(newline="") as handle:
        for fields in csv.reader(handle, skipinitialspace=True):
            if len(fields) < 7:
                continue
            try:
                stamp = parse_timestamp(fields[0])
                util, memory, memory_total, power = map(float, fields[3:7])
            except (ValueError, TypeError):
                continue
            item = by_gpu[fields[1].strip()]
            item["time"].append(stamp)
            item["util"].append(util)
            item["memory"].append(memory)
            item["memory_total"].append(memory_total)
            item["power"].append(power)
    rows: list[dict[str, Any]] = []
    all_utils: list[float] = []
    all_memory: list[float] = []
    all_power: list[float] = []
    starts: list[dt.datetime] = []
    ends: list[dt.datetime] = []
    for gpu, item in sorted(by_gpu.items()):
        util = item["util"]
        active = [value for value in util if value >= 10]
        starts.append(min(item["time"]))
        ends.append(max(item["time"]))
        all_utils.extend(util)
        all_memory.extend(item["memory"])
        all_power.extend(item["power"])
        rows.append({
            "run": run,
            "gpu_index": gpu,
            "samples": len(util),
            "duration_seconds": (max(item["time"]) - min(item["time"])).total_seconds(),
            "util_mean_percent": statistics.fmean(util),
            "util_median_percent": statistics.median(util),
            "util_p95_percent": percentile(util, 0.95),
            "active_fraction_ge_10pct": len(active) / len(util),
            "active_util_mean_percent": statistics.fmean(active) if active else 0.0,
            "memory_mean_mib": statistics.fmean(item["memory"]),
            "memory_peak_mib": max(item["memory"]),
            "memory_total_mib": max(item["memory_total"]),
            "power_mean_w": statistics.fmean(item["power"]),
            "power_peak_w": max(item["power"]),
        })
    if not rows:
        raise ValueError(f"No telemetry samples parsed from {path}")
    aggregate = {
        "gpu_count": len(rows),
        "telemetry_samples": len(all_utils),
        "telemetry_duration_seconds": (max(ends) - min(starts)).total_seconds(),
        "gpu_util_mean_percent": statistics.fmean(all_utils),
        "gpu_util_p95_percent": percentile(all_utils, 0.95),
        "gpu_active_fraction_ge_10pct": sum(x >= 10 for x in all_utils) / len(all_utils),
        "gpu_memory_peak_mib": max(all_memory),
        "gpu_power_mean_w": statistics.fmean(all_power),
    }
    return rows, aggregate


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_sacct(run_dir: Path, job_id: str, native: bool) -> list[dict[str, str]]:
    fields = [
        "JobID", "JobName", "Account", "Partition", "QOS", "State", "ExitCode",
        "Elapsed", "Timelimit", "ReqTRES", "AllocTRES", "NodeList", "Reason",
    ]
    sacct_files = sorted((run_dir / "slurm").glob(f"sacct-{job_id}.tsv")) if (run_dir / "slurm").is_dir() else []
    text = ""
    if sacct_files:
        text = sacct_files[-1].read_text()
    elif native and shutil.which("sacct"):
        command = ["sacct", "-j", job_id, "-X", "--noheader", "-P", "-o", ",".join(fields)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            text = completed.stdout
    rows = []
    for line in text.splitlines():
        values = line.rstrip("|").split("|")
        if len(values) >= len(fields):
            row = dict(zip(fields, values[: len(fields)]))
            row["run"] = run_dir.name
            rows.append(row)
    return rows


def artifact_row(run: str, root: Path, path: Path, kind: str, digest: str = "") -> dict[str, Any]:
    return {
        "run": run,
        "kind": kind,
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def plot_results(out_dir: Path, steps: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    matplotlib_cache = out_dir / ".matplotlib-cache"
    matplotlib_cache.mkdir()
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        grouped[row["run"]].append(row)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir()
    for run, rows in grouped.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], alpha=0.22, linewidth=0.7, label="step loss")
        ax.plot([r["step"] for r in rows], [r["loss_rolling_100"] for r in rows], linewidth=1.8, label="rolling mean (100)")
        ax.set(title=run.replace("_", " "), xlabel="Optimizer step", ylabel="Training loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{run}_loss.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for run, rows in grouped.items():
        max_step = rows[-1]["step"]
        ax.plot([r["step"] / max_step for r in rows], [r["loss_rolling_100"] for r in rows], linewidth=1.3, label=run)
    ax.set(title="Formal M0 runs: loss over normalized training progress", xlabel="Fraction of configured optimizer steps", ylabel="Training loss (rolling mean, 100 steps)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "combined_normalized_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["run"] for row in summaries]
    values = [row["gpu_util_mean_percent"] for row in summaries]
    ax.bar(labels, values)
    ax.set(title="Mean GPU utilization from nvidia-smi telemetry", ylabel="GPU utilization (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "gpu_utilization_mean.png", dpi=180)
    plt.close(fig)
    shutil.rmtree(matplotlib_cache)


def build_report(out_dir: Path, summaries: list[dict[str, Any]], warnings: list[str], hash_dcp: bool) -> None:
    lines = [
        "# M0 Formal Training Result Collection",
        "",
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"All {len(summaries)} selected formal run{'s' if len(summaries) != 1 else ''} passed completion, configuration, adapter-history, final-adapter, final-DCP-marker, and training-log checks. "
        + ("Final DCP shard SHA-256 hashes were also recorded." if hash_dcp else "Final DCP shard hashing was explicitly skipped."),
        "",
        "| Run | Source | Steps | Final loss | Epoch losses | Duration (h) | Mean GPU util | Active fraction | Final adapter SHA-256 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        epoch_text = ", ".join(f"e{x['epoch']}={x['avg_loss']:.4f}" for x in row["epoch_losses"])
        lines.append(
            f"| {row['run']} | {row['source_cluster']} | {row['steps']} | {row['final_step_loss']:.4f} | "
            f"{epoch_text} | {row['telemetry_duration_seconds']/3600:.2f} | {row['gpu_util_mean_percent']:.1f}% | "
            f"{100*row['gpu_active_fraction_ge_10pct']:.1f}% | `{row['final_adapter_sha256']}` |"
        )
    lines += [
        "",
        "The duration is the span covered by per-GPU telemetry and is intended as a consistent wall-time proxy. GPU utilization includes synchronization, checkpointing, and startup/teardown samples; the active fraction is the proportion of samples at or above 10% utilization.",
        "",
        "The combined loss plot uses normalized optimizer-step progress because the datasets have different sizes. Raw and rolling loss values are retained in `step_metrics.csv`; this avoids treating differently sized datasets as if their step axes were directly comparable.",
        "",
        "## Files",
        "",
        "`run_summary.csv` is the compact run-level table. `step_metrics.csv` and `epoch_metrics.csv` contain parsed training curves. `gpu_telemetry_summary.csv` contains per-GPU telemetry statistics. `artifact_inventory.csv` records checked artifacts and hashes. `slurm_accounting.csv` preserves available scheduler accounting. `collection_manifest.json` contains full validation metadata.",
    ]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in warnings]
    (out_dir / "collection_report.md").write_text("\n".join(lines) + "\n")


def collect_run(
    run: str,
    run_dir: Path,
    spec: dict[str, Any],
    hash_dcp: bool,
    inventory: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    print(f"[{run}] validating {run_dir}", flush=True)
    required = ["COMPLETED", "resolved-config.yaml", "training.log"]
    for name in required:
        if not (run_dir / name).is_file():
            raise ValueError(f"{run}: missing {name}")
    completed = read_kv(run_dir / "COMPLETED")
    if completed.get("status") != "completed" or completed.get("run") != run:
        raise ValueError(f"{run}: invalid COMPLETED record")
    max_steps = spec["steps"]
    if int(completed.get("step", -1)) != max_steps:
        raise ValueError(f"{run}: completion step differs from expected {max_steps}")
    if spec["source"] == "spartan":
        import_marker = run_dir / "IMPORT_VERIFIED"
        if not import_marker.is_file():
            raise ValueError(f"{run}: missing IMPORT_VERIFIED; use m0_pull_spartan_results.sh")
        imported = read_kv(import_marker)
        if (
            imported.get("status") != "verified"
            or imported.get("run") != run
            or int(imported.get("step", -1)) != max_steps
            or imported.get("rsync_content_check") != "passed"
        ):
            raise ValueError(f"{run}: invalid Spartan import verification record")
        inventory.append(artifact_row(run, run_dir, import_marker, "import_verification", sha256(import_marker)))

    config_path = run_dir / "resolved-config.yaml"
    config_digest = sha256(config_path)
    if config_digest != completed.get("config_sha256"):
        raise ValueError(f"{run}: resolved configuration SHA-256 mismatch")
    config = yaml.safe_load(config_path.read_text())
    checks = {
        "data.seq_len": config["data"]["seq_len"] == 1024,
        "training.batch_size": config["training"]["batch_size"] == 2,
        "training.grad_accum": config["training"]["grad_accum"] == 4,
        "training.lr": math.isclose(float(config["training"]["lr"]), 1e-4),
        "training.num_epochs": config["training"]["num_epochs"] == 2,
        "training.max_steps": config["training"]["max_steps"] == max_steps,
        "training.seed": config["training"]["seed"] == 20260820,
        "training.strategy": config["training"]["distributed"]["strategy"] == "ddp",
        "adapter.interval": config["checkpoint"]["adapter_save_interval"] == spec["interval"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run}: configuration invariant failures: {', '.join(failed)}")

    adapter_root = run_dir / "adapter_history"
    actual_dirs = sorted(
        int(path.name.removeprefix("step_"))
        for path in adapter_root.glob("step_*")
        if path.is_dir() and path.name.removeprefix("step_").isdigit()
    )
    expected_dirs = expected_adapter_steps(max_steps, spec["interval"])
    if actual_dirs != expected_dirs:
        raise ValueError(f"{run}: adapter checkpoint steps differ from schedule")
    schema_values = set()
    for step in actual_dirs:
        step_dir = adapter_root / f"step_{step:07d}"
        marker = step_dir / ".complete"
        meta_path = step_dir / "meta.json"
        adapter_path = step_dir / "adapter_model.safetensors"
        if not marker.is_file() or not meta_path.is_file() or not adapter_path.is_file():
            raise ValueError(f"{run}: incomplete adapter checkpoint at step {step}")
        meta = json.loads(meta_path.read_text())
        digest = sha256(adapter_path)
        if int(meta["step"]) != step or digest != meta["sha256"]:
            raise ValueError(f"{run}: invalid adapter metadata/hash at step {step}")
        count, params, schema_sha = parse_safetensors_header(adapter_path)
        if count != 128 or params != 8_388_608:
            raise ValueError(f"{run}: unexpected adapter tensor schema at step {step}")
        schema_values.add(schema_sha)
        inventory.append(artifact_row(run, run_dir, adapter_path, "adapter", digest))
        inventory.append(artifact_row(run, run_dir, meta_path, "adapter_metadata", sha256(meta_path)))
    if len(schema_values) != 1:
        raise ValueError(f"{run}: adapter tensor schema changed during training")

    final_dir = adapter_root / f"step_{max_steps:07d}"
    final_adapter = final_dir / "adapter_model.safetensors"
    final_digest = sha256(final_adapter)
    if final_digest != completed.get("adapter_sha256"):
        raise ValueError(f"{run}: final adapter differs from COMPLETED record")

    dcp_dir = run_dir / "checkpoints" / f"step_{max_steps:07d}"
    dcp_meta_path = dcp_dir / "meta.json"
    if not (dcp_dir / ".complete").is_file() or not dcp_meta_path.is_file():
        raise ValueError(f"{run}: missing final DCP completion marker/metadata")
    dcp_meta = json.loads(dcp_meta_path.read_text())
    if int(dcp_meta["step"]) != max_steps:
        raise ValueError(f"{run}: final DCP metadata step mismatch")
    for relative in ["model/.metadata", "model/__0_0.distcp", "model/__1_0.distcp", "optim/.metadata", "optim/__0_0.distcp", "optim/__1_0.distcp"]:
        path = dcp_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{run}: missing/empty final DCP file {relative}")
        digest = sha256(path) if hash_dcp else ""
        inventory.append(artifact_row(run, run_dir, path, "dcp_final", digest))

    steps, epoch_rows = parse_training_log(run, run_dir / "training.log")
    sequence = [row["step"] for row in steps]
    if sequence != list(range(1, max_steps + 1)):
        raise ValueError(f"{run}: training log does not contain exactly steps 1..{max_steps}")
    if len(epoch_rows) != 2 or [row["epoch"] for row in epoch_rows] != [0, 1]:
        raise ValueError(f"{run}: expected exactly two epoch averages")
    for row in steps:
        if not all(math.isfinite(row[key]) for key in ("loss", "lr", "grad_norm", "tokens_per_second", "mfu_percent")):
            raise ValueError(f"{run}: non-finite training metric at step {row['step']}")

    gpu_files = list(run_dir.glob("gpu-*.csv"))
    if len(gpu_files) != 1:
        raise ValueError(f"{run}: expected one gpu-<job>.csv, found {len(gpu_files)}")
    match = re.fullmatch(r"gpu-(\d+)\.csv", gpu_files[0].name)
    if not match:
        raise ValueError(f"{run}: malformed telemetry filename")
    job_id = match.group(1)
    telemetry_rows, telemetry = parse_telemetry(run, gpu_files[0])
    sacct_rows = read_sacct(run_dir, job_id, spec["source"] == "m3")

    inventory.extend([
        artifact_row(run, run_dir, run_dir / "COMPLETED", "completion_record", sha256(run_dir / "COMPLETED")),
        artifact_row(run, run_dir, config_path, "resolved_config", config_digest),
        artifact_row(run, run_dir, run_dir / "training.log", "training_log", sha256(run_dir / "training.log")),
        artifact_row(run, run_dir, gpu_files[0], "gpu_telemetry", sha256(gpu_files[0])),
        artifact_row(run, run_dir, dcp_meta_path, "dcp_metadata", sha256(dcp_meta_path)),
    ])
    summary = {
        "run": run,
        "source_cluster": spec["source"],
        "source_path": str(run_dir),
        "job_id": job_id,
        "steps": max_steps,
        "epochs": 2,
        "batch_size_per_gpu": 2,
        "world_size": 2,
        "grad_accum": 4,
        "sequence_length": 1024,
        "learning_rate": 1e-4,
        "final_step_loss": steps[-1]["loss"],
        "final_rolling_100_loss": steps[-1]["loss_rolling_100"],
        "epoch_losses": epoch_rows,
        "final_adapter_sha256": final_digest,
        "adapter_schema_sha256": next(iter(schema_values)),
        "adapter_checkpoint_count": len(actual_dirs),
        "dcp_sha256_recorded": hash_dcp,
        **telemetry,
    }
    return summary, steps, epoch_rows, telemetry_rows, sacct_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--import-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runs", nargs="+", choices=list(RUN_SPECS))
    parser.add_argument("--skip-dcp-hash", action="store_true", help="Inventory but do not SHA-256 final DCP shards")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = args.runtime_root.resolve()
    import_root = (args.import_root or runtime / "imported_spartan").resolve()
    output_root = (args.output_root or runtime / "summary").resolve()
    runs = args.runs or list(RUN_SPECS)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    final_dir = output_root / f"m0-t5-{timestamp}"
    staging = output_root / f".m0-t5-{timestamp}.incoming"
    if final_dir.exists() or staging.exists():
        raise ValueError(f"Output already exists for timestamp {timestamp}")
    staging.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    all_epochs: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    sacct_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    try:
        for run in runs:
            spec = RUN_SPECS[run]
            run_dir = (runtime / run) if spec["source"] == "m3" else (import_root / run)
            summary, steps, epochs, telemetry, accounting = collect_run(
                run, run_dir, spec, not args.skip_dcp_hash, inventory
            )
            summaries.append(summary)
            all_steps.extend(steps)
            all_epochs.extend(epochs)
            telemetry_rows.extend(telemetry)
            sacct_rows.extend(accounting)
            if not accounting:
                warnings.append(f"No Slurm accounting record was available for {run} (job {summary['job_id']}).")

        flat_summaries = []
        for row in summaries:
            flat = {key: value for key, value in row.items() if key != "epoch_losses"}
            flat["epoch_0_avg_loss"] = row["epoch_losses"][0]["avg_loss"]
            flat["epoch_1_avg_loss"] = row["epoch_losses"][1]["avg_loss"]
            flat_summaries.append(flat)
        write_csv(staging / "run_summary.csv", flat_summaries)
        write_csv(staging / "step_metrics.csv", all_steps)
        write_csv(staging / "epoch_metrics.csv", all_epochs)
        write_csv(staging / "gpu_telemetry_summary.csv", telemetry_rows)
        write_csv(staging / "artifact_inventory.csv", inventory)
        sacct_fields = ["run", "JobID", "JobName", "Account", "Partition", "QOS", "State", "ExitCode", "Elapsed", "Timelimit", "ReqTRES", "AllocTRES", "NodeList", "Reason"]
        write_csv(staging / "slurm_accounting.csv", sacct_rows, sacct_fields)
        manifest = {
            "schema_version": 1,
            "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "runtime_root": str(runtime),
            "spartan_import_root": str(import_root),
            "dcp_sha256_recorded": not args.skip_dcp_hash,
            "validation_status": "passed",
            "runs": summaries,
            "warnings": warnings,
        }
        (staging / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        plot_results(staging, all_steps, summaries)
        build_report(staging, summaries, warnings, not args.skip_dcp_hash)
        os.rename(staging, final_dir)
    except Exception:
        print(f"Collection failed; diagnostic staging directory retained at {staging}", file=sys.stderr)
        raise

    print("M0 FORMAL RESULT COLLECTION PASSED")
    print(f"Summary directory: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
