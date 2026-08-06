#!/usr/bin/env python3
"""Capture node/GPU identity and check allocation-scoped process residue."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def metadata(args: argparse.Namespace) -> None:
    import torch

    payload = {
        "schema_version": 1,
        "peer_name": args.peer_name,
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "node_ip": args.node_ip,
        "seed_peer": None if args.seed_peer == "none" else args.seed_peer,
        "gpu_id_configured": args.gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_NUM_NODES",
                "SLURM_NNODES",
                "SLURM_NODEID",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_CPUS_PER_TASK",
            )
        },
    }
    write_json(args.output, payload)


def process_job_id(pid: int) -> str | None:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = b"SLURM_JOB_ID="
    for entry in raw.split(b"\0"):
        if entry.startswith(prefix):
            return entry[len(prefix) :].decode(errors="replace")
    return None


def clean(args: argparse.Namespace) -> None:
    markers = (
        "iiitd",
        "ml_engine.py",
        "launch_training.py",
        "raylet",
        "gcs_server",
        "dashboard.py",
        "log_monitor.py",
    )
    residue: list[dict[str, Any]] = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        if pid == os.getpid() or process_job_id(pid) != args.job_id:
            continue
        try:
            command = child.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in markers):
            residue.append({"pid": pid, "command": command.strip()})
    write_json(
        args.output,
        {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "slurm_job_id": args.job_id,
            "residue": residue,
        },
    )
    if residue:
        raise SystemExit(f"allocation-scoped process residue found: {residue}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    metadata_parser = subparsers.add_parser("metadata")
    metadata_parser.add_argument("--output", type=Path, required=True)
    metadata_parser.add_argument("--peer-name", choices=("peer-a", "peer-b"), required=True)
    metadata_parser.add_argument("--node-ip", required=True)
    metadata_parser.add_argument("--seed-peer", required=True)
    metadata_parser.add_argument("--gpu-id", type=int, required=True)
    metadata_parser.set_defaults(func=metadata)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--output", type=Path, required=True)
    clean_parser.add_argument("--job-id", required=True)
    clean_parser.set_defaults(func=clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
