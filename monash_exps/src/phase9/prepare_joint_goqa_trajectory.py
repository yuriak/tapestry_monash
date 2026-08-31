#!/usr/bin/env python3
"""Prepare cross-country FL LoRA rounds for shared GOQA trajectory evaluation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from safetensors.torch import save_file

EXPECTED_TENSORS = 128
EXPECTED_PARAMETERS = 8_388_608


def parse_rounds(value: str) -> list[int]:
    rounds: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            if first < 1 or last < first:
                raise argparse.ArgumentTypeError(f"Invalid round range: {part}")
            rounds.update(range(first, last + 1))
        else:
            number = int(part)
            if number < 1:
                raise argparse.ArgumentTypeError("Rounds must be positive")
            rounds.add(number)
    if not rounds:
        raise argparse.ArgumentTypeError("At least one round is required")
    return sorted(rounds)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != EXPECTED_TENSORS:
        raise RuntimeError(f"Unexpected LoRA tensor count in {path}: {len(state)}")
    count = sum(value.numel() for value in state.values())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Unexpected LoRA parameter count in {path}: {count}")
    for key, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise RuntimeError(f"Invalid LoRA tensor {key} in {path}")
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"Non-finite LoRA tensor {key} in {path}")
    normalized = {}
    for key, value in state.items():
        # Bhaskera retains PEFT's in-memory adapter namespace (``default``),
        # whereas saved PEFT adapters and vLLM use lora_A.weight/lora_B.weight.
        normalized_key = key.replace(".lora_A.default.weight", ".lora_A.weight")
        normalized_key = normalized_key.replace(
            ".lora_B.default.weight", ".lora_B.weight"
        )
        if normalized_key in normalized:
            raise RuntimeError(f"Duplicate normalized LoRA key: {normalized_key}")
        normalized[normalized_key] = value.detach().cpu().contiguous()
    if any(".default." in key for key in normalized):
        raise RuntimeError(f"Unsupported PEFT training namespace remains in {path}")
    return normalized


def parse_delta_mapping(
    joint_log: Path, runtime_log: Path, node_id: str, checkpoint_dir: Path
) -> tuple[dict[int, dict[str, Any]], str | None]:
    import pandas as pd

    timestamp_re = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
    size_re = re.compile(r"Network Payload Size: (\d+) bytes")
    received: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    offline_at = None
    for line in joint_log.read_text(errors="replace").splitlines():
        match = timestamp_re.search(line)
        if not match:
            continue
        timestamp = (
            pd.to_datetime(match.group(1), utc=True)
            .tz_convert("Australia/Melbourne")
            .tz_localize(None)
        )
        if "Gossiped model update received" in line:
            size_match = size_re.search(line)
            if size_match and int(size_match.group(1)) > 1_000_000:
                received.append({"delta_id": f"D{len(received) + 1}", "received": timestamp})
        elif "Network Delta Extracted" in line:
            candidates = [item for item in received if item["received"] <= timestamp]
            if candidates:
                extracted.append({"extracted": timestamp, **candidates[-1]})
        elif "Peer left gossip mesh" in line:
            offline_at = timestamp.isoformat()

    load_times = []
    with runtime_log.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and row[1] == node_id and row[3] == "peer_delta_loaded":
                load_times.append(pd.to_datetime(row[0]))

    mapping: dict[int, dict[str, Any]] = {}
    for checkpoint in checkpoint_dir.glob("sync_round_*.pth"):
        round_number = int(checkpoint.stem.rsplit("_", 1)[1])
        checkpoint_time = (
            pd.to_datetime(checkpoint.stat().st_mtime, unit="s", utc=True)
            .tz_convert("Australia/Melbourne")
            .tz_localize(None)
        )
        nearby = [time for time in load_times if abs((time - checkpoint_time).total_seconds()) <= 5]
        if not nearby:
            continue
        load_time = max(nearby)
        candidates = [item for item in extracted if item["extracted"] <= load_time]
        if candidates:
            item = candidates[-1]
            mapping[round_number] = {
                "delta_id": item["delta_id"],
                "delta_received_at": item["received"].isoformat(),
                "delta_extracted_at": item["extracted"].isoformat(),
                "delta_loaded_at": load_time.isoformat(),
            }
    return mapping, offline_at


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--joint-log", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-round", type=int, default=21)
    parser.add_argument(
        "--rounds", type=parse_rounds,
        help="Selected rounds, for example 1-7,16. Defaults to 1 through --last-round.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    base_model = args.base_model.resolve()
    output_dir = args.output_dir.resolve()
    node_id = checkpoint_dir.name.removeprefix("sync_ckpt_")
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Base model is missing: {base_model}")
    if args.last_round < 1:
        raise ValueError("--last-round must be positive")
    selected_rounds = args.rounds or list(range(1, args.last_round + 1))
    if selected_rounds[-1] > args.last_round:
        raise ValueError("A selected round exceeds --last-round")

    delta_mapping, offline_at = parse_delta_mapping(
        args.joint_log.resolve(), args.runtime_log.resolve(), node_id, checkpoint_dir
    )
    adapters_root = output_dir / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)
    entries = []
    config = {
        "base_model_name_or_path": str(base_model),
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 64,
        "lora_dropout": 0.03,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
    }
    for round_number in selected_rounds:
        source = checkpoint_dir / f"sync_round_{round_number}.pth"
        if not source.is_file():
            raise FileNotFoundError(source)
        adapter_dir = adapters_root / f"round_{round_number:02d}"
        adapter_dir.mkdir(exist_ok=True)
        weights = adapter_dir / "adapter_model.safetensors"
        source_hash = sha256(source)
        # Always materialize from the authoritative .pth. This also upgrades
        # adapter views made by earlier versions of this converter.
        temporary = weights.with_suffix(".safetensors.tmp")
        save_file(load_state(source), temporary)
        temporary.replace(weights)
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        entry = {
            "name": f"round_{round_number:02d}",
            "round": round_number,
            "source": str(source),
            "source_sha256": source_hash,
            "adapter": str(adapter_dir),
            "weights_sha256": sha256(weights),
            **delta_mapping.get(round_number, {"delta_id": None}),
        }
        entries.append(entry)

    before_offline = None
    after_offline = None
    if offline_at:
        offline_time = dt.datetime.fromisoformat(offline_at).replace(
            tzinfo=ZoneInfo("Australia/Melbourne")
        ).timestamp()
        all_checkpoints = sorted(
            checkpoint_dir.glob("sync_round_*.pth"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        before = [path for path in all_checkpoints if path.stat().st_mtime <= offline_time]
        after = [path for path in all_checkpoints if path.stat().st_mtime > offline_time]
        if before:
            before_offline = int(before[-1].stem.rsplit("_", 1)[1])
        if after:
            after_offline = int(after[0].stem.rsplit("_", 1)[1])

    manifest = {
        "schema_version": 1,
        "kind": "cross-country-fl-goqa-trajectory",
        "node_id": node_id,
        "base_model": str(base_model),
        "last_round": args.last_round,
        "selected_rounds": selected_rounds,
        "peer_offline_at": offline_at,
        "last_checkpoint_before_offline": before_offline,
        "first_checkpoint_after_offline": after_offline,
        "adapters": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    shards = [[], []]
    for entry in entries:
        shards[(entry["round"] - 1) % 2].append(entry)
    for shard_index, shard_entries in enumerate(shards):
        shard = {
            **{key: value for key, value in manifest.items() if key != "adapters"},
            "shard": shard_index,
            "include_base": shard_index == 0,
            "adapters": shard_entries,
        }
        (output_dir / f"adapter_manifest_gpu{shard_index}.json").write_text(
            json.dumps(shard, indent=2) + "\n"
        )
    print(json.dumps({"manifest": str(manifest_path), "adapters": len(entries), "offline_at": offline_at}, indent=2))
    print("JOINT GOQA TRAJECTORY PREPARATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
