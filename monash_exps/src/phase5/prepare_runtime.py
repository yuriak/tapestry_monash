#!/usr/bin/env python3
"""Materialize one isolated Phase 5 Rust peer runtime."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--peer-name", choices=("peer-a", "peer-b"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--p2p-port", type=int, required=True)
    parser.add_argument("--ws-port", type=int, required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--seed-peer", action="append", default=[])
    parser.add_argument("--epoch-duration", type=int, default=120)
    parser.add_argument("--sync-deadline", type=int, default=115)
    parser.add_argument("--config-name", default="node.toml")
    parser.add_argument("--reuse-runtime", action="store_true")
    args = parser.parse_args()

    if args.gpu_id < 0 or args.gpu_id > 31:
        raise RuntimeError(f"invalid GPU id: {args.gpu_id}")
    ports = [args.p2p_port, args.ws_port, args.api_port]
    if len(set(ports)) != 3 or any(port < 1024 or port > 65535 for port in ports):
        raise RuntimeError(f"invalid Phase 5 ports: {ports}")
    if args.epoch_duration <= 0 or not 0 < args.sync_deadline < args.epoch_duration:
        raise RuntimeError("sync deadline must be positive and shorter than the epoch")

    runtime = args.runtime_dir.resolve()
    data_dir = args.data_dir.resolve()
    if args.reuse_runtime:
        if not runtime.is_dir() or not data_dir.is_dir():
            raise RuntimeError("--reuse-runtime requires existing runtime and data directories")
    else:
        runtime.mkdir(parents=True, exist_ok=False)
        data_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.bridge.resolve(), runtime / "ml_engine.py")

    seed_peers = ", ".join(json.dumps(peer) for peer in args.seed_peer)
    replacements = {
        "__DATA_DIR__": str(data_dir),
        "__PEER_NAME__": args.peer_name,
        "__GPU_ID__": str(args.gpu_id),
        "__P2P_PORT__": str(args.p2p_port),
        "__WS_PORT__": str(args.ws_port),
        "__API_PORT__": str(args.api_port),
        "__SEED_PEERS__": seed_peers,
        "__EPOCH_DURATION__": str(args.epoch_duration),
        "__SYNC_DEADLINE__": str(args.sync_deadline),
    }
    text = args.template.resolve().read_text(encoding="utf-8")
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    unresolved = [marker for marker in replacements if marker in text]
    if unresolved:
        raise RuntimeError(f"unresolved Phase 5 markers: {unresolved}")
    (runtime / args.config_name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
