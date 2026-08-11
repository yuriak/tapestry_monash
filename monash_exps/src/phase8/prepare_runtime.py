#!/usr/bin/env python3
"""Create one cluster-neutral Phase 8 Slakshna runtime."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--site", choices=("site-a", "site-b"), required=True)
    parser.add_argument("--federation-id", required=True)
    parser.add_argument("--federation-name", default="Slakshna Phase 8 Cross-Cluster SFT")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--p2p-port", type=int, required=True)
    parser.add_argument("--ws-port", type=int, required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--seed-peer", action="append", default=[])
    parser.add_argument("--allowed-peer", action="append", default=[])
    parser.add_argument("--epoch-duration", type=int, default=900)
    parser.add_argument("--sync-deadline", type=int, default=870)
    parser.add_argument("--config-name", default="node.toml")
    parser.add_argument("--reuse-runtime", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{7,127}", args.federation_id):
        raise RuntimeError("federation id must be 8-128 safe characters")
    if args.gpu_id < 0 or args.gpu_id > 31:
        raise RuntimeError(f"invalid GPU id: {args.gpu_id}")
    ports = [args.p2p_port, args.ws_port, args.api_port]
    if len(set(ports)) != 3 or any(port < 1024 or port > 65535 for port in ports):
        raise RuntimeError(f"invalid Phase 8 ports: {ports}")
    if args.epoch_duration < 300 or not 0 < args.sync_deadline < args.epoch_duration:
        raise RuntimeError("sync deadline must be positive, shorter than epoch, and epoch >= 300")
    if len(set(args.allowed_peer)) != len(args.allowed_peer):
        raise RuntimeError("duplicate allowed peer IDs")
    if any("@" in peer for peer in args.allowed_peer):
        raise RuntimeError("allowed peers must be bare Iroh EndpointIds")

    runtime = args.runtime_dir.resolve()
    data_dir = args.data_dir.resolve()
    site_root = args.site_root.resolve()
    if not (site_root / "site-manifest.json").is_file():
        raise RuntimeError(f"site has not been prepared: {site_root}")
    if not (site_root / "global-0/g0-manifest.json").is_file():
        raise RuntimeError(f"G0 has not been installed: {site_root}")
    if args.reuse_runtime:
        if not runtime.is_dir() or not data_dir.is_dir():
            raise RuntimeError("--reuse-runtime requires existing runtime and data directories")
    else:
        runtime.mkdir(parents=True, exist_ok=False)
        data_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.bridge.resolve(), runtime / "ml_engine.py")

    replacements = {
        "__FEDERATION_ID__": args.federation_id,
        "__FEDERATION_NAME__": args.federation_name,
        "__DATA_DIR__": str(data_dir),
        "__SITE__": args.site,
        "__GPU_ID__": str(args.gpu_id),
        "__P2P_PORT__": str(args.p2p_port),
        "__WS_PORT__": str(args.ws_port),
        "__API_PORT__": str(args.api_port),
        "__SEED_PEERS__": ", ".join(json.dumps(value) for value in args.seed_peer),
        "__ALLOWED_PEERS__": ", ".join(json.dumps(value) for value in args.allowed_peer),
        "__EPOCH_DURATION__": str(args.epoch_duration),
        "__SYNC_DEADLINE__": str(args.sync_deadline),
    }
    text = args.template.resolve().read_text(encoding="utf-8")
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if unresolved:
        raise RuntimeError(f"unresolved Phase 8 template markers: {unresolved}")
    config_path = runtime / args.config_name
    config_path.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "site": args.site,
        "federation_id": args.federation_id,
        "gpu_id": args.gpu_id,
        "ports": {"p2p": args.p2p_port, "ws": args.ws_port, "api": args.api_port},
        "seed_peers": args.seed_peer,
        "allowed_peers": args.allowed_peer,
        "discovery": {"mdns": False, "dht": False, "dns": False, "relay": False},
        "config_name": args.config_name,
    }
    (runtime / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
