#!/usr/bin/env python3
"""Create one isolated stock-Slakshna runtime for the formal local FL run."""

from __future__ import annotations

import argparse
import errno
import hashlib
import ipaddress
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import toml
import yaml

SITES = {
    "au": {"view": "australia_nz", "steps": 117, "warmup": 4, "rows": 9337},
    "india": {"view": "south_asia", "steps": 192, "warmup": 6, "rows": 15331},
}
G0_SHA256 = "0e87f53ad240ca04a4aaadc93079643e3b7cc1d0b38b7574e8b87e559361918c"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_stock_sources(source: Path, runtime: Path) -> dict[str, str]:
    inputs = [source / "ml_engine.py"]
    inputs.extend(sorted((source / "federated_communication").rglob("*.py")))
    copied = {}
    for item in inputs:
        relative = item.relative_to(source)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        if sha256_file(item) != sha256_file(destination):
            raise RuntimeError(f"stock runtime copy differs: {relative}")
        copied[relative.as_posix()] = sha256_file(destination)
    return copied


def install_cache(source: Path, destination: Path) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        target = destination / item.name
        if target.exists():
            if sha256_file(target) != sha256_file(item):
                raise RuntimeError(f"existing cache file differs: {target}")
        else:
            try:
                os.link(item, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(item, target)
        records.append(
            {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    if not any(str(item["path"]).endswith(".parquet") for item in records):
        raise RuntimeError(f"no parquet files found in {source}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--allowed-peer-endpoint", required=True)
    parser.add_argument("--seed-peer-endpoint")
    parser.add_argument("--seed-peer-public-ip")
    parser.add_argument("--seed-peer-public-port", type=int)
    parser.add_argument("--p2p-port", required=True, type=int)
    parser.add_argument("--api-port", required=True, type=int)
    parser.add_argument("--ws-port", required=True, type=int)
    parser.add_argument("--federation-id", required=True)
    parser.add_argument("--epoch-duration", type=int, default=780)
    parser.add_argument("--sync-deadline", type=int, default=720)
    args = parser.parse_args()

    if args.sync_deadline >= args.epoch_duration:
        parser.error("sync deadline must be below the epoch duration")
    if len(args.allowed_peer_endpoint) != 64:
        parser.error("allowed peer must be a 64-character Iroh EndpointId")
    seed_values = (
        args.seed_peer_endpoint,
        args.seed_peer_public_ip,
        args.seed_peer_public_port,
    )
    if any(value is not None for value in seed_values) and not all(
        value is not None for value in seed_values
    ):
        parser.error(
            "seed peer endpoint, public IP, and public port must be set together"
        )
    seed = None
    if args.seed_peer_endpoint is not None:
        if len(args.seed_peer_endpoint) != 64:
            parser.error("seed peer must be a 64-character Iroh EndpointId")
        try:
            peer_ip = ipaddress.ip_address(args.seed_peer_public_ip)
        except ValueError as error:
            parser.error(f"seed peer public IP is invalid: {error}")
        if peer_ip.version != 4 or not peer_ip.is_global:
            parser.error(
                f"seed address must be a global IPv4 routed through Playit, got {peer_ip}"
            )
        if not 1024 <= args.seed_peer_public_port <= 65535:
            parser.error("seed peer public port must be between 1024 and 65535")
        seed = f"{args.seed_peer_endpoint}@{peer_ip}:{args.seed_peer_public_port}"

    workspace = Path(__file__).resolve().parents[3]
    experiment = workspace / "monash_exps"
    source = workspace / "Slakshna"
    runtime_assets = experiment / ".runtime"
    profile = SITES[args.site]
    template = experiment / "configs/m0_fl" / f"node_template.{args.site}.yaml"
    model = (runtime_assets / "models/m0/OLMo-2-1124-7B-Instruct").resolve()
    g0 = runtime_assets / "artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth"
    cache_parent = (
        runtime_assets / "data/m0/tokenized/olmo2-7b-chatml-seq1024" / profile["view"]
    )
    cache_dirs = sorted(path for path in cache_parent.iterdir() if path.is_dir())
    if len(cache_dirs) != 1:
        raise RuntimeError(
            f"expected one cache directory for {args.site}: {cache_dirs}"
        )
    if sha256_file(g0) != G0_SHA256:
        raise RuntimeError(f"frozen G0 checksum mismatch: {g0}")

    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    copied_sources = copy_stock_sources(source, runtime)
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["model"]["name"] = str(model)
    if config["training"]["max_steps"] != profile["steps"]:
        raise RuntimeError("site template step count changed unexpectedly")
    (runtime / "node_template.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    cache_root = runtime / "data" / f"data_{args.node_id}" / "tokenized_cache"
    cache_destination = cache_root / cache_dirs[0].name
    cache_records = install_cache(cache_dirs[0], cache_destination)
    model_root = runtime / "ml_models"
    model_root.mkdir(parents=True, exist_ok=True)
    base_lora = model_root / f"{args.node_id}_base_lora.pth"
    if base_lora.exists():
        if sha256_file(base_lora) != G0_SHA256:
            raise RuntimeError(
                f"base LoRA already changed; refusing to reset training state: {base_lora}"
            )
    else:
        shutil.copy2(g0, base_lora)

    node_config = {
        "federation": {"id": args.federation_id, "name": "M0 Local FL"},
        "training": {
            "epoch_duration_secs": args.epoch_duration,
            "sync_deadline_secs": args.sync_deadline,
            "expected_peers": 2,
        },
        "compression": {
            "enabled": True,
            "sparsity": 0.1,
            "quantization": "symmetric_int8",
            "allow_legacy_delta_format": False,
            "max_payload_bytes": 7_340_032,
            "max_tensor_elements": 10_000_000,
        },
        "node": {
            "id": f"m0-fl-{args.site}",
            "data_dir": str((runtime / "node-state").resolve()),
            "gpu_id": 0,
            "num_gpus": 2,
        },
        "network": {
            "host": "0.0.0.0",
            "p2p_port": args.p2p_port,
            "ws_port": args.ws_port,
            "api_port": args.api_port,
            "peers": [seed] if seed else [],
            "allowed_peers": [args.allowed_peer_endpoint],
        },
        "discovery": {"mdns": False, "dht": False, "dns": False, "relay": False},
        "logging": {"level": "info"},
    }
    node_path = runtime / "node.toml"
    node_path.write_text(toml.dumps(node_config), encoding="utf-8")

    from bhaskera.config import load_config

    audited = load_config(str(runtime / "node_template.yaml"))
    observed = (
        audited.training.batch_size,
        audited.training.grad_accum,
        audited.training.max_steps,
        audited.training.warmup_steps,
        audited.training.distributed.strategy,
        audited.lora.r,
        audited.data.seq_len,
    )
    expected = (2, 4, profile["steps"], profile["warmup"], "ddp", 16, 1024)
    if observed != expected:
        raise RuntimeError(
            f"resolved Bhaskera config mismatch: {observed} != {expected}"
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "site": args.site,
        "node_id": args.node_id,
        "allowed_peer_endpoint": args.allowed_peer_endpoint,
        "seed_peer": seed,
        "runtime": str(runtime),
        "rows": profile["rows"],
        "steps_per_round": profile["steps"],
        "rounds": 10,
        "effective_epochs": profile["steps"] * 10 * 16 / profile["rows"],
        "epoch_duration_secs": args.epoch_duration,
        "sync_deadline_secs": args.sync_deadline,
        "stock_source_hashes": copied_sources,
        "cache": cache_records,
        "base_lora_sha256": sha256_file(base_lora),
        "node_template_sha256": sha256_file(runtime / "node_template.yaml"),
        "node_config_sha256": sha256_file(node_path),
    }
    (runtime / "preparation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
