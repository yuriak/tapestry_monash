#!/usr/bin/env python3
"""Fail-closed verifier for the Phase 3 Slakshna single-node lifecycle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def fail(message: str) -> None:
    raise RuntimeError(message)


def model_updates(api_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for record in api_payload.get("updates", []):
        kind = record.get("kind", {})
        update = kind.get("ModelUpdate") if isinstance(kind, dict) else None
        if update is not None:
            records.append({"record": record, "update": update})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--rust-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    output = json.loads((root / "ml-engine-output.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "ml-bridge-audit.json").read_text(encoding="utf-8"))
    phase1 = json.loads(
        (root / "local-training/verification-summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((root / "update/update-manifest.json").read_text(encoding="utf-8"))
    updates_payload = json.loads(args.updates.read_text(encoding="utf-8"))
    status_payload = json.loads(args.status.read_text(encoding="utf-8"))
    rust_log = args.rust_log.read_text(encoding="utf-8", errors="replace")

    if phase1.get("status") != "PASS":
        fail("the Rust-triggered local training verifier did not pass")
    launcher = phase1["checks"]["launcher_resources"]["detail"]
    if launcher.get("visible_gpus") != 1 or launcher.get("ray_resources", {}).get("GPU") != 1.0:
        fail(f"training did not use exactly one allocated GPU: {launcher}")
    if manifest.get("nonzero_delta_values", 0) <= 0:
        fail("canonical LoRA delta contains no non-zero values")

    compressed_delta = output.get("compressed_delta", "")
    if not compressed_delta:
        fail("ML engine output has an empty compressed_delta")
    try:
        delta_bytes = zlib.decompress(base64.b64decode(compressed_delta, validate=True))
    except Exception as exc:
        fail(f"compressed_delta is not valid base64(zlib(...)): {exc}")
    delta_path = root / "update/delta.safetensors"
    if delta_bytes != delta_path.read_bytes():
        fail("decoded Slakshna payload differs from canonical delta.safetensors")
    model_hash = hashlib.sha256(delta_bytes).hexdigest()
    if output.get("model_hash") != model_hash or audit.get("model_hash") != model_hash:
        fail("model_hash does not identify the canonical delta bytes")
    state = load(delta_bytes)
    if len(state) != manifest.get("tensor_count"):
        fail("decoded tensor count differs from the update manifest")
    if sum(int(torch.count_nonzero(value)) for value in state.values()) <= 0:
        fail("decoded delta tensors are all zero")
    if any(not torch.isfinite(value).all() for value in state.values()):
        fail("decoded delta contains NaN or Inf")

    records = model_updates(updates_payload)
    if len(records) != 1:
        fail(f"expected exactly one Rust ModelUpdate, found {len(records)}")
    record = records[0]["record"]
    rust_update = records[0]["update"]
    if rust_update.get("delta_hash") == "error_hash":
        fail("Rust recorded its fallback error_hash update")
    if rust_update.get("delta_hash") != model_hash:
        fail("Rust history delta_hash differs from the bridge output")
    if rust_update.get("compressed_delta") != compressed_delta:
        fail("Rust history payload differs from the bridge output")
    if record.get("node_id") != audit.get("node_id"):
        fail("Rust history node identity differs from the bridge argument")

    digest = hashlib.sha256()
    digest.update(record["node_id"].encode())
    digest.update(record["prev_hash"].encode())
    digest.update(b"model_update")
    digest.update(model_hash.encode())
    digest.update(compressed_delta.encode())
    if record.get("hash") != digest.hexdigest():
        fail("Rust ModelUpdate hash chain record is invalid")

    required_log_markers = ["Node is LIVE", "ML Engine pinned to GPU 0", "Local Training Complete"]
    missing_markers = [marker for marker in required_log_markers if marker not in rust_log]
    if missing_markers:
        fail(f"Rust log is missing lifecycle markers: {missing_markers}")
    if "Python ML Engine failed" in rust_log or "Failed to start Python process" in rust_log:
        fail("Rust log reports an ML engine failure")
    if status_payload.get("federation_id") != "slakshna-phase3-local":
        fail(f"unexpected federation status: {status_payload}")

    summary = {
        "status": "PASS",
        "mode": "phase3",
        "checks": {
            "rust_node_lifecycle": {
                "status": "PASS",
                "detail": {
                    "federation_id": status_payload["federation_id"],
                    "endpoint_id": status_payload.get("endpoint_id"),
                    "model_updates": len(records),
                    "record_hash": record["hash"],
                },
            },
            "rust_triggered_training": {
                "status": "PASS",
                "detail": {
                    "gpu_names": launcher["gpu_names"],
                    "visible_gpus": launcher["visible_gpus"],
                    "checkpoint_step": phase1["checks"]["checkpoint"]["detail"]["step"],
                    "final_median_loss": audit["final_median_loss"],
                },
            },
            "recorded_update": {
                "status": "PASS",
                "detail": {
                    "model_hash": model_hash,
                    "tensor_count": manifest["tensor_count"],
                    "parameter_count": manifest["parameter_count"],
                    "nonzero_delta_values": manifest["nonzero_delta_values"],
                    "uncompressed_bytes": len(delta_bytes),
                    "compressed_bytes": audit["delta_compressed_bytes"],
                },
            },
        },
        "failures": [],
    }
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PHASE3 PASSED")


if __name__ == "__main__":
    main()
