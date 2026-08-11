#!/usr/bin/env python3
"""Strict dense-delta wire contract for the two-site Phase 8 experiment."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load

from phase2.adapter_delta import require_compatible, state_sha256


FORMAT = "slakshna-phase8-dense-delta"
VERSION = 1
CODEC = "base64+zlib+safetensors-fp32"
SITES = ("site-a", "site-b")
MAX_WIRE_BYTES = 7 * 1024 * 1024
MAX_COMPRESSED_BYTES = 6 * 1024 * 1024
MAX_RAW_BYTES = 16 * 1024 * 1024
REQUIRED_FIELDS = {
    "format",
    "version",
    "codec",
    "sender_site",
    "sender_node_id",
    "round",
    "base_state_sha256",
    "delta_file_sha256",
    "delta_state_sha256",
    "raw_bytes",
    "compressed_bytes",
    "tensor_count",
    "parameter_count",
    "data",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} is not a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a SHA-256 hex digest") from exc
    return value


def other_site(site: str) -> str:
    if site not in SITES:
        raise RuntimeError(f"invalid Phase 8 site: {site}")
    return "site-b" if site == "site-a" else "site-a"


def _decode_zlib(compressed: bytes) -> bytes:
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise RuntimeError(f"compressed delta exceeds {MAX_COMPRESSED_BYTES} bytes")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, MAX_RAW_BYTES + 1)
    if decoder.unconsumed_tail or len(raw) > MAX_RAW_BYTES:
        raise RuntimeError(f"uncompressed delta exceeds {MAX_RAW_BYTES} bytes")
    raw += decoder.flush(MAX_RAW_BYTES + 1 - len(raw))
    if len(raw) > MAX_RAW_BYTES or not decoder.eof or decoder.unused_data:
        raise RuntimeError("invalid or oversized zlib delta stream")
    return raw


def _load_raw_state(raw: bytes) -> dict[str, torch.Tensor]:
    try:
        state = {name: value.float().contiguous() for name, value in load(raw).items()}
    except Exception as exc:
        raise RuntimeError(f"invalid safetensors delta: {exc}") from exc
    if not state:
        raise RuntimeError("decoded delta is empty")
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"decoded delta tensor is non-finite: {name}")
    return state


@dataclass(frozen=True)
class DecodedDelta:
    envelope: dict[str, Any]
    raw: bytes
    state: dict[str, torch.Tensor]


def encode_delta_file(
    path: Path,
    *,
    sender_site: str,
    sender_node_id: str,
    round_number: int,
    base_state_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if sender_site not in SITES:
        raise RuntimeError(f"invalid sender site: {sender_site}")
    if not sender_node_id:
        raise RuntimeError("sender node id is empty")
    if round_number <= 0:
        raise RuntimeError(f"invalid round: {round_number}")
    require_sha256(base_state_sha256, "base_state_sha256")
    raw = path.resolve().read_bytes()
    if len(raw) > MAX_RAW_BYTES:
        raise RuntimeError(f"raw delta exceeds {MAX_RAW_BYTES} bytes")
    state = _load_raw_state(raw)
    compressed = zlib.compress(raw, level=9)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise RuntimeError(f"compressed delta exceeds {MAX_COMPRESSED_BYTES} bytes")
    envelope = {
        "format": FORMAT,
        "version": VERSION,
        "codec": CODEC,
        "sender_site": sender_site,
        "sender_node_id": sender_node_id,
        "round": round_number,
        "base_state_sha256": base_state_sha256,
        "delta_file_sha256": sha256_bytes(raw),
        "delta_state_sha256": state_sha256(state),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "data": base64.b64encode(compressed).decode("ascii"),
    }
    payload = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    if len(payload.encode("utf-8")) > MAX_WIRE_BYTES:
        raise RuntimeError(f"wire envelope exceeds {MAX_WIRE_BYTES} bytes")
    public_manifest = {key: value for key, value in envelope.items() if key != "data"}
    public_manifest["wire_bytes"] = len(payload.encode("utf-8"))
    return payload, public_manifest


def decode_delta_payload(
    payload: str,
    *,
    expected_sender_site: str,
    expected_sender_node_id: str,
    expected_round: int,
    expected_base_state_sha256: str,
) -> DecodedDelta:
    wire_bytes = len(payload.encode("utf-8"))
    if wire_bytes > MAX_WIRE_BYTES:
        raise RuntimeError(f"wire envelope exceeds {MAX_WIRE_BYTES} bytes")
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Phase 8 JSON envelope: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != REQUIRED_FIELDS:
        present = set(envelope) if isinstance(envelope, dict) else set()
        raise RuntimeError(
            f"Phase 8 envelope fields differ: missing={sorted(REQUIRED_FIELDS - present)}, "
            f"extra={sorted(present - REQUIRED_FIELDS)}"
        )
    expected_values = {
        "format": FORMAT,
        "version": VERSION,
        "codec": CODEC,
        "sender_site": expected_sender_site,
        "sender_node_id": expected_sender_node_id,
        "round": expected_round,
        "base_state_sha256": expected_base_state_sha256,
    }
    for key, expected in expected_values.items():
        if envelope[key] != expected:
            raise RuntimeError(f"Phase 8 envelope {key} mismatch: {envelope[key]!r} != {expected!r}")
    require_sha256(envelope["base_state_sha256"], "base_state_sha256")
    require_sha256(envelope["delta_file_sha256"], "delta_file_sha256")
    require_sha256(envelope["delta_state_sha256"], "delta_state_sha256")
    for key in ("raw_bytes", "compressed_bytes", "tensor_count", "parameter_count"):
        if not isinstance(envelope[key], int) or envelope[key] <= 0:
            raise RuntimeError(f"Phase 8 envelope {key} must be a positive integer")
    if not isinstance(envelope["data"], str):
        raise RuntimeError("Phase 8 envelope data must be base64 text")
    try:
        compressed = base64.b64decode(envelope["data"], validate=True)
    except Exception as exc:
        raise RuntimeError(f"invalid Phase 8 base64 delta: {exc}") from exc
    if len(compressed) != envelope["compressed_bytes"]:
        raise RuntimeError("compressed delta byte count mismatch")
    raw = _decode_zlib(compressed)
    if len(raw) != envelope["raw_bytes"]:
        raise RuntimeError("raw delta byte count mismatch")
    if sha256_bytes(raw) != envelope["delta_file_sha256"]:
        raise RuntimeError("delta file SHA-256 mismatch")
    state = _load_raw_state(raw)
    if len(state) != envelope["tensor_count"]:
        raise RuntimeError("delta tensor count mismatch")
    if sum(value.numel() for value in state.values()) != envelope["parameter_count"]:
        raise RuntimeError("delta parameter count mismatch")
    if state_sha256(state) != envelope["delta_state_sha256"]:
        raise RuntimeError("delta state SHA-256 mismatch")
    return DecodedDelta(envelope=envelope, raw=raw, state=state)


def aggregate_equal_weight(
    base: dict[str, torch.Tensor],
    own_delta: dict[str, torch.Tensor],
    peer_delta: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    require_compatible(base, own_delta, "base/own delta")
    require_compatible(base, peer_delta, "base/peer delta")
    aggregate = {
        name: (own_delta[name] * 0.5 + peer_delta[name] * 0.5).float().contiguous()
        for name in sorted(base)
    }
    global_state = {
        name: (base[name] + aggregate[name]).float().contiguous() for name in sorted(base)
    }
    if not all(torch.isfinite(value).all() for value in global_state.values()):
        raise RuntimeError("aggregated global state is non-finite")
    return aggregate, global_state
