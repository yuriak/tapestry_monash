#!/usr/bin/env python3
"""Apply both transmitted round-10 deltas without launching another trainer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SITES = ("au", "india")
EXPECTED_TENSORS = 128
EXPECTED_PARAMETERS = 8_388_608


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_state(state: Any, label: str) -> dict[str, Any]:
    if not isinstance(state, dict) or "dummy" in state:
        raise RuntimeError(f"{label}: invalid or dummy LoRA state")
    if len(state) != EXPECTED_TENSORS:
        raise RuntimeError(f"{label}: expected {EXPECTED_TENSORS} tensors")
    parameter_count = sum(
        value.numel() for value in state.values() if torch.is_tensor(value)
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"{label}: parameter count {parameter_count} != {EXPECTED_PARAMETERS}"
        )
    if any(
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or not torch.isfinite(value).all().item()
        for value in state.values()
    ):
        raise RuntimeError(f"{label}: non-floating or non-finite tensor")
    return {
        "tensor_count": len(state),
        "parameter_count": parameter_count,
        "l2_norm": math.sqrt(
            sum(value.float().pow(2).sum().item() for value in state.values())
        ),
    }


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    audit_state(state, str(path))
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def identities(run_root: Path) -> dict[str, str]:
    result = {}
    for site in SITES:
        value = json.loads((run_root / site / "identity.json").read_text())
        if value.get("site") != site or not value.get("node_id"):
            raise RuntimeError(f"invalid identity for {site}")
        result[site] = value["node_id"]
    if len(set(result.values())) != len(SITES):
        raise RuntimeError("site identities are not unique")
    return result


def final_payload_history(
    run_root: Path, node_ids: dict[str, str]
) -> tuple[int, int, dict[int, dict[str, dict[str, str]]]]:
    # Each completion snapshot contains the same converged two-peer history. Read
    # one copy only; these files are deliberately large because upstream embeds
    # the Base64 payloads in the API response.
    completion = json.loads((run_root / "au" / "COMPLETED.json").read_text())
    records = completion["api"]["updates"]["updates"]
    model_updates = []
    for record in records:
        update = record.get("kind", {}).get("ModelUpdate")
        signature = record.get("signature", "")
        if update is None or not signature.startswith("node_signature_"):
            continue
        epoch = int(signature.rsplit("_", 1)[1])
        model_updates.append((epoch, record["node_id"], update))
    if not model_updates:
        raise RuntimeError("completion history contains no model updates")
    epochs = sorted({item[0] for item in model_updates})
    if len(epochs) < 2:
        raise RuntimeError("completion history contains fewer than two FL rounds")
    previous_epoch, final_epoch = epochs[-2:]
    selected: dict[int, dict[str, dict[str, str]]] = {
        previous_epoch: {},
        final_epoch: {},
    }
    for epoch, node_id, update in model_updates:
        if epoch in selected:
            if node_id in selected[epoch]:
                raise RuntimeError(f"duplicate update from {node_id} at {epoch}")
            selected[epoch][node_id] = update
    for epoch, payloads in selected.items():
        if set(payloads) != set(node_ids.values()):
            raise RuntimeError(
                f"senders at {epoch} {sorted(payloads)} "
                f"!= identities {sorted(node_ids.values())}"
            )
    return previous_epoch, final_epoch, selected


def decode_payloads(
    payloads: dict[str, dict[str, str]],
    decode_delta_envelope: Any,
    validate_peer_delta: Any,
    label: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    decoded = {}
    payload_manifest = {}
    for node_id, update in payloads.items():
        payload = update["compressed_delta"]
        delta = decode_delta_envelope(payload, torch.device("cpu"))
        if not validate_peer_delta(delta, max_allowed_norm=10.0):
            raise RuntimeError(f"{label} delta failed native validation: {node_id}")
        decoded[node_id] = delta
        payload_manifest[node_id] = {
            "delta_hash": update["delta_hash"],
            "base64_bytes": len(payload),
            "base64_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            **audit_state(delta, f"decoded {label} delta {node_id}"),
        }
    return decoded, payload_manifest


def state_difference(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[float, float]:
    l2_squared = 0.0
    max_abs = 0.0
    for key, left_value in left.items():
        difference = left_value.float() - right[key].float()
        l2_squared += difference.pow(2).sum().item()
        max_abs = max(max_abs, difference.abs().max().item())
    return math.sqrt(l2_squared), max_abs


def save_torch_state(state: dict[str, torch.Tensor], path: Path) -> None:
    torch.save(state, path)


def save_safe_state(state: dict[str, torch.Tensor], path: Path) -> None:
    save_file(state, path)


def final_weights(path: Path, observer: str, peers: set[str]) -> dict[str, float]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["observer_node"] == observer:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no trust rows in {path}")
    latest = max(row["timestamp"] for row in rows)
    weights = {
        row["peer_node"]: float(row["weight"])
        for row in rows
        if row["timestamp"] == latest
    }
    if set(weights) != peers or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise RuntimeError(f"invalid final trust weights for {observer}: {weights}")
    return weights


def write_once(path: Path, write: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite finalized artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "finalized_round10").resolve()
    if not (run_root / "COMPLETED.json").is_file():
        raise RuntimeError(f"formal run is not complete: {run_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text(), end="")
        print("M0 ROUND-10 FINALIZATION ALREADY COMPLETE")
        return 0

    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace / "Slakshna"))
    from federated_communication import (
        aggregate_deltas,
        decode_delta_envelope,
        validate_peer_delta,
    )

    node_ids = identities(run_root)
    previous_epoch, final_epoch, payload_history = final_payload_history(
        run_root, node_ids
    )
    previous_decoded, previous_payload_manifest = decode_payloads(
        payload_history[previous_epoch],
        decode_delta_envelope,
        validate_peer_delta,
        "round-9",
    )
    decoded, payload_manifest = decode_payloads(
        payload_history[final_epoch],
        decode_delta_envelope,
        validate_peer_delta,
        "round-10",
    )

    outputs = {}
    peer_set = set(node_ids.values())
    for site in SITES:
        node_id = node_ids[site]
        peer_id = next(value for value in peer_set if value != node_id)
        sync_root = next((run_root / site / "ml_models").glob("sync_ckpt_*"))
        base_path = sync_root / "sync_round_9.pth"
        raw_path = sync_root / "sync_round_10.pth"
        base = load_state(base_path)
        raw = load_state(raw_path)
        weights = final_weights(
            run_root / site / "logs" / "trust_scores_new.csv", node_id, peer_set
        )
        # The native round-10 checkpoint was formed from the site's fresh
        # round-10 delta plus its peer's latest available round-9 delta. Rebuild
        # it first so this finalization fails closed if that semantic assumption
        # ever changes upstream.
        raw_inputs = {
            node_id: decoded[node_id],
            peer_id: previous_decoded[peer_id],
        }
        reconstructed_raw_delta = aggregate_deltas(raw_inputs, weights)
        reconstructed_raw = {
            key: (
                base[key].float() + reconstructed_raw_delta[key].float()
            ).to(base[key].dtype)
            for key in base
        }
        raw_reproduction_l2, raw_reproduction_max_abs = state_difference(
            reconstructed_raw, raw
        )
        if raw_reproduction_max_abs > 1e-5:
            raise RuntimeError(
                f"{site}: native round-10 reproduction failed: "
                f"max_abs={raw_reproduction_max_abs:.8g}, "
                f"l2={raw_reproduction_l2:.8g}"
            )

        aggregated = aggregate_deltas(decoded, weights)
        if set(aggregated) != set(base):
            raise RuntimeError(f"{site}: aggregated delta schema differs from round-9 base")
        final = {
            key: (base[key].float() + aggregated[key].float()).to(base[key].dtype).contiguous()
            for key in base
        }
        final_audit = audit_state(final, f"{site} finalized adapter")
        difference_l2, difference_max_abs = state_difference(final, raw)
        if difference_l2 <= 0.0:
            raise RuntimeError(f"{site}: finalization did not change the raw round-10 state")

        site_dir = output_dir / site
        site_dir.mkdir()
        pth_path = site_dir / "adapter_model.pth"
        safe_path = site_dir / "adapter_model.safetensors"
        write_once(pth_path, lambda path, state=final: save_torch_state(state, path))
        write_once(safe_path, lambda path, state=final: save_safe_state(state, path))
        outputs[site] = {
            "node_id": node_id,
            "peer_id": peer_id,
            "round9_base": str(base_path),
            "round9_base_sha256": sha256(base_path),
            "raw_round10": str(raw_path),
            "raw_round10_sha256": sha256(raw_path),
            "weights": weights,
            "raw_round10_reproduction_l2": raw_reproduction_l2,
            "raw_round10_reproduction_max_abs": raw_reproduction_max_abs,
            "finalized_pth": str(pth_path),
            "finalized_pth_sha256": sha256(pth_path),
            "finalized_safetensors": str(safe_path),
            "finalized_safetensors_sha256": sha256(safe_path),
            "difference_from_raw_round10_l2": difference_l2,
            "difference_from_raw_round10_max_abs": difference_max_abs,
            **final_audit,
        }

    manifest = {
        "schema_version": 1,
        "operation": "native-semantics-round10-aggregation-only",
        "run_root": str(run_root),
        "final_epoch_boundary": final_epoch,
        "previous_epoch_boundary": previous_epoch,
        "round": 10,
        "source_passes": 2,
        "node_ids": node_ids,
        "payloads": payload_manifest,
        "previous_payloads": previous_payload_manifest,
        "outputs": outputs,
    }
    write_once(
        manifest_path,
        lambda path: path.write_text(json.dumps(manifest, indent=2) + "\n"),
    )
    print(json.dumps(manifest, indent=2))
    print("M0 ROUND-10 FINALIZATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
