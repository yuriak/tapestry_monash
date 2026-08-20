#!/usr/bin/env python3
"""Tokenize the seven formal M0 views through Bhaskera's native CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from bhaskera.config import load_config

SCHEMA_VERSION = 1
WORKFLOW_VERSION = "m0-native-bhaskera-tokenized-views-v1"
PROCESS_ORDER = (
    "australia_nz",
    "south_asia",
    "centralized_variant_1",
    "australia_nz_western_europe",
    "centralized_variant_2",
    "australia_nz_us_canada_uk",
    "centralized_variant_3",
)
VIEW_TMP_IDS = {name: f"v{index}" for index, name in enumerate(PROCESS_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_portable(value: str, workspace_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def validate_inputs(
    *,
    workspace_root: Path,
    prepared_manifest_path: Path,
    decision_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, str]]:
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))

    hashes = {
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "sequence_decision_sha256": sha256_file(decision_path),
        "tokenize_config_sha256": sha256_file(config_path),
    }
    if decision.get("status") != "frozen" or decision.get("selected_seq_len") != 1024:
        raise RuntimeError(f"sequence-length decision is not frozen at 1024: {decision}")
    if decision.get("prepared_manifest_sha256") != hashes["prepared_manifest_sha256"]:
        raise RuntimeError("sequence decision does not bind the current prepared manifest")
    audit_path = resolve_portable(decision["tokenizer_length_audit"], workspace_root)
    if sha256_file(audit_path) != decision["tokenizer_length_audit_sha256"]:
        raise RuntimeError("tokenizer-length audit hash no longer matches the decision")
    if tuple(decision.get("formal_views", ())) != (
        "south_asia",
        "australia_nz",
        "australia_nz_western_europe",
        "australia_nz_us_canada_uk",
        "centralized_variant_1",
        "centralized_variant_2",
        "centralized_variant_3",
    ):
        raise RuntimeError("unexpected formal-view list in sequence decision")
    if set(PROCESS_ORDER) != set(decision["formal_views"]):
        raise RuntimeError("tokenization process order does not cover the formal views")

    expected_config = {
        "model.name": "monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct",
        "data.name": "local",
        "data.format": "chatml",
        "data.seq_len": 1024,
        "data.pack_sequences": False,
        "data.is_cpt": False,
        "data.tokenize_compression": "snappy",
    }
    actual_config = {
        "model.name": config.model.name,
        "data.name": config.data.name,
        "data.format": config.data.format,
        "data.seq_len": config.data.seq_len,
        "data.pack_sequences": config.data.pack_sequences,
        "data.is_cpt": config.data.is_cpt,
        "data.tokenize_compression": config.data.tokenize_compression,
    }
    if actual_config != expected_config:
        raise RuntimeError(
            f"tokenization config mismatch: actual={actual_config}, expected={expected_config}"
        )
    model_path = resolve_portable(config.model.name, workspace_root)
    if not (model_path / "tokenizer.json").is_file():
        raise RuntimeError(f"configured local tokenizer is missing: {model_path}")

    for view_name in PROCESS_ORDER:
        view = prepared["views"].get(view_name)
        if view is None:
            raise RuntimeError(f"prepared manifest is missing formal view {view_name}")
        source = resolve_portable(view["path"], workspace_root)
        if not source.is_file() or sha256_file(source) != view["sha256"]:
            raise RuntimeError(f"prepared source hash mismatch: {view_name}")

    return prepared, decision, config, hashes


def binding_payload(
    view_name: str,
    view: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "view": view_name,
        "source_path": view["path"],
        "source_rows": view["rows"],
        "source_sha256": view["sha256"],
        "seq_len": 1024,
        **hashes,
    }


def ensure_binding(view_root: Path, expected: dict[str, Any]) -> None:
    binding_path = view_root / "source-binding.json"
    cache_dirs = [path for path in view_root.glob("local_train_*") if path.is_dir()]
    if binding_path.is_file():
        actual = json.loads(binding_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"existing token cache binding is stale at {binding_path}; "
                "preserve it for review and use a new M0_TOKENIZED_ROOT"
            )
        return
    if cache_dirs:
        raise RuntimeError(
            f"unbound native token cache exists under {view_root}; "
            "preserve it for review and use a new M0_TOKENIZED_ROOT"
        )
    atomic_json(expected, binding_path)


def matching_cache(view_root: Path, config: Any) -> Path:
    matches = []
    for metadata_path in view_root.glob("local_train_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model_name": config.model.name,
            "seq_len": 1024,
            "dataset_name": "local_train",
            "format_name": "chatml",
            "format_options": {"messages_field": "messages"},
            "is_cpt": False,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one matching Bhaskera cache in {view_root}: {matches}")
    return matches[0].resolve()


def audit_cache(
    cache_path: Path,
    *,
    workspace_root: Path,
    config: Any,
    expected_rows: int,
) -> dict[str, Any]:
    metadata_path = cache_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_name": config.model.name,
        "seq_len": 1024,
        "dataset_name": "local_train",
        "num_rows": expected_rows,
        "format_name": "chatml",
        "format_options": {"messages_field": "messages"},
        "is_cpt": False,
        "schema": ["input_ids", "attention_mask", "labels"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"cache metadata mismatch at {cache_path}: "
                f"{key}={metadata.get(key)!r}, expected={value!r}"
            )

    parquet_files = sorted(cache_path.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"native cache has no Parquet files: {cache_path}")
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
    if rows != expected_rows:
        raise RuntimeError(f"Parquet rows mismatch at {cache_path}: {rows} != {expected_rows}")

    sample_batch = next(pq.ParquetFile(parquet_files[0]).iter_batches(batch_size=4))
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(sample_batch.schema.names):
        raise RuntimeError(f"native cache schema is incomplete: {sample_batch.schema}")
    samples = sample_batch.to_pylist()
    if not samples:
        raise RuntimeError(f"native cache contains no sample rows: {cache_path}")
    for index, row in enumerate(samples):
        lengths = {name: len(row[name]) for name in required}
        if set(lengths.values()) != {1024}:
            raise RuntimeError(f"sample {index} has invalid tensor lengths: {lengths}")
        for token, mask, label in zip(
            row["input_ids"], row["attention_mask"], row["labels"], strict=True
        ):
            if mask not in (0, 1) or label != (token if mask else -100):
                raise RuntimeError(f"sample {index} violates native SFT label masking")

    parquet = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for path in parquet_files
    ]
    return {
        "cache_path": portable(cache_path, workspace_root),
        "rows": rows,
        "seq_len": 1024,
        "columns": sorted(required),
        "metadata": metadata,
        "metadata_sha256": sha256_file(metadata_path),
        "parquet": parquet,
        "parquet_bytes": sum(item["bytes"] for item in parquet),
    }


def validate_cached_manifest(
    manifest_path: Path,
    *,
    binding: dict[str, Any],
    workspace_root: Path,
    config: Any,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("binding") != binding:
        raise RuntimeError(f"stale per-view token manifest: {manifest_path}")
    cache_path = resolve_portable(manifest["cache"]["cache_path"], workspace_root)
    current = audit_cache(
        cache_path,
        workspace_root=workspace_root,
        config=config,
        expected_rows=binding["source_rows"],
    )
    if current != manifest["cache"]:
        raise RuntimeError(f"token cache contents changed after manifesting: {cache_path}")
    return manifest


def run_native_tokenizer(
    *,
    python: Path,
    config_path: Path,
    source_path: Path,
    view_root: Path,
    workers: int,
    workspace_root: Path,
    view_name: str,
) -> None:
    # Ray appends a long session/sockets suffix and AF_UNIX paths are limited
    # to 107 bytes on Linux. Keep both the default base and per-view component
    # deliberately short; a descriptive path fails before Ray can start.
    ray_tmp = Path(
        os.environ.get("M0_RAY_TMPDIR_BASE", f"/tmp/m0r{os.getuid()}")
    ) / VIEW_TMP_IDS[view_name]
    ray_tmp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(workers),
            "RAY_TMPDIR": str(ray_tmp),
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    command = [
        str(python),
        "-m",
        "bhaskera.launcher.tokenize",
        "--config",
        str(config_path),
        "--split",
        "train",
        "--train-path",
        str(source_path),
        "--storage-path",
        str(view_root),
        "--num-workers",
        str(workers),
    ]
    print("Running native Bhaskera: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=workspace_root, env=environment, check=True)


def main() -> int:
    inferred_workspace = Path(__file__).resolve().parents[3]
    runtime = inferred_workspace / "monash_exps" / ".runtime"
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    parser.add_argument(
        "--prepared-manifest",
        type=Path,
        default=runtime / "data" / "m0" / "prepared" / "manifests" / "prepared-data.json",
    )
    parser.add_argument(
        "--sequence-decision",
        type=Path,
        default=runtime / "manifests" / "m0" / "sequence-length-decision.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=inferred_workspace
        / "monash_exps"
        / "configs"
        / "m0"
        / "tokenize_olmo2_7b_chatml.yaml",
    )
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=runtime / "data" / "m0" / "tokenized" / "olmo2-7b-chatml-seq1024",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=runtime / "manifests" / "m0" / "tokenized-formal-views.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    workspace_root = args.workspace_root.resolve()
    prepared_path = args.prepared_manifest.resolve()
    decision_path = args.sequence_decision.resolve()
    config_path = args.config.resolve()
    tokenized_root = args.tokenized_root.resolve()
    manifest_path = args.manifest.resolve()
    for required in (prepared_path, decision_path, config_path):
        if not required.is_file():
            parser.error(f"missing required input: {required}")

    prepared, _, config, hashes = validate_inputs(
        workspace_root=workspace_root,
        prepared_manifest_path=prepared_path,
        decision_path=decision_path,
        config_path=config_path,
    )
    tokenized_root.mkdir(parents=True, exist_ok=True)
    per_view: dict[str, Any] = {}
    # Keep the virtual-environment launcher path intact. Resolving this symlink
    # selects the standalone base interpreter and drops the venv site-packages
    # (including the editable Bhaskera installation) in child processes.
    python = Path(sys.executable).absolute()

    for view_name in PROCESS_ORDER:
        view = prepared["views"][view_name]
        view_root = tokenized_root / view_name
        view_root.mkdir(parents=True, exist_ok=True)
        binding = binding_payload(view_name, view, hashes)
        ensure_binding(view_root, binding)
        per_view_manifest = view_root / "m0-cache-manifest.json"
        existing = validate_cached_manifest(
            per_view_manifest,
            binding=binding,
            workspace_root=workspace_root,
            config=config,
        )
        if existing is not None:
            print(f"Verified existing native cache: {view_name}", flush=True)
            per_view[view_name] = existing
            continue

        print(f"Tokenizing formal view: {view_name} ({view['rows']} rows)", flush=True)
        source_path = resolve_portable(view["path"], workspace_root)
        run_native_tokenizer(
            python=python,
            config_path=config_path,
            source_path=source_path,
            view_root=view_root,
            workers=args.workers,
            workspace_root=workspace_root,
            view_name=view_name,
        )
        cache_path = matching_cache(view_root, config)
        cache = audit_cache(
            cache_path,
            workspace_root=workspace_root,
            config=config,
            expected_rows=view["rows"],
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "binding": binding,
            "cache": cache,
        }
        atomic_json(result, per_view_manifest)
        per_view[view_name] = result
        print(
            f"Completed {view_name}: {cache['rows']} rows, "
            f"{cache['parquet_bytes']} Parquet bytes",
            flush=True,
        )

    master = {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "tokenized_root": portable(tokenized_root, workspace_root),
        "seq_len": 1024,
        "formal_views": list(PROCESS_ORDER),
        "excluded_views": {"not_matched": "not in the first formal training matrix"},
        **hashes,
        "views": per_view,
        "total_rows_with_cross_view_reuse": sum(
            result["cache"]["rows"] for result in per_view.values()
        ),
        "total_parquet_bytes": sum(
            result["cache"]["parquet_bytes"] for result in per_view.values()
        ),
    }
    atomic_json(master, manifest_path)
    print(f"Master manifest: {manifest_path}")
    print("M0 NATIVE OFFLINE TOKENIZATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
