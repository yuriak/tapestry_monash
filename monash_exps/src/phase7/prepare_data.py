#!/usr/bin/env python3
"""Materialize and tokenize the fixed Phase 7 non-IID Dolly shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase1.prepare_experiment import download_model, resolve_template, tokenize  # noqa: E402


DATASET_ID = "databricks/databricks-dolly-15k"
DATASET_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DATASET_SPLIT = "train"
TRAIN_ROWS = 1152
VALIDATION_ROWS = 128
PEER_CATEGORIES = {
    "peer-a": {"closed_qa", "open_qa", "information_extraction"},
    "peer-b": {"brainstorming", "creative_writing", "summarization"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def chat_record(row: dict[str, Any]) -> dict[str, Any]:
    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    response = str(row.get("response") or "").strip()
    if not instruction or not response:
        raise RuntimeError("Dolly record has an empty instruction or response")
    user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    }


def materialize_shards(data_root: Path) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    data_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    selected: dict[str, list[tuple[int, dict[str, Any]]]] = {
        peer: [] for peer in PEER_CATEGORIES
    }
    for index, raw in enumerate(dataset):
        row = dict(raw)
        category = str(row.get("category") or "")
        for peer, categories in PEER_CATEGORIES.items():
            if category in categories and len(selected[peer]) < TRAIN_ROWS + VALIDATION_ROWS:
                selected[peer].append((index, row))
    manifests: dict[str, dict[str, Any]] = {}
    all_indices: set[int] = set()
    for peer, rows in selected.items():
        expected = TRAIN_ROWS + VALIDATION_ROWS
        if len(rows) != expected:
            raise RuntimeError(
                f"{peer} categories supplied {len(rows)} usable rows, expected {expected}"
            )
        indices = {index for index, _ in rows}
        if all_indices.intersection(indices):
            raise RuntimeError(f"{peer} overlaps another peer's source indices")
        all_indices.update(indices)
        train_rows = rows[:TRAIN_ROWS]
        validation_rows = rows[TRAIN_ROWS:]
        paths = {
            "train": data_root / f"{peer}-train-{TRAIN_ROWS}.jsonl",
            "validation": data_root / f"{peer}-validation-{VALIDATION_ROWS}.jsonl",
        }
        for split_name, split_rows in (
            ("train", train_rows),
            ("validation", validation_rows),
        ):
            temporary = paths[split_name].with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for _, row in split_rows:
                    handle.write(
                        json.dumps(chat_record(row), ensure_ascii=False, sort_keys=True) + "\n"
                    )
            temporary.replace(paths[split_name])
        category_counts: dict[str, int] = {}
        for _, row in rows:
            category = str(row["category"])
            category_counts[category] = category_counts.get(category, 0) + 1
        manifests[peer] = {
            "categories": sorted(PEER_CATEGORIES[peer]),
            "category_counts": category_counts,
            "train": {
                "path": str(paths["train"].resolve()),
                "rows": TRAIN_ROWS,
                "source_indices": [index for index, _ in train_rows],
                "sha256": sha256_file(paths["train"]),
            },
            "validation": {
                "path": str(paths["validation"].resolve()),
                "rows": VALIDATION_ROWS,
                "source_indices": [index for index, _ in validation_rows],
                "sha256": sha256_file(paths["validation"]),
            },
        }
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    bootstrap_root = artifact_root / "bootstrap"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    data_root = EXPERIMENT_ROOT / ".runtime" / "data" / "phase7" / DATASET_REVISION[:12]
    shards = materialize_shards(data_root)
    model_path = download_model()

    for peer, shard in shards.items():
        peer_dir = bootstrap_root / peer
        peer_dir.mkdir(parents=True, exist_ok=True)
        provisional = peer_dir / "provisional-config.yaml"
        resolved = peer_dir / "resolved-config.yaml"
        tokenized_root = data_root / "tokenized" / peer
        resolve_template(
            args.template.resolve(),
            provisional,
            model_path,
            Path(shard["train"]["path"]),
            tokenized_root.resolve(),
            peer_dir / "unused-checkpoints",
            f"{artifact_root.name}-bootstrap-{peer}",
            None,
        )
        tokenized_path = tokenize(
            provisional,
            peer_dir / "ray-tokenize",
            f"phase7_{DATASET_REVISION[:12]}_{peer}_{TRAIN_ROWS}",
        )
        from bhaskera.config import load_config

        cfg = load_config(str(provisional))
        cfg.data.tokenized_path = str(Path(tokenized_path).resolve())
        resolved.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
        provisional.unlink()
        metadata = Path(tokenized_path) / "metadata.json"
        write_json(
            peer_dir / "input-manifest.json",
            {
                "dataset": shard,
                "model": {
                    "id": "Qwen/Qwen3-0.6B",
                    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                    "snapshot_path": str(model_path),
                    "license": "apache-2.0",
                },
                "resolved_config": str(resolved),
                "resolved_config_sha256": sha256_file(resolved),
                "tokenized_path": str(Path(tokenized_path).resolve()),
                "tokenized_metadata_sha256": sha256_file(metadata),
            },
        )

    write_json(
        artifact_root / "data-manifest.json",
        {
            "schema_version": 1,
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
                "license": "cc-by-sa-3.0",
            },
            "partition_type": "disjoint-category-non-iid",
            "train_rows_per_peer": TRAIN_ROWS,
            "validation_rows_per_peer": VALIDATION_ROWS,
            "peers": shards,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset_revision": DATASET_REVISION,
                "peers": {
                    peer: {
                        "categories": shard["categories"],
                        "train_rows": shard["train"]["rows"],
                        "train_sha256": shard["train"]["sha256"],
                        "validation_rows": shard["validation"]["rows"],
                        "validation_sha256": shard["validation"]["sha256"],
                    }
                    for peer, shard in shards.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
