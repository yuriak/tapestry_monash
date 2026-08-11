#!/usr/bin/env python3
"""Materialize and tokenize exactly one Phase 8 site's non-IID Dolly shard."""
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
from phase8.protocol import SITES  # noqa: E402


DATASET_ID = "databricks/databricks-dolly-15k"
DATASET_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DATASET_SPLIT = "train"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TRAIN_ROWS = 1152
VALIDATION_ROWS = 128
SITE_CATEGORIES = {
    "site-a": {"closed_qa", "open_qa", "information_extraction"},
    "site-b": {"brainstorming", "creative_writing", "summarization"},
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


def materialize_site(site: str, data_root: Path) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    categories = SITE_CATEGORIES[site]
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, raw in enumerate(dataset):
        row = dict(raw)
        if str(row.get("category") or "") in categories:
            selected.append((index, row))
            if len(selected) == TRAIN_ROWS + VALIDATION_ROWS:
                break
    if len(selected) != TRAIN_ROWS + VALIDATION_ROWS:
        raise RuntimeError(f"{site} supplied {len(selected)} rows, expected 1280")

    data_root.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": selected[:TRAIN_ROWS],
        "validation": selected[TRAIN_ROWS:],
    }
    split_manifests: dict[str, Any] = {}
    for split, rows in splits.items():
        path = data_root / f"{site}-{split}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for _, row in rows:
                handle.write(json.dumps(chat_record(row), ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
        split_manifests[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "source_indices": [index for index, _ in rows],
            "sha256": sha256_file(path),
        }
    counts: dict[str, int] = {}
    for _, row in selected:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    return {
        "categories": sorted(categories),
        "category_counts": counts,
        **split_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=SITES, required=True)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/phase7/client_ten_epochs.yaml",
    )
    args = parser.parse_args()

    site_root = args.site_root.resolve()
    bootstrap = site_root / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    data_root = site_root / "private-data"
    shard = materialize_site(args.site, data_root)
    model_path = download_model()

    provisional = bootstrap / "provisional-config.yaml"
    resolved = bootstrap / "resolved-config.yaml"
    tokenized_root = site_root / "private-tokenized"
    resolve_template(
        args.template.resolve(),
        provisional,
        model_path,
        Path(shard["train"]["path"]),
        tokenized_root,
        bootstrap / "unused-checkpoints",
        f"{site_root.name}-{args.site}-bootstrap",
        None,
    )
    tokenized_path = tokenize(
        provisional,
        bootstrap / "ray-tokenize",
        f"phase8_{DATASET_REVISION[:12]}_{args.site}_{TRAIN_ROWS}",
    )
    from bhaskera.config import load_config

    cfg = load_config(str(provisional))
    cfg.data.tokenized_path = str(Path(tokenized_path).resolve())
    resolved.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    provisional.unlink()
    metadata = Path(tokenized_path) / "metadata.json"

    manifest = {
        "schema_version": 1,
        "site": args.site,
        "partition_type": "disjoint-category-non-iid",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "license": "cc-by-sa-3.0",
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_path": str(model_path),
            "license": "apache-2.0",
        },
        "shard": shard,
        "resolved_config": str(resolved),
        "resolved_config_sha256": sha256_file(resolved),
        "tokenized_path": str(Path(tokenized_path).resolve()),
        "tokenized_metadata_sha256": sha256_file(metadata),
    }
    write_json(site_root / "site-manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "site": args.site,
                "categories": shard["categories"],
                "train_rows": shard["train"]["rows"],
                "train_sha256": shard["train"]["sha256"],
                "validation_rows": shard["validation"]["rows"],
                "validation_sha256": shard["validation"]["sha256"],
                "resolved_config": str(resolved),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
