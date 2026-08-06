#!/usr/bin/env python3
"""Download pinned public inputs, tokenize them, and resolve a Phase 1 config."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import yaml


DATASET_ID = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
DATASET_REVISION = "14f543216b9ba42b6b951dc5bd199460d193b162"
DATASET_SPLIT = "train_sft"
DATASET_ROWS = 8
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocated_cpus() -> int:
    explicit_limit = os.environ.get("SLAKSHNA_CPU_LIMIT", "")
    if explicit_limit.isdigit() and int(explicit_limit) > 0:
        return int(explicit_limit)
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name, "").split("(", 1)[0]
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return len(os.sched_getaffinity(0))


def materialize_dataset(
    data_root: Path, row_indices: list[int] | None = None
) -> tuple[Path, dict]:
    from datasets import load_dataset

    data_root.mkdir(parents=True, exist_ok=True)
    selected_indices = list(range(DATASET_ROWS)) if row_indices is None else row_indices
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise RuntimeError(f"dataset row indices must be non-empty and unique: {selected_indices}")
    if min(selected_indices) < 0 or max(selected_indices) >= DATASET_ROWS:
        raise RuntimeError(
            f"dataset row indices must be within [0, {DATASET_ROWS}): {selected_indices}"
        )
    if row_indices is None:
        output = data_root / f"everyday-conversations-{DATASET_REVISION[:12]}-{DATASET_ROWS}.jsonl"
        selection_description = f"select(range({DATASET_ROWS}))"
    else:
        selection_label = "-".join(str(index) for index in selected_indices)
        output = data_root / (
            f"everyday-conversations-{DATASET_REVISION[:12]}-rows-{selection_label}.jsonl"
        )
        selection_description = f"select({selected_indices})"
    if not output.is_file():
        temporary = output.with_suffix(".jsonl.tmp")
        if row_indices is not None:
            # Phase 4 shards derive from the already pinned/materialized Phase 1
            # source. This avoids both a second remote query and any chance that
            # two clients observe different upstream dataset state.
            full_path, _ = materialize_dataset(data_root, None)
            full_lines = full_path.read_text(encoding="utf-8").splitlines()
            with temporary.open("w", encoding="utf-8") as handle:
                for index in selected_indices:
                    handle.write(full_lines[index] + "\n")
        else:
            dataset = load_dataset(
                DATASET_ID,
                split=DATASET_SPLIT,
                revision=DATASET_REVISION,
                cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            )
            if len(dataset) < DATASET_ROWS:
                raise RuntimeError(f"dataset has only {len(dataset)} rows")
            selected = dataset.select(selected_indices)
            with temporary.open("w", encoding="utf-8") as handle:
                for row in selected:
                    messages = row.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise RuntimeError("dataset row has no messages")
                    record = {"messages": messages}
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(output)

    lines = output.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(selected_indices):
        raise RuntimeError(
            f"expected {len(selected_indices)} materialized rows, got {len(lines)}"
        )
    return output.resolve(), {
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "selected_rows": len(selected_indices),
        "selected_indices": selected_indices,
        "selection": selection_description,
        "license": "apache-2.0",
        "materialized_sha256": sha256_file(output),
    }


def download_model() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(Path(os.environ["HF_HOME"]) / "hub"),
    )
    path = Path(snapshot).resolve()
    if not (path / "config.json").is_file():
        raise RuntimeError(f"model snapshot is incomplete: {path}")
    return path


def resolve_template(
    template: Path,
    output: Path,
    model_path: Path,
    train_path: Path,
    tokenized_root: Path,
    checkpoint_dir: Path,
    run_name: str,
    lora_resume_path: Path | None = None,
) -> None:
    text = template.read_text(encoding="utf-8")
    replacements = {
        "__MODEL_PATH__": str(model_path),
        "__TRAIN_PATH__": str(train_path),
        "__TOKENIZED_ROOT__": str(tokenized_root),
        "__CHECKPOINT_DIR__": str(checkpoint_dir),
        "__RUN_NAME__": run_name,
        "__RESUME_PATH__": str(lora_resume_path) if lora_resume_path else "",
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    unresolved = [key for key in replacements if key in text]
    if unresolved:
        raise RuntimeError(f"unresolved config markers: {unresolved}")
    output.write_text(text, encoding="utf-8")


def tokenize(config_path: Path, ray_root: Path, cache_name: str = "phase1_smoke_8") -> str:
    import ray
    import ray.data
    from bhaskera.config import load_config
    from bhaskera.data.datasets.local_chat import _build_raw
    from bhaskera.data.tokenize import persist_tokenized

    cfg = load_config(str(config_path))
    cpu_count = max(3, min(allocated_cpus(), 8))
    # Ray's Unix-domain sockets have a 107-byte path limit. Artifact paths are
    # intentionally descriptive and can exceed it, so runtime sockets belong
    # in the allocation's short-lived node-local temporary directory.
    temp_base = Path(os.environ.get("TMPDIR", "/tmp"))
    short_ray_root = temp_base / f"slakshna-p1-tokenize-{os.getpid()}"
    short_ray_root.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=cpu_count, include_dashboard=False, _temp_dir=str(short_ray_root.resolve()))
    try:
        raw = _build_raw(cfg, split="train")
        return persist_tokenized(raw, cfg, "text", cache_name)
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--lora-resume-path", type=Path)
    parser.add_argument(
        "--row-indices",
        help="comma-separated deterministic subset of the first eight source rows",
    )
    parser.add_argument(
        "--data-label",
        help="filesystem-safe tokenization cache label; required with --row-indices",
    )
    args = parser.parse_args()

    experiment_root = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = experiment_root / ".runtime" / "data" / "phase1"
    row_indices = None
    if args.row_indices:
        try:
            row_indices = [int(value) for value in args.row_indices.split(",")]
        except ValueError as exc:
            raise SystemExit(f"invalid --row-indices: {args.row_indices}") from exc
        if not args.data_label:
            raise SystemExit("--data-label is required with --row-indices")
    if args.data_label and not all(
        character.isalnum() or character in {"-", "_"} for character in args.data_label
    ):
        raise SystemExit("--data-label may contain only letters, digits, '-' and '_'")
    tokenized_root = data_root / "tokenized"
    if args.data_label:
        tokenized_root = tokenized_root / args.data_label

    train_path, dataset_manifest = materialize_dataset(data_root / "source", row_indices)
    model_path = download_model()
    provisional = run_dir / "provisional-config.yaml"
    resolved = run_dir / "resolved-config.yaml"
    resolve_template(
        args.template.resolve(), provisional, model_path, train_path,
        tokenized_root.resolve(), args.checkpoint_dir.resolve(), args.run_name,
        args.lora_resume_path.resolve() if args.lora_resume_path else None,
    )
    cache_name = f"phase1_{args.data_label}" if args.data_label else "phase1_smoke_8"
    tokenized_path = tokenize(provisional, run_dir / "ray-tokenize", cache_name)

    from bhaskera.config import load_config

    cfg = load_config(str(provisional))
    cfg.data.tokenized_path = str(Path(tokenized_path).resolve())
    resolved.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    provisional.unlink()

    cache_metadata = Path(tokenized_path) / "metadata.json"
    manifest = {
        "dataset": dataset_manifest,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "apache-2.0",
            "snapshot_path": str(model_path),
        },
        "config": str(resolved),
        "config_sha256": sha256_file(resolved),
        "tokenized_path": str(Path(tokenized_path).resolve()),
        "tokenized_metadata_sha256": sha256_file(cache_metadata),
    }
    (run_dir / "input-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(resolved)


if __name__ == "__main__":
    main()
