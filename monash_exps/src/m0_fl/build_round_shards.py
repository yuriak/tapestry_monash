#!/usr/bin/env python3
"""Build two deterministic data passes as ten complete DDP round shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SITE_VIEWS = {"au": "australia_nz", "india": "south_asia"}
ROUNDS_PER_PASS = 5
PASSES = 2
EFFECTIVE_BATCH = 16


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_step_counts(total_steps: int) -> list[int]:
    base, extra = divmod(total_steps, ROUNDS_PER_PASS)
    return [base + (index < extra) for index in range(ROUNDS_PER_PASS)]


def split(values: list[int], sizes: list[int]) -> list[list[int]]:
    result = []
    offset = 0
    for size in sizes:
        result.append(values[offset : offset + size])
        offset += size
    if offset != len(values):
        raise RuntimeError(f"split consumed {offset} values, expected {len(values)}")
    return result


def write_site(
    site: str,
    source_dir: Path,
    output_root: Path,
    seed: int,
    force: bool,
) -> dict[str, object]:
    parquet_files = sorted(source_dir.glob("*.parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(f"expected one source parquet for {site}: {parquet_files}")
    source_file = parquet_files[0]
    source = pq.read_table(source_file)
    source_rows = source.num_rows
    if source_rows <= 0:
        raise RuntimeError(f"empty source table: {source_file}")

    site_root = output_root / site
    if site_root.exists():
        if not force:
            raise RuntimeError(f"output already exists (use --force): {site_root}")
        shutil.rmtree(site_root)
    site_root.mkdir(parents=True)

    padded_rows = math.ceil(source_rows / EFFECTIVE_BATCH) * EFFECTIVE_BATCH
    total_steps = padded_rows // EFFECTIVE_BATCH
    step_counts = balanced_step_counts(total_steps)
    row_counts = [steps * EFFECTIVE_BATCH for steps in step_counts]
    rounds: list[dict[str, object]] = []

    for pass_index in range(PASSES):
        permutation = list(range(source_rows))
        random.Random(seed + pass_index).shuffle(permutation)
        padding_count = padded_rows - source_rows
        padding = permutation[:padding_count]
        scheduled = permutation + padding
        chunks = split(scheduled, row_counts)
        originals_seen: list[int] = []

        for within_pass, indices in enumerate(chunks, start=1):
            round_index = pass_index * ROUNDS_PER_PASS + within_pass
            round_dir = site_root / f"round_{round_index:02d}"
            round_dir.mkdir()
            table = source.take(pa.array(indices, type=pa.int64()))
            output_file = round_dir / "data.parquet"
            pq.write_table(table, output_file, compression="zstd")
            if pq.ParquetFile(output_file).metadata.num_rows != len(indices):
                raise RuntimeError(f"row-count mismatch after writing {output_file}")
            if not pq.read_table(output_file).equals(table):
                raise RuntimeError(f"round-trip content mismatch after writing {output_file}")

            # Padding is appended to the complete pass and therefore always
            # belongs to its last shard. Preserve the exact distinction.
            if within_pass == ROUNDS_PER_PASS and padding_count:
                original_indices = indices[:-padding_count]
                round_padding = indices[-padding_count:]
            else:
                original_indices = indices
                round_padding = []
            originals_seen.extend(original_indices)
            rounds.append(
                {
                    "round": round_index,
                    "pass": pass_index + 1,
                    "within_pass": within_pass,
                    "relative_path": f"round_{round_index:02d}",
                    "rows": len(indices),
                    "max_steps": step_counts[within_pass - 1],
                    "original_rows": len(original_indices),
                    "padding_rows": len(round_padding),
                    "padding_source_indices": round_padding,
                    "source_indices": original_indices,
                    "parquet_sha256": sha256_file(output_file),
                    "parquet_bytes": output_file.stat().st_size,
                }
            )

        if len(originals_seen) != source_rows:
            raise RuntimeError(
                f"pass {pass_index + 1} for {site} covered "
                f"{len(originals_seen)} originals, expected {source_rows}"
            )
        if set(originals_seen) != set(range(source_rows)):
            raise RuntimeError(f"pass {pass_index + 1} coverage mismatch for {site}")
        if len(set(originals_seen)) != source_rows:
            raise RuntimeError(f"pass {pass_index + 1} duplicates originals for {site}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "site": site,
        "source_directory": str(source_dir.resolve()),
        "source_parquet": str(source_file.resolve()),
        "source_parquet_sha256": sha256_file(source_file),
        "source_rows": source_rows,
        "effective_global_batch": EFFECTIVE_BATCH,
        "passes": PASSES,
        "rounds_per_pass": ROUNDS_PER_PASS,
        "round_count": PASSES * ROUNDS_PER_PASS,
        "seed": seed,
        "padding_policy": (
            "append the first rows of each deterministic pass permutation "
            "until the pass is divisible by the effective global batch"
        ),
        "padding_rows_per_pass": padded_rows - source_rows,
        "scheduled_rows_per_pass": padded_rows,
        "rounds": rounds,
    }
    manifest_path = site_root / "round-shards-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    reports = {}
    for site, view in SITE_VIEWS.items():
        parent = args.cache_root / view
        candidates = sorted(path for path in parent.iterdir() if path.is_dir())
        if len(candidates) != 1:
            raise RuntimeError(f"expected one token cache for {view}: {candidates}")
        reports[site] = write_site(
            site=site,
            source_dir=candidates[0],
            output_root=args.output_root,
            seed=args.seed + (0 if site == "au" else 10_000),
            force=args.force,
        )

    summary = {
        site: {
            "source_rows": report["source_rows"],
            "scheduled_rows_per_pass": report["scheduled_rows_per_pass"],
            "padding_rows_per_pass": report["padding_rows_per_pass"],
            "round_steps": [item["max_steps"] for item in report["rounds"]],
        }
        for site, report in reports.items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
