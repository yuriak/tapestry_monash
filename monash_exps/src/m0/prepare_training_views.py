#!/usr/bin/env python3
"""Build deterministic M0 training views from the new continent splits only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TRANSFORM_VERSION = "m0-continent-views-v1"

EXPECTED_PARTITIONS = (
    "Central_Europe.jsonl",
    "East_Asia.jsonl",
    "Eastern_Europe.jsonl",
    "Latin_America.jsonl",
    "Mediterranean_Europe.jsonl",
    "Middle_East.jsonl",
    "North_America.jsonl",
    "Northern_Europe_Nordic.jsonl",
    "Oceania.jsonl",
    "South_Asia.jsonl",
    "South_East_Asia.jsonl",
    "Unmatched_or_Other.jsonl",
    "Western_Europe.jsonl",
)

SOURCE_ORDER = (
    "South_Asia.jsonl",
    "Oceania.jsonl",
    "North_America.jsonl",
    "Western_Europe.jsonl",
    "Unmatched_or_Other.jsonl",
)

AUSTRALIA_NZ = frozenset(("Australia", "New Zealand"))
NORTH_AMERICA = frozenset(("United States of America", "Canada"))
UNITED_KINGDOM = frozenset(
    ("United Kingdom of Great Britain and Northern Ireland",)
)
NOT_MATCHED = frozenset(("NOT MATCHED",))


def component(source: str, countries: frozenset[str] | None = None) -> dict[str, Any]:
    return {"source": source, "countries": countries}


VIEW_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "south_asia": (component("South_Asia.jsonl"),),
    "australia_nz": (component("Oceania.jsonl", AUSTRALIA_NZ),),
    "australia_nz_western_europe": (
        component("Oceania.jsonl", AUSTRALIA_NZ),
        component("Western_Europe.jsonl"),
    ),
    "australia_nz_us_canada_uk": (
        component("Oceania.jsonl", AUSTRALIA_NZ),
        component("North_America.jsonl", NORTH_AMERICA),
        component("Western_Europe.jsonl", UNITED_KINGDOM),
    ),
    "not_matched": (
        component("Unmatched_or_Other.jsonl", NOT_MATCHED),
    ),
    "centralized_variant_1": (
        component("South_Asia.jsonl"),
        component("Oceania.jsonl", AUSTRALIA_NZ),
    ),
    "centralized_variant_2": (
        component("South_Asia.jsonl"),
        component("Oceania.jsonl", AUSTRALIA_NZ),
        component("Western_Europe.jsonl"),
    ),
    "centralized_variant_3": (
        component("South_Asia.jsonl"),
        component("Oceania.jsonl", AUSTRALIA_NZ),
        component("North_America.jsonl", NORTH_AMERICA),
        component("Western_Europe.jsonl", UNITED_KINGDOM),
    ),
}


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


def canonical_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_row(row: Any, path: Path, line_number: int) -> dict[str, Any]:
    where = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise TypeError(f"{where}: expected a JSON object")
    missing = {"country", "doc_id", "messages"}.difference(row)
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    if not isinstance(row["country"], str) or not row["country"].strip():
        raise ValueError(f"{where}: country must be a non-empty string")
    if not isinstance(row["doc_id"], str) or not row["doc_id"].strip():
        raise ValueError(f"{where}: doc_id must be a non-empty string")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{where}: expected exactly three messages")
    roles = tuple(message.get("role") if isinstance(message, dict) else None for message in messages)
    if roles != ("system", "user", "assistant"):
        raise ValueError(f"{where}: unexpected message roles {roles}")
    for index, message in enumerate(messages):
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{where}: message {index} has empty/non-string content")
    return row


def iter_rows(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            row = validate_row(row, path, line_number)
            yield line_number, row, canonical_line(row)


def audit_sources(source_root: Path, workspace_root: Path) -> dict[str, Any]:
    observed = tuple(sorted(path.name for path in source_root.glob("*.jsonl")))
    expected = tuple(sorted(EXPECTED_PARTITIONS))
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise RuntimeError(f"continent source inventory changed; missing={missing}, extra={extra}")

    partitions: dict[str, Any] = {}
    row_locations: dict[str, str] = {}
    cross_partition_exact_duplicates = 0
    cross_partition_pairs: Counter[tuple[str, str]] = Counter()
    for name in EXPECTED_PARTITIONS:
        path = source_root / name
        countries: Counter[str] = Counter()
        doc_ids: Counter[str] = Counter()
        row_hashes: Counter[str] = Counter()
        row_count = 0
        for line_number, row, normalized in iter_rows(path):
            row_count += 1
            countries[row["country"]] += 1
            doc_ids[row["doc_id"]] += 1
            row_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            row_hashes[row_hash] += 1
            previous = row_locations.setdefault(row_hash, f"{name}:{line_number}")
            if not previous.startswith(f"{name}:"):
                cross_partition_exact_duplicates += 1
                previous_name = previous.rsplit(":", 1)[0]
                cross_partition_pairs[tuple(sorted((previous_name, name)))] += 1
        partitions[name] = {
            "path": portable(path, workspace_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": row_count,
            "countries": dict(sorted(countries.items())),
            "unique_doc_ids": len(doc_ids),
            "repeated_doc_id_rows": sum(count - 1 for count in doc_ids.values()),
            "unique_exact_rows": len(row_hashes),
            "repeated_exact_rows": sum(count - 1 for count in row_hashes.values()),
        }

    return {
        "authority": "new_data_team_continent_splits_only",
        "source_root": portable(source_root, workspace_root),
        "partitions": partitions,
        "cross_partition_repeated_exact_rows": cross_partition_exact_duplicates,
        "cross_partition_repeated_exact_row_pairs": {
            " <-> ".join(pair): count
            for pair, count in sorted(cross_partition_pairs.items())
        },
        "notes": [
            "Repeated doc_id values are expected because one source document may yield multiple instructions.",
            "Exact repetitions supplied across region files are recorded but not removed.",
            "No older CultureInstruct exports are read or compared by this transform.",
        ],
    }


def selected(row: dict[str, Any], countries: frozenset[str] | None) -> bool:
    return countries is None or row["country"] in countries


def build_views(
    source_root: Path,
    staging_root: Path,
    final_root: Path,
    workspace_root: Path,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    manifests_dir = staging_root / "manifests"
    manifests_dir.mkdir(parents=True)
    view_stats: dict[str, dict[str, Any]] = {}

    with ExitStack() as stack:
        handles = {}
        for view_name, specs in VIEW_SPECS.items():
            view_dir = staging_root / view_name
            view_dir.mkdir(parents=True)
            handles[view_name] = stack.enter_context(
                (view_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n")
            )
            view_stats[view_name] = {
                "rows": 0,
                "countries": Counter(),
                "components": [],
                "specs": specs,
            }

        views_by_source: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for view_name, specs in VIEW_SPECS.items():
            for spec in specs:
                views_by_source[spec["source"]].append((view_name, spec))

        for source_name in SOURCE_ORDER:
            source_path = source_root / source_name
            starts = {
                view_name: view_stats[view_name]["rows"]
                for view_name, _ in views_by_source[source_name]
            }
            selected_counts: Counter[str] = Counter()
            for _, row, normalized in iter_rows(source_path):
                for view_name, spec in views_by_source[source_name]:
                    if not selected(row, spec["countries"]):
                        continue
                    handles[view_name].write(normalized + "\n")
                    view_stats[view_name]["rows"] += 1
                    view_stats[view_name]["countries"][row["country"]] += 1
                    selected_counts[view_name] += 1
            for view_name, spec in views_by_source[source_name]:
                count = selected_counts[view_name]
                if count == 0:
                    raise RuntimeError(f"selection produced zero rows: {view_name}/{source_name}")
                view_stats[view_name]["components"].append(
                    {
                        "source": portable(source_path, workspace_root),
                        "source_sha256": source_audit["partitions"][source_name]["sha256"],
                        "selection": (
                            {"country_in": sorted(spec["countries"])}
                            if spec["countries"] is not None
                            else {"all_rows": True}
                        ),
                        "output_row_start": starts[view_name],
                        "output_row_end_exclusive": starts[view_name] + count,
                        "rows": count,
                    }
                )

    views_manifest: dict[str, Any] = {}
    for view_name, stats in view_stats.items():
        staged_path = staging_root / view_name / "train.jsonl"
        final_path = final_root / view_name / "train.jsonl"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "transform_version": TRANSFORM_VERSION,
            "view": view_name,
            "path": portable(final_path, workspace_root),
            "bytes": staged_path.stat().st_size,
            "sha256": sha256_file(staged_path),
            "rows": stats["rows"],
            "countries": dict(sorted(stats["countries"].items())),
            "components": stats["components"],
            "deduplication": "none",
            "output_schema": ["country", "doc_id", "messages"],
        }
        manifest_path = manifests_dir / f"{view_name}.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        views_manifest[view_name] = {
            **manifest,
            "manifest": portable(final_root / "manifests" / manifest_path.name, workspace_root),
        }
    return views_manifest


def resolve_portable(path: str, workspace_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else workspace_root / candidate


def verify_existing(output_root: Path, workspace_root: Path) -> bool:
    master_path = output_root / "manifests" / "prepared-data.json"
    if not master_path.is_file():
        return False
    manifest = json.loads(master_path.read_text(encoding="utf-8"))
    if manifest.get("transform_version") != TRANSFORM_VERSION:
        return False
    for source in manifest["source_audit"]["partitions"].values():
        path = resolve_portable(source["path"], workspace_root)
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            return False
    for view in manifest["views"].values():
        path = resolve_portable(view["path"], workspace_root)
        if not path.is_file() or path.stat().st_size != view["bytes"]:
            return False
        if sha256_file(path) != view["sha256"]:
            return False
    print(f"Existing prepared views verified: {master_path}")
    for name, view in sorted(manifest["views"].items()):
        print(f"  {name}: {view['rows']} rows")
    return True


def main() -> int:
    inferred_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=inferred_workspace / "local_data" / "m0_incoming" / "continent_splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=inferred_workspace / "monash_exps" / ".runtime" / "data" / "m0" / "prepared",
    )
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    workspace_root = args.workspace_root.resolve()
    if not source_root.is_dir():
        parser.error(f"missing continent source directory: {source_root}")

    if output_root.exists():
        if verify_existing(output_root, workspace_root):
            print("M0 PREPARED DATA VERIFIED")
            return 0
        raise RuntimeError(
            f"existing prepared root is incomplete or stale: {output_root}; "
            "preserve it for review and choose a new --output-root"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".prepared-staging-", dir=output_root.parent)
    )
    try:
        print("Auditing authoritative continent partitions...")
        source_audit = audit_sources(source_root, workspace_root)
        audited_rows = sum(
            partition["rows"] for partition in source_audit["partitions"].values()
        )
        print(
            f"  {len(source_audit['partitions'])} partitions, {audited_rows} rows, "
            f"{source_audit['cross_partition_repeated_exact_rows']} supplied "
            "cross-partition exact repetitions"
        )
        print("Building deterministic training views without deduplication...")
        views = build_views(
            source_root,
            staging_root,
            output_root,
            workspace_root,
            source_audit,
        )
        master = {
            "schema_version": SCHEMA_VERSION,
            "transform_version": TRANSFORM_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authority": "new_data_team_continent_splits_only",
            "transformation_command": [
                "python",
                "monash_exps/src/m0/prepare_training_views.py",
                "--source-root",
                portable(source_root, workspace_root),
                "--output-root",
                portable(output_root, workspace_root),
            ],
            "source_audit": source_audit,
            "views": views,
        }
        master_path = staging_root / "manifests" / "prepared-data.json"
        master_path.write_text(
            json.dumps(master, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_root, output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"Prepared root: {output_root}")
    for name, view in sorted(views.items()):
        print(f"  {name}: {view['rows']} rows, sha256={view['sha256']}")
    print("M0 PREPARED DATA PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
