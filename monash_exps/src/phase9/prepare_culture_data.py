#!/usr/bin/env python3
"""Build deterministic Phase 9 Alpaca JSONL datasets from cultureInstruct."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

SCHEMA_VERSION = 1
TRANSFORM_VERSION = "culture-instruct-alpaca-olmo-v1"
DEFAULT_MODEL_ID = "allenai/OLMo-1B-hf"
DEFAULT_MODEL_REVISION = "aee7752d9c08ee4775e9b0091426d8410e8f6a89"
DEFAULT_COUNTRIES = ("Australia", "India")

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        " ".join(line.split())
        for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ).strip()


def compact_metadata(value: Any, max_chars: int = 240) -> str:
    text = normalized_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def normalized_url(value: Any) -> str:
    url = normalized_text(value)
    if not url:
        return ""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def iter_leaf_records(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        if "tit" in value and "summ" in value:
            yield path, value
            return
        for key, child in value.items():
            yield from iter_leaf_records(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_leaf_records(child, path + (str(index),))


def category_for(path: tuple[str, ...], row: dict[str, Any]) -> str:
    explicit = compact_metadata(row.get("categorization_result"))
    if explicit:
        return explicit
    for part in path:
        if not part.isdigit():
            return compact_metadata(part)
    return "Uncategorized"


def source_record(country: str, path: tuple[str, ...], row: dict[str, Any]) -> dict[str, Any] | None:
    title = normalized_text(row.get("tit"))
    summary = normalized_text(row.get("summ"))
    if not title or not summary:
        return None

    url = normalized_url(row.get("url"))
    body_value = row.get("pg_content")
    if isinstance(body_value, str):
        body = body_value.replace("\r\n", "\n").replace("\r", "\n").strip()
    elif body_value is None:
        body = ""
    else:
        body = json.dumps(body_value, ensure_ascii=False, sort_keys=True)
    body_sha256 = sha256_text(body) if body else ""
    source_fingerprint = sha256_text(
        json.dumps(
            {
                "country": country,
                "path": path,
                "title": title,
                "url": url,
                "summary_sha256": sha256_text(summary),
                "body_sha256": body_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return {
        "country": country,
        "title": title,
        "summary": summary,
        "summary_sha256": sha256_text(summary),
        "url": url,
        "body_sha256": body_sha256,
        "category": category_for(path, row),
        "geo_subregion": compact_metadata(row.get("geo_subregion")),
        "ethnoling_group": compact_metadata(row.get("ethnoling_group")),
        "source_path": "/".join(path),
        "source_fingerprint": source_fingerprint,
    }


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Sorting before selection makes duplicate resolution independent of JSON
    # traversal or dict insertion order.
    ordered = sorted(records, key=lambda row: row["source_fingerprint"])
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()
    seen_fallbacks: set[str] = set()
    kept: list[dict[str, Any]] = []
    counts = {"duplicate_url": 0, "duplicate_body": 0, "duplicate_fallback": 0}

    for row in ordered:
        url = row["url"]
        body_sha256 = row["body_sha256"]
        fallback = sha256_text(f"{row['title']}\0{row['summary_sha256']}")
        if url and url in seen_urls:
            counts["duplicate_url"] += 1
            continue
        if body_sha256 and body_sha256 in seen_bodies:
            counts["duplicate_body"] += 1
            continue
        if not url and not body_sha256 and fallback in seen_fallbacks:
            counts["duplicate_fallback"] += 1
            continue
        if url:
            seen_urls.add(url)
        if body_sha256:
            seen_bodies.add(body_sha256)
        seen_fallbacks.add(fallback)
        kept.append(row)
    return kept, counts


def build_instruction(row: dict[str, Any]) -> tuple[str, str]:
    instruction = (
        f"Write a concise, factual overview of the following cultural topic "
        f"from {row['country']}."
    )
    fields = [
        ("Country", row["country"]),
        ("Topic", row["title"]),
        ("Category", row["category"]),
        ("Geographic subregion", row["geo_subregion"]),
        ("Ethnolinguistic group", row["ethnoling_group"]),
    ]
    input_text = "\n".join(f"{name}: {value}" for name, value in fields if value)
    return instruction, input_text


def render_alpaca(instruction: str, input_text: str, output: str) -> str:
    return ALPACA_WITH_INPUT.format(
        instruction=instruction,
        input=input_text,
        output=output,
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def fit_output(tokenizer: Any, instruction: str, input_text: str, output: str, seq_len: int) -> tuple[str, int, bool]:
    rendered = render_alpaca(instruction, input_text, output)
    rendered_tokens = token_count(tokenizer, rendered)
    if rendered_tokens <= seq_len:
        return output, rendered_tokens, False

    output_ids = tokenizer(output, add_special_tokens=False, truncation=False)["input_ids"]
    low, high = 0, len(output_ids)
    best_text = ""
    best_tokens = 0
    while low <= high:
        midpoint = (low + high) // 2
        candidate = tokenizer.decode(
            output_ids[:midpoint],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        candidate_tokens = token_count(
            tokenizer, render_alpaca(instruction, input_text, candidate)
        )
        if candidate_tokens <= seq_len:
            best_text = candidate
            best_tokens = candidate_tokens
            low = midpoint + 1
        else:
            high = midpoint - 1

    if not best_text:
        raise RuntimeError(
            f"Alpaca prompt leaves no output tokens at seq_len={seq_len}: "
            f"instruction={instruction!r}, input={input_text!r}"
        )
    return best_text, best_tokens, True


def make_training_record(tokenizer: Any, source: dict[str, Any], seq_len: int) -> tuple[dict[str, Any], dict[str, Any]]:
    instruction, input_text = build_instruction(source)
    output, rendered_tokens, truncated = fit_output(
        tokenizer, instruction, input_text, source["summary"], seq_len
    )
    output_tokens = len(
        tokenizer(output, add_special_tokens=False, truncation=False)["input_ids"]
    )
    source_id = sha256_text(
        f"{source['country']}\0{source['url']}\0{source['title']}\0{source['summary_sha256']}"
    )
    record = {
        "category": source["category"],
        "country": source["country"],
        "input": input_text,
        "instruction": instruction,
        "output": output,
        "output_truncated": truncated,
        "source_body_sha256": source["body_sha256"],
        "source_id": source_id,
        "source_summary_sha256": source["summary_sha256"],
        "source_url": source["url"],
        "title": source["title"],
    }
    audit = {
        "category": source["category"],
        "output_tokens": output_tokens,
        "output_truncated": truncated,
        "rendered_tokens": rendered_tokens,
        "source_id": source_id,
    }
    return record, audit


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def token_statistics(audits: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = [int(row["rendered_tokens"]) for row in audits]
    outputs = [int(row["output_tokens"]) for row in audits]
    return {
        "output_tokens": {
            "min": min(outputs),
            "p50": percentile(outputs, 0.50),
            "p95": percentile(outputs, 0.95),
            "max": max(outputs),
        },
        "rendered_tokens": {
            "min": min(rendered),
            "p50": percentile(rendered, 0.50),
            "p95": percentile(rendered, 0.95),
            "max": max(rendered),
        },
        "truncated_outputs": sum(bool(row["output_truncated"]) for row in audits),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
    return count, sha256_file(path)


def file_entry(root: Path, path: Path, rows: int, digest: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def build_country(
    *,
    country: str,
    source_path: Path,
    output_root: Path,
    tokenizer: Any,
    seq_len: int,
    validation_fraction: float,
    smoke_train_rows: int,
    smoke_validation_rows: int,
) -> dict[str, Any]:
    print(f"Loading {country}: {source_path} ({source_path.stat().st_size:,} bytes)", flush=True)
    source_sha256 = sha256_file(source_path)
    with source_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    raw_records: list[dict[str, Any]] = []
    missing_required = 0
    for path, raw in iter_leaf_records(document):
        prepared = source_record(country, path, raw)
        if prepared is None:
            missing_required += 1
        else:
            raw_records.append(prepared)
    raw_leaf_rows = len(raw_records) + missing_required
    del document
    gc.collect()

    sources, duplicate_counts = deduplicate(raw_records)
    del raw_records
    gc.collect()

    records_and_audits = [
        make_training_record(tokenizer, source, seq_len) for source in sources
    ]
    training_records = [pair[0] for pair in records_and_audits]
    audits = [pair[1] for pair in records_and_audits]
    del records_and_audits, sources
    gc.collect()

    # Stable hash order gives a deterministic, category-mixed split.
    order = sorted(
        range(len(training_records)),
        key=lambda index: sha256_text(
            f"phase9-split-v1\0{country}\0{training_records[index]['source_id']}"
        ),
    )
    validation_rows = max(1, round(len(order) * validation_fraction))
    validation_indices = order[:validation_rows]
    train_indices = order[validation_rows:]
    if len(train_indices) < smoke_train_rows:
        raise RuntimeError(
            f"{country} has {len(train_indices)} training rows, fewer than smoke size {smoke_train_rows}"
        )
    if len(validation_indices) < smoke_validation_rows:
        raise RuntimeError(
            f"{country} has {len(validation_indices)} validation rows, fewer than smoke size {smoke_validation_rows}"
        )

    country_dir = output_root / country.lower().replace(" ", "-")
    full_train = country_dir / "full" / "train.jsonl"
    full_validation = country_dir / "full" / "validation.jsonl"
    smoke_train = country_dir / "smoke-1k" / "train.jsonl"
    smoke_validation = country_dir / "smoke-1k" / "validation.jsonl"

    output_specs = {
        "full_train": (full_train, train_indices),
        "full_validation": (full_validation, validation_indices),
        "smoke_train": (smoke_train, train_indices[:smoke_train_rows]),
        "smoke_validation": (
            smoke_validation,
            validation_indices[:smoke_validation_rows],
        ),
    }
    files: dict[str, Any] = {}
    for name, (path, indices) in output_specs.items():
        rows, digest = write_jsonl(path, (training_records[index] for index in indices))
        files[name] = file_entry(output_root, path, rows, digest)

    category_counts = Counter(row["category"] for row in training_records)
    manifest = {
        "country": country,
        "source": {
            "path": source_path.name,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
            "raw_leaf_rows": raw_leaf_rows,
            "missing_required_rows": missing_required,
            "duplicate_rows": sum(duplicate_counts.values()),
            "duplicate_reasons": duplicate_counts,
            "deduplicated_rows": len(training_records),
        },
        "split": {
            "method": "sha256-rank-phase9-split-v1",
            "validation_fraction": validation_fraction,
            "full_train_rows": len(train_indices),
            "full_validation_rows": len(validation_indices),
            "smoke_train_rows": smoke_train_rows,
            "smoke_validation_rows": smoke_validation_rows,
        },
        "categories": dict(sorted(category_counts.items())),
        "token_statistics": token_statistics(audits),
        "files": files,
    }
    country_manifest_path = country_dir / "manifest.json"
    country_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {country}: raw={raw_leaf_rows}, deduplicated={len(training_records)}, "
        f"train={len(train_indices)}, validation={len(validation_indices)}, "
        f"truncated={manifest['token_statistics']['truncated_outputs']}",
        flush=True,
    )
    return manifest


def verify_reusable(
    output_root: Path,
    expected: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any] | None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    country_manifests = manifest.get("countries", {})
    for country, source_path in source_paths.items():
        country_manifest = country_manifests.get(country)
        if country_manifest is None:
            return None
        source_entry = country_manifest.get("source", {})
        if (
            source_entry.get("bytes") != source_path.stat().st_size
            or source_entry.get("sha256") != sha256_file(source_path)
        ):
            return None
        for entry in country_manifest.get("files", {}).values():
            path = output_root / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                return None
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--smoke-train-rows", type=int, default=1000)
    parser.add_argument("--smoke-validation-rows", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be between 0 and 0.5")
    if args.seq_len < 128:
        raise SystemExit("--seq-len must be at least 128")
    if args.smoke_train_rows < 1 or args.smoke_validation_rows < 1:
        raise SystemExit("smoke split sizes must be positive")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_paths = {country: source_root / f"{country}.json" for country in args.countries}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing cultureInstruct source files: " + ", ".join(missing))

    identity = {
        "schema_version": SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "model": {"id": args.model_id, "revision": args.model_revision},
        "sequence_length": args.seq_len,
        "countries_requested": list(args.countries),
        "validation_fraction": args.validation_fraction,
        "smoke_train_rows": args.smoke_train_rows,
        "smoke_validation_rows": args.smoke_validation_rows,
    }
    reusable = (
        verify_reusable(output_root, identity, source_paths)
        if output_root.exists()
        else None
    )
    if reusable is not None and not args.overwrite:
        print(json.dumps({"status": "REUSED", "manifest": str(output_root / "manifest.json")}, indent=2))
        return
    if output_root.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists but is incomplete or does not match this recipe: {output_root}. "
            "Review it, then rerun with --overwrite to preserve it as a timestamped backup."
        )

    from transformers import AutoTokenizer

    print(
        f"Loading tokenizer {args.model_id}@{args.model_revision}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=False,
        cache_dir=os.environ.get("HF_HOME"),
    )

    stage = output_root.with_name(f".{output_root.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        country_manifests = {
            country: build_country(
                country=country,
                source_path=source_paths[country],
                output_root=stage,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                validation_fraction=args.validation_fraction,
                smoke_train_rows=args.smoke_train_rows,
                smoke_validation_rows=args.smoke_validation_rows,
            )
            for country in args.countries
        }
        zip_path = source_root.parent.parent / "culture-instruct.zip"
        manifest = {
            **identity,
            "dataset": {
                "name": "cultureInstruct",
                "source_layout": "country-specific nested Wikipedia-derived JSON",
                "archive": (
                    {
                        "name": zip_path.name,
                        "bytes": zip_path.stat().st_size,
                        "sha256": sha256_file(zip_path),
                    }
                    if zip_path.is_file()
                    else None
                ),
            },
            "format": {
                "name": "alpaca",
                "use_chat_template": False,
                "instruction_contract": "country/topic/category metadata to factual overview",
                "output_source": "summ",
                "truncation": "token-boundary response truncation preserving rendered length <= sequence_length",
            },
            "countries": country_manifests,
            "tokenizer": {
                "class": tokenizer.__class__.__name__,
                "name_or_path": tokenizer.name_or_path,
                "vocab_size": len(tokenizer),
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if output_root.exists():
            backup = output_root.with_name(
                f"{output_root.name}.previous-{time.strftime('%Y%m%dT%H%M%S')}"
            )
            output_root.rename(backup)
            print(f"Preserved previous output at {backup}")
        stage.rename(output_root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    summary = {
        "status": "PASS",
        "manifest": str(output_root / "manifest.json"),
        "countries": {
            country: {
                "deduplicated_rows": data["source"]["deduplicated_rows"],
                "full_train_rows": data["split"]["full_train_rows"],
                "full_validation_rows": data["split"]["full_validation_rows"],
                "smoke_train_rows": data["split"]["smoke_train_rows"],
                "smoke_validation_rows": data["split"]["smoke_validation_rows"],
                "truncated_outputs": data["token_statistics"]["truncated_outputs"],
            }
            for country, data in country_manifests.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
