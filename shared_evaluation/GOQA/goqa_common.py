#!/usr/bin/env python3
"""Shared, dependency-free utilities for the AU/NZ/India GOQA package."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
PROMPT_VARIANTS = 5
LABELS = tuple(chr(ord("A") + index) for index in range(18))
SYSTEM_PROMPT = (
    "Answer the survey question by choosing exactly one of the listed options. "
    "Return only the option letter."
)

AUSTRALIA = "Australia"
NEW_ZEALAND = "New Zealand"
INDIA_CURRENT = "India (Current national sample)"
INDIA_NON_NATIONAL = "India (Non-national sample)"
INDIA_OLD = "India (Old national sample)"
TARGET_GROUPS = (
    AUSTRALIA,
    NEW_ZEALAND,
    INDIA_CURRENT,
    INDIA_NON_NATIONAL,
    INDIA_OLD,
)
AU_NZ_GROUPS = (AUSTRALIA, NEW_ZEALAND)
INDIA_GROUPS = (INDIA_CURRENT, INDIA_NON_NATIONAL, INDIA_OLD)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(text: str) -> dict[str, list[float]]:
    """Parse the dictionary-like selections field used by the source CSV."""
    cleaned = text
    prefix = "defaultdict(<class 'list'>, "
    if cleaned.startswith(prefix) and cleaned.endswith(")"):
        cleaned = cleaned[len(prefix) : -1]
    value = ast.literal_eval(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Selections field is not a dictionary")
    return value


def normalize(values: Iterable[float]) -> list[float] | None:
    numbers = [max(0.0, float(value)) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("Invalid probability distribution")
    total = sum(numbers)
    if total <= 0:
        return None
    return [value / total for value in numbers]


def option_order(question_id: str, option_count: int, variant: int) -> list[int]:
    """Match the deterministic five-prompt ordering used in the M0 baseline."""
    if not 0 <= variant < PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {variant}")
    order = list(range(option_count))
    if variant:
        material = f"goqa-five-prompt-v1:{question_id}:{variant}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        random.Random(seed).shuffle(order)
    return order


def build_messages(question: dict[str, Any], order: list[int]) -> list[dict[str, str]]:
    options = question["options"]
    option_lines = [
        f"({LABELS[position]}) {options[source_index]}"
        for position, source_index in enumerate(order)
    ]
    user = "\n".join(
        [
            question["question"],
            "",
            "Here are the options:",
            *option_lines,
            "",
            "If you had to select one of the options, return only its letter.",
            "Answer:",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def softmax(log_probabilities: list[float]) -> list[float]:
    if not log_probabilities:
        raise ValueError("Cannot normalize an empty log-probability vector")
    maximum = max(log_probabilities)
    weights = [math.exp(value - maximum) for value in log_probabilities]
    total = sum(weights)
    return [value / total for value in weights]


def js_distance(left: list[float], right: list[float]) -> float:
    """Return base-2 Jensen-Shannon distance (sqrt of JS divergence)."""
    if len(left) != len(right) or not left:
        raise ValueError("Jensen-Shannon distributions have incompatible dimensions")
    midpoint = [(p + q) / 2 for p, q in zip(left, right)]

    def kl(source: list[float], target: list[float]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(source, target)
            if value > 0
        )

    return math.sqrt((kl(left, midpoint) + kl(right, midpoint)) / 2)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except Exception as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or question_id in seen:
                raise ValueError(f"Missing or duplicate question_id at {path}:{number}")
            seen.add(question_id)
            options = row.get("options")
            if not isinstance(options, list) or not 2 <= len(options) <= len(LABELS):
                raise ValueError(f"{question_id}: unsupported option count")
            human = row.get("human_distributions")
            if not isinstance(human, dict) or not human:
                raise ValueError(f"{question_id}: target human distributions are missing")
            if not set(human).issubset(TARGET_GROUPS):
                raise ValueError(f"{question_id}: non-target human distribution found")
            for group, distribution in human.items():
                if len(distribution) != len(options):
                    raise ValueError(f"{question_id}/{group}: distribution length mismatch")
                normalized = normalize(distribution)
                if normalized is None or any(
                    abs(left - right) > 1e-10
                    for left, right in zip(normalized, distribution)
                ):
                    raise ValueError(f"{question_id}/{group}: distribution is not normalized")
            questions.append(row)
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def validate_model_distribution(
    question_id: str, values: Any, option_count: int
) -> list[float]:
    if not isinstance(values, list) or len(values) != option_count:
        raise ValueError(f"{question_id}: model distribution length mismatch")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ValueError(f"{question_id}: invalid model probability")
    if not math.isclose(sum(numbers), 1.0, abs_tol=1e-7):
        raise ValueError(f"{question_id}: model probabilities do not sum to one")
    return numbers
