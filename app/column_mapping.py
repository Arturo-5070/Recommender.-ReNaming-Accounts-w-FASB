"""Name-only column mapping for heterogeneous ERP exports."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable

import pandas as pd


TARGET_COLUMNS: dict[str, tuple[str, ...]] = {
    "Balances": (
        "account number",
        "account type",
        "balances",
        "financial period",
        "account description",
    ),
    "Journal entries": (
        "document number",
        "posting date",
        "account number",
        "amount",
        "financial period",
    ),
    "Auditor labels": (
        "account number",
        "auditor approved label",
    ),
}


def normalize_name(value: object) -> str:
    """Normalize an ERP header without inspecting any row values."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _ngrams(value: object) -> Counter[str]:
    normalized = normalize_name(value)
    padded = f"^{normalized}$"
    result: Counter[str] = Counter()
    for width in (2, 3, 4, 5):
        for start in range(len(padded) - width + 1):
            result[padded[start : start + width]] += 1
    return result


def cosine_similarity(left: object, right: object) -> float:
    """Cosine similarity between character n-gram vectors for two headers."""
    left_vector = _ngrams(left)
    right_vector = _ngrams(right)
    if not left_vector or not right_vector:
        return 0.0
    common = set(left_vector).intersection(right_vector)
    numerator = sum(left_vector[key] * right_vector[key] for key in common)
    left_length = math.sqrt(sum(value * value for value in left_vector.values()))
    right_length = math.sqrt(sum(value * value for value in right_vector.values()))
    if left_length == 0 or right_length == 0:
        return 0.0
    return numerator / (left_length * right_length)


def suggest_mappings(source_columns: Iterable[object], file_type: str, alternatives: int = 3) -> dict[str, list[dict[str, object]]]:
    """Rank source-header candidates for every target using only header text."""
    if file_type not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported file type: {file_type}")
    source = [str(column) for column in source_columns]
    output: dict[str, list[dict[str, object]]] = {}
    for target in TARGET_COLUMNS[file_type]:
        ranked = [
            {
                "source_column": column,
                "target_column": target,
                "cosine_similarity": round(cosine_similarity(column, target), 4),
            }
            for column in source
        ]
        ranked.sort(key=lambda item: (item["cosine_similarity"], item["source_column"]), reverse=True)
        output[target] = ranked[: max(1, alternatives)]
    return output


def default_mapping(source_columns: Iterable[object], file_type: str, minimum_similarity: float = 0.35) -> dict[str, str | None]:
    """Choose non-duplicated suggestions only when their cosine score is adequate."""
    columns = [str(column) for column in source_columns]
    suggestions = suggest_mappings(columns, file_type, alternatives=len(columns) or 1)
    selected: dict[str, str | None] = {}
    used: set[str] = set()
    for target in TARGET_COLUMNS[file_type]:
        choice: str | None = None
        for item in suggestions[target]:
            source = str(item["source_column"])
            score = float(item["cosine_similarity"])
            if source not in used and score >= minimum_similarity:
                choice = source
                used.add(source)
                break
        selected[target] = choice
    return selected


def rename_using_mapping(frame: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename mapped source columns and preserve all unmapped columns."""
    inverse: dict[str, str] = {}
    for target, source in mapping.items():
        if source and source != "Do not map":
            inverse[source] = target
    return frame.rename(columns=inverse).copy()


def mapping_status(mapping: dict[str, str | None], file_type: str) -> dict[str, object]:
    required = list(TARGET_COLUMNS[file_type])
    missing = [target for target in required if not mapping.get(target) or mapping[target] == "Do not map"]
    chosen = [str(mapping[target]) for target in required if mapping.get(target) and mapping[target] != "Do not map"]
    duplicates = sorted({value for value in chosen if chosen.count(value) > 1})
    return {
        "valid": len(missing) == 0 and len(duplicates) == 0,
        "required_columns": required,
        "missing_targets": missing,
        "duplicate_sources": duplicates,
    }
