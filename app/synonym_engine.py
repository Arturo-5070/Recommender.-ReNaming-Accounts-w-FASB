"""Reusable industry/context synonym motor for account-label candidates.

The motor is deterministic. It never replaces connector words and never uses
row values to generate a synonym suggestion. It returns the intermediate
variants so the UI can show evidence and auditors can review the process.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LIBRARY_PATH = Path(__file__).parents[1] / "data" / "synonym_library.json"
DEFAULT_CONNECTOR_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with", "without", "per", "via"}


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: object, connector_words: set[str]) -> list[str]:
    return [token for token in _normalize_text(value).split() if token and token not in connector_words]


def _token_counts(value: object, connector_words: set[str]) -> Counter[str]:
    return Counter(_tokens(value, connector_words))


def text_cosine_similarity(left: object, right: object, connector_words: set[str] | None = None) -> float:
    """Cosine similarity over normalized content-word frequency vectors."""
    connectors = DEFAULT_CONNECTOR_WORDS if connector_words is None else connector_words
    left_vector, right_vector = _token_counts(left, connectors), _token_counts(right, connectors)
    if not left_vector or not right_vector:
        return 0.0
    shared = set(left_vector).intersection(right_vector)
    numerator = sum(left_vector[key] * right_vector[key] for key in shared)
    left_length = math.sqrt(sum(value * value for value in left_vector.values()))
    right_length = math.sqrt(sum(value * value for value in right_vector.values()))
    return numerator / (left_length * right_length) if left_length and right_length else 0.0


class SynonymEngine:
    def __init__(self, library_path: str | Path = LIBRARY_PATH):
        with Path(library_path).open("r", encoding="utf-8") as handle:
            library = json.load(handle)
        self.version = str(library.get("library_version", "unknown"))
        self.connector_words = set(str(word).lower() for word in library.get("connector_words", []))
        self.industry_aliases = {_normalize_text(key): _normalize_text(value) for key, value in library.get("industry_aliases", {}).items()}
        self.entries: list[dict[str, Any]] = list(library.get("entries", []))

    def normalize_industry(self, industry: object) -> str:
        normalized = _normalize_text(industry)
        return self.industry_aliases.get(normalized, normalized)

    def _industry_match(self, entry: dict[str, Any], industry: object) -> float:
        requested = self.normalize_industry(industry)
        entry_industry = _normalize_text(entry.get("industry", "all"))
        if entry_industry in {"", "all"}:
            return 0.15
        if not requested:
            return 0.0
        if entry_industry == requested or entry_industry in requested or requested in entry_industry:
            return 0.35
        return -0.05

    def _account_type_match(self, entry: dict[str, Any], account_type: object) -> float:
        requested = _normalize_text(account_type)
        expected = _normalize_text(entry.get("account_type", ""))
        if not requested or not expected:
            return 0.0
        return 0.18 if expected in requested or requested in expected else -0.04

    def relevant_entries(self, industry: object = "", account_type: object = "", context: object = "") -> list[tuple[dict[str, Any], float]]:
        context_text = _normalize_text(context)
        scored: list[tuple[dict[str, Any], float]] = []
        for entry in self.entries:
            context_bonus = 0.0
            context_values = [_normalize_text(value) for value in entry.get("contexts", [])]
            if context_text and any(value and (value in context_text or context_text in value) for value in context_values):
                context_bonus = 0.16
            score = float(entry.get("priority", 1)) * 0.01 + self._industry_match(entry, industry) + self._account_type_match(entry, account_type) + context_bonus
            if score >= 0.0:
                scored.append((entry, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _replace_phrase(self, text: str, source: str, replacement: str) -> str | None:
        normalized_text = _normalize_text(text)
        normalized_source = _normalize_text(source)
        if not normalized_source:
            return None
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_source)}(?![a-z0-9])"
        if not re.search(pattern, normalized_text):
            return None
        return re.sub(pattern, _normalize_text(replacement), normalized_text, count=1)

    def _variant_key(self, value: str) -> str:
        return " ".join(_tokens(value, self.connector_words))

    def generate_variants(self, description: str, industry: str = "", account_type: str = "", context: str = "", max_iterations: int = 2, max_variants: int = 24) -> list[dict[str, Any]]:
        """Generate iterative synonym variants while excluding connector words."""
        original = _normalize_text(description)
        if not original:
            return []
        relevant = self.relevant_entries(industry, account_type, context)
        variants: list[dict[str, Any]] = [{"text": original, "iteration": 0, "changes": [], "content_words": _tokens(original, self.connector_words)}]
        seen = {self._variant_key(original)}
        frontier = variants.copy()
        for iteration in range(1, max(1, max_iterations) + 1):
            next_frontier: list[dict[str, Any]] = []
            for current in frontier:
                for entry, relevance in relevant:
                    replacement_pairs = list(zip(entry.get("source_terms", []), entry.get("synonyms", [])))
                    replacement_pairs += [(term, synonym) for term in entry.get("source_terms", []) for synonym in entry.get("synonyms", [])[:2]]
                    for source_term, synonym in replacement_pairs:
                        replaced = self._replace_phrase(current["text"], source_term, synonym)
                        if not replaced:
                            continue
                        key = self._variant_key(replaced)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        change = {"source_term": _normalize_text(source_term), "synonym": _normalize_text(synonym), "canonical_label": entry.get("canonical_label", ""), "entry_relevance": round(relevance, 4)}
                        next_frontier.append({"text": replaced, "iteration": iteration, "changes": current["changes"] + [change], "content_words": _tokens(replaced, self.connector_words)})
                        if len(variants) + len(next_frontier) >= max_variants:
                            break
                    if len(variants) + len(next_frontier) >= max_variants:
                        break
                if len(variants) + len(next_frontier) >= max_variants:
                    break
            variants.extend(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return variants[:max_variants]

    def best_variant_for_label(self, description: str, canonical_label: str, industry: str = "", account_type: str = "", context: str = "", max_iterations: int = 2) -> dict[str, Any]:
        variants = self.generate_variants(description, industry, account_type, context, max_iterations=max_iterations)
        if not variants:
            return {"best_text": "", "best_score": 0.0, "original_score": 0.0, "variants": [], "matched_changes": [], "library_version": self.version}
        ranked = []
        for variant in variants:
            score = text_cosine_similarity(variant["text"], canonical_label, self.connector_words)
            ranked.append({**variant, "cosine_similarity": round(score, 4)})
        ranked.sort(key=lambda item: (item["cosine_similarity"], -item["iteration"]), reverse=True)
        original_score = next((item["cosine_similarity"] for item in ranked if item["iteration"] == 0), 0.0)
        best = ranked[0]
        return {"best_text": best["text"], "best_score": best["cosine_similarity"], "original_score": original_score, "variants": ranked[:12], "matched_changes": best["changes"], "library_version": self.version}


_ENGINE: SynonymEngine | None = None


def get_engine() -> SynonymEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SynonymEngine()
    return _ENGINE
