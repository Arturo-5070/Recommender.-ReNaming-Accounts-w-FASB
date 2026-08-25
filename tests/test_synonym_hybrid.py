import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from llm_refinement import validate_refined_label
from synonym_engine import SynonymEngine, text_cosine_similarity


def test_connector_words_are_excluded_from_cosine_tokens():
    assert text_cosine_similarity("cost of revenue", "cost revenue") == pytest.approx(1.0)


def test_saas_synonym_variant_is_iterative_and_reusable():
    engine = SynonymEngine()
    result = engine.best_variant_for_label(
        "Unearned subscription income",
        "Deferred Revenue",
        industry="SaaS",
        account_type="liability",
        context="subscriptions",
    )
    assert result["library_version"] == "1.0.0"
    assert result["best_score"] >= result["original_score"]
    assert "deferred subscription revenue" in result["best_text"]
    assert result["matched_changes"]


def test_llm_output_is_accepted_only_after_similarity_revalidation():
    accepted = validate_refined_label("Deferred Revenue", "Unearned subscription income", "Deferred subscription revenue", "Deferred Revenue")
    rejected = validate_refined_label("Office Furniture", "Unearned subscription income", "Deferred subscription revenue", "Deferred Revenue")
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["reasons"]
