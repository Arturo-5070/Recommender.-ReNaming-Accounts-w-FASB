import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from column_mapping import cosine_similarity, default_mapping, mapping_status, rename_using_mapping, suggest_mappings


def test_cosine_similarity_is_name_based():
    assert cosine_similarity("Acct No", "account number") > cosine_similarity("posting date", "account number")
    assert cosine_similarity("posting date", "posting date") == 1.0


def test_mapping_returns_ranked_alternatives_and_scores():
    columns = ["Acct No", "Type", "Closing Balance", "Period", "Name"]
    suggestions = suggest_mappings(columns, "Balances", alternatives=3)
    assert "account number" in suggestions
    assert suggestions["account number"][0]["source_column"] == "Acct No"
    assert 0.0 <= suggestions["account number"][0]["cosine_similarity"] <= 1.0


def test_default_mapping_and_manual_override_preserve_unmapped_fields():
    source = ["Acct No", "Type", "Closing Balance", "Period", "Name", "ERP Region"]
    mapping = default_mapping(source, "Balances", minimum_similarity=0.0)
    mapping["account description"] = "Name"
    status = mapping_status(mapping, "Balances")
    assert status["valid"] is True
    frame = pd.DataFrame({column: ["x"] for column in source})
    normalized = rename_using_mapping(frame, mapping)
    assert "account description" in normalized.columns
    assert "ERP Region" in normalized.columns
