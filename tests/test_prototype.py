import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from io_utils import clean_dataframe, read_csv_utf16, validate_schema
from recommender import materiality_and_quality_flags, recommend_account, recommend_dataframe


def test_recommendation_for_deferred_revenue():
    result = recommend_account("Unearned subscription income", account_type="liability", industry="SaaS")
    assert result["context"][0]["recommended_label"] == "Deferred Revenue"
    assert result["context"][0]["confidence"] > 0.4


def test_auditor_label_changes_auditor_layer():
    result = recommend_account("Customer income", account_type="revenue", industry="software", auditor_label="Contract Liability")
    assert result["auditor"][0]["recommended_label"] == "Deferred Revenue"
    assert result["auditor_label_available"] is True


def test_utf16_reader_and_cleaning():
    raw = "account number;account type;balances;financial period;account description\n100;Asset;10;2026 01;  Cash   and   equivalents  \n".encode("utf-16")
    df = read_csv_utf16(io.BytesIO(raw), separator=";", clean=True, collapse_spaces=True, keep_printable=True)
    assert list(df.columns) == ["account number", "account type", "balances", "financial period", "account description"]
    assert df.loc[0, "account description"] == "Cash and equivalents"


def test_schema_and_flags():
    df = pd.DataFrame({
        "account number": ["100", "200"],
        "account type": ["asset", "liability"],
        "balances": [9000, 1000],
        "financial period": ["2026 01", "2026 01"],
        "account description": ["Cash", "Accounts payable"],
    })
    assert validate_schema(df, "Balances")["valid"] is True
    recs = recommend_dataframe(df, top_n=3)
    flagged = materiality_and_quality_flags(recs, materiality_ratio=0.80)
    assert int(flagged["materiality_flag"].sum()) == 1
    assert "review_reason" in flagged.columns
