from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app"))

from column_mapping import mapping_status, rename_using_mapping, suggest_mappings
from recommender import materiality_and_quality_flags, recommend_dataframe
from io_utils import make_balance_view


balances = pd.DataFrame([
    ["1000", "Asset", "1250000", "2026-01", "Cash at Bank", "North America"],
    ["2100", "Liability", "-250000", "2026-01", "Unearned Subscription Income", "North America"],
], columns=["Acct No", "Class", "Closing Balance", "Fiscal Month", "Name", "ERP Region"])
journal = pd.DataFrame([
    ["JE-1", "2026-01-03", "1000", "150000", "2026-01"],
], columns=["Journal ID", "Entry Date", "Acct No", "Local Amount", "Fiscal Month"])

mapping = {
    "account number": "Acct No",
    "account type": "Class",
    "balances": "Closing Balance",
    "financial period": "Fiscal Month",
    "account description": "Name",
}
assert mapping_status(mapping, "Balances")["valid"]
assert suggest_mappings(list(balances.columns), "Balances")["account number"][0]["source_column"] == "Acct No"
normalized_balances = rename_using_mapping(balances, mapping)
normalized_journal = rename_using_mapping(journal, {
    "document number": "Journal ID",
    "posting date": "Entry Date",
    "account number": "Acct No",
    "amount": "Local Amount",
    "financial period": "Fiscal Month",
})
recommendations = recommend_dataframe(normalized_balances, industry="SaaS software", chart_column="ERP Region", top_n=3)
recommendations = materiality_and_quality_flags(recommendations, materiality_ratio=0.01)
recommendations = make_balance_view(recommendations, normalized_journal)
assert "Deferred Revenue" == recommendations.loc[1, "context_top_label"]
assert "journal_entry_count" in recommendations.columns
assert "review_reason" in recommendations.columns
print("smoke flow passed")
