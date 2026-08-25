"""CSV ingestion and cleaning helpers for the prototype."""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Iterable

import pandas as pd


JOURNAL_REQUIRED = ["document number", "posting date", "account number", "amount", "financial period"]
BALANCE_REQUIRED = ["account number", "account type", "balances", "financial period", "account description"]


def clean_utf16_text(value: object, collapse_spaces: bool = True, keep_printable: bool = True) -> str:
    """Normalize text and optionally remove control/non-printable characters.

    The file is decoded with UTF-16 by pandas before this function runs. This
    function only removes characters that are unsafe for clean tabular display;
    it does not silently transliterate or change business words.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if keep_printable:
        text = "".join(character for character in text if character in "\n\r\t" or unicodedata.category(character)[0] != "C")
    if collapse_spaces:
        text = re.sub(r"[ \t\u00a0]+", " ", text)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = text.strip()
    return text


def clean_dataframe(df: pd.DataFrame, collapse_spaces: bool = True, keep_printable: bool = True) -> pd.DataFrame:
    result = df.copy()
    result.columns = [clean_utf16_text(column, collapse_spaces=True, keep_printable=True).lower() for column in result.columns]
    for column in result.columns:
        if result[column].dtype == "object" or pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].map(lambda value: clean_utf16_text(value, collapse_spaces, keep_printable))
    return result


def read_csv_utf16(uploaded_file, separator: str = ",", clean: bool = True, collapse_spaces: bool = True, keep_printable: bool = True) -> pd.DataFrame:
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    # utf-16 handles the BOM and common little-endian uploads.
    text = raw.decode("utf-16", errors="strict")
    df = pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=False)
    return clean_dataframe(df, collapse_spaces=collapse_spaces, keep_printable=keep_printable) if clean else df


def detect_separators(uploaded_file) -> list[tuple[str, int]]:
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    text = raw.decode("utf-16", errors="strict")
    sample = "\n".join(text.splitlines()[:20])
    candidates = [",", ";", "\t", "|"]
    counts = [(separator, sample.count(separator)) for separator in candidates]
    return sorted(counts, key=lambda item: item[1], reverse=True)


def missing_columns(df: pd.DataFrame, file_type: str) -> list[str]:
    required = JOURNAL_REQUIRED if file_type == "Journal entries" else BALANCE_REQUIRED
    columns = set(df.columns)
    return [column for column in required if column not in columns]


def validate_schema(df: pd.DataFrame, file_type: str) -> dict[str, object]:
    required = JOURNAL_REQUIRED if file_type == "Journal entries" else BALANCE_REQUIRED
    missing = missing_columns(df, file_type)
    return {
        "file_type": file_type,
        "required_columns": required,
        "missing_columns": missing,
        "valid": len(missing) == 0,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def make_balance_view(balance_df: pd.DataFrame, journal_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a balance-level table; journal data is retained for future context.

    Journal entries do not have a description column in the agreed test schema,
    so they are not forced into the account-name recommendation flow. The UI
    still validates and previews them and uses them for optional account activity
    metrics when the account number is present in both files.
    """
    result = balance_df.copy()
    if journal_df is not None and "account number" in journal_df.columns and "account number" in result.columns:
        journal_counts = journal_df.groupby("account number", dropna=False).size().rename("journal_entry_count")
        result = result.merge(journal_counts, how="left", left_on="account number", right_index=True)
        result["journal_entry_count"] = result["journal_entry_count"].fillna(0).astype(int)
    return result
