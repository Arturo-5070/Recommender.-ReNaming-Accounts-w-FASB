"""Deterministic account-label recommendation engine for the Streamlit prototype.

The taxonomy is intentionally compact and uses public US GAAP/XBRL-style labels.
It is a product prototype, not an accounting conclusion engine.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from synonym_engine import get_engine
from llm_refinement import refine_label_with_llm


TAXONOMY: list[dict[str, Any]] = [
    {"label": "Cash and Cash Equivalents", "category": "Assets", "account_types": {"asset", "current asset", "cash"}, "synonyms": ["cash", "bank", "checking", "savings", "cash equivalents", "petty cash", "treasury"], "industries": {"all": 0.05}, "asc_hint": "ASC 210 / presentation"},
    {"label": "Accounts Receivable, Net", "category": "Assets", "account_types": {"asset", "current asset", "receivable"}, "synonyms": ["accounts receivable", "trade receivable", "customer receivable", "ar", "receivables", "trade debtors"], "industries": {"all": 0.05}, "asc_hint": "ASC 310 / receivables"},
    {"label": "Inventory", "category": "Assets", "account_types": {"asset", "current asset", "inventory"}, "synonyms": ["inventory", "stock", "merchandise", "raw materials", "work in process", "finished goods", "wip"], "industries": {"retail": 0.18, "manufacturing": 0.20, "consumer goods": 0.18}, "asc_hint": "ASC 330"},
    {"label": "Prepaid Expenses and Other Current Assets", "category": "Assets", "account_types": {"asset", "current asset", "prepaid"}, "synonyms": ["prepaid", "prepayments", "prepaid expense", "deferred expense", "other current asset", "short term asset"], "industries": {"all": 0.05}, "asc_hint": "ASC 210 / presentation"},
    {"label": "Property, Plant and Equipment, Net", "category": "Assets", "account_types": {"asset", "noncurrent asset", "fixed asset", "ppe"}, "synonyms": ["property plant equipment", "ppe", "fixed assets", "fixed asset", "equipment", "machinery", "buildings", "land", "capital assets"], "industries": {"manufacturing": 0.18, "energy": 0.20, "real estate": 0.16}, "asc_hint": "ASC 360"},
    {"label": "Operating Lease Right-of-Use Assets", "category": "Assets", "account_types": {"asset", "noncurrent asset", "lease"}, "synonyms": ["right of use asset", "rou asset", "operating lease asset", "lease asset", "use of asset"], "industries": {"retail": 0.12, "transportation": 0.12}, "asc_hint": "ASC 842"},
    {"label": "Accounts Payable", "category": "Liabilities", "account_types": {"liability", "current liability", "payable"}, "synonyms": ["accounts payable", "trade payable", "vendor payable", "ap", "supplier payable", "trade creditors"], "industries": {"all": 0.05}, "asc_hint": "ASC 405 / presentation"},
    {"label": "Accrued Liabilities", "category": "Liabilities", "account_types": {"liability", "current liability", "accrual"}, "synonyms": ["accrued liability", "accrual", "accrued expense", "accrued expenses", "accrued payable", "accrued payroll", "accrued interest"], "industries": {"all": 0.05}, "asc_hint": "ASC 450 / presentation"},
    {"label": "Short-Term Debt", "category": "Liabilities", "account_types": {"liability", "current liability", "debt"}, "synonyms": ["short term debt", "current debt", "current portion of debt", "revolver", "line of credit", "bank loan current"], "industries": {"all": 0.05}, "asc_hint": "ASC 470"},
    {"label": "Long-Term Debt", "category": "Liabilities", "account_types": {"liability", "noncurrent liability", "debt"}, "synonyms": ["long term debt", "term loan", "senior notes", "bonds payable", "mortgage payable", "noncurrent debt"], "industries": {"all": 0.05}, "asc_hint": "ASC 470"},
    {"label": "Deferred Revenue", "category": "Liabilities", "account_types": {"liability", "current liability", "deferred revenue"}, "synonyms": ["deferred revenue", "unearned revenue", "contract liability", "customer advances", "deferred income", "subscription liability"], "industries": {"software": 0.20, "saas": 0.22, "media": 0.12}, "asc_hint": "ASC 606"},
    {"label": "Common Stock", "category": "Equity", "account_types": {"equity", "capital"}, "synonyms": ["common stock", "share capital", "ordinary shares", "common shares", "paid in capital"], "industries": {"all": 0.05}, "asc_hint": "ASC 505"},
    {"label": "Retained Earnings", "category": "Equity", "account_types": {"equity", "retained earnings"}, "synonyms": ["retained earnings", "accumulated earnings", "prior year earnings", "accumulated deficit", "deficit"], "industries": {"all": 0.05}, "asc_hint": "ASC 505"},
    {"label": "Revenue", "category": "Income", "account_types": {"income", "revenue", "sales"}, "synonyms": ["revenue", "sales", "income", "turnover", "service revenue", "subscription revenue", "product revenue", "net sales"], "industries": {"software": 0.15, "saas": 0.17, "retail": 0.13}, "asc_hint": "ASC 606"},
    {"label": "Cost of Revenue", "category": "Expense", "account_types": {"expense", "cost", "cogs"}, "synonyms": ["cost of revenue", "cost of goods sold", "cogs", "cost of sales", "cost of services", "direct cost", "cost of products"], "industries": {"retail": 0.18, "manufacturing": 0.17, "software": 0.10}, "asc_hint": "ASC 330 / ASC 720 presentation"},
    {"label": "Selling, General and Administrative Expense", "category": "Expense", "account_types": {"expense", "operating expense", "sga", "sg&a"}, "synonyms": ["selling general administrative", "sg&a", "sga", "general and administrative", "administrative expense", "selling expense", "overhead", "operating expense"], "industries": {"all": 0.06}, "asc_hint": "ASC 720 / presentation"},
    {"label": "Research and Development Expense", "category": "Expense", "account_types": {"expense", "operating expense", "r&d", "research"}, "synonyms": ["research and development", "r&d", "rd expense", "product development", "engineering expense", "research expense"], "industries": {"software": 0.22, "saas": 0.22, "biotech": 0.22, "technology": 0.18}, "asc_hint": "ASC 730"},
    {"label": "Depreciation and Amortization Expense", "category": "Expense", "account_types": {"expense", "depreciation", "amortization"}, "synonyms": ["depreciation", "amortization", "depreciation expense", "amortization expense", "d&a", "depreciation and amortisation"], "industries": {"manufacturing": 0.12, "energy": 0.15, "real estate": 0.14}, "asc_hint": "ASC 360 / ASC 350"},
    {"label": "Interest Expense", "category": "Expense", "account_types": {"expense", "interest"}, "synonyms": ["interest expense", "finance cost", "borrowing cost", "interest paid", "debt interest"], "industries": {"all": 0.05}, "asc_hint": "ASC 835"},
    {"label": "Income Tax Expense", "category": "Expense", "account_types": {"expense", "tax"}, "synonyms": ["income tax", "tax expense", "corporate tax", "provision for income taxes", "current tax", "deferred tax expense"], "industries": {"all": 0.05}, "asc_hint": "ASC 740"},
    {"label": "Other Income (Expense), Net", "category": "Income", "account_types": {"income", "expense", "other income"}, "synonyms": ["other income", "other expense", "non operating income", "nonoperating income", "miscellaneous income", "miscellaneous expense", "gain loss"], "industries": {"all": 0.02}, "asc_hint": "ASC 225 / presentation"},
]

STOP_WORDS = {"and", "the", "of", "for", "in", "on", "net", "other", "expense", "expenses", "account", "accounts", "current", "noncurrent", "long", "short"}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s/-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[_/ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if token and token not in STOP_WORDS}


def similarity(left: str, right: str) -> float:
    left_norm, right_norm = normalize_text(left), normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens, right_tokens = tokens(left_norm), tokens(right_norm)
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return min(1.0, 0.65 * jaccard + 0.35 * sequence)


def _account_type_bonus(account_type: str, item: dict[str, Any]) -> float:
    normalized_type = normalize_text(account_type)
    if not normalized_type:
        return 0.0
    item_types = {normalize_text(value) for value in item["account_types"]}
    if any(t in normalized_type or normalized_type in t for t in item_types):
        return 1.0
    category = normalize_text(item["category"])
    if category in normalized_type or normalized_type in category:
        return 0.45
    return 0.0


def _industry_bonus(industry: str, item: dict[str, Any]) -> float:
    normalized_industry = normalize_text(industry)
    if not normalized_industry:
        return 0.0
    for key, value in item["industries"].items():
        key_norm = normalize_text(key)
        if key_norm == "all":
            continue
        if key_norm in normalized_industry or normalized_industry in key_norm:
            return float(value)
    return float(item["industries"].get("all", 0.0))


def _evidence(description: str, item: dict[str, Any], account_type: str, industry: str, chart_context: str, auditor_label: str) -> list[str]:
    evidence: list[str] = []
    desc_tokens = tokens(description)
    matched = sorted({syn for syn in item["synonyms"] if tokens(syn) & desc_tokens})
    if matched:
        evidence.append("matched synonym(s): " + ", ".join(matched[:4]))
    if _account_type_bonus(account_type, item) > 0:
        evidence.append(f"account type supports {item['category'].lower()} classification")
    if _industry_bonus(industry, item) > 0.10:
        evidence.append(f"industry context supports {normalize_text(industry)} usage")
    if chart_context and similarity(chart_context, item["label"]) >= 0.45:
        evidence.append("similar label found in chart-of-accounts context")
    if auditor_label and similarity(auditor_label, item["label"]) >= 0.50:
        evidence.append("historical auditor label is similar")
    if not evidence:
        evidence.append("closest public taxonomy-style label; manual review recommended")
    return evidence


def recommend_account(description: str, account_type: str = "", industry: str = "", chart_context: str = "", auditor_label: str = "", amount: float | None = None, top_n: int = 3, llm_refine: bool = False, llm_model: str = "gpt-5-mini") -> dict[str, Any]:
    """Return three ranked recommendation lists from one scoring engine."""
    desc = normalize_text(description)
    context_text = " ".join(part for part in [description, account_type, industry, chart_context] if part)
    candidates: list[dict[str, Any]] = []
    synonym_engine = get_engine()
    for item in TAXONOMY:
        lexical = max([similarity(desc, item["label"])] + [similarity(desc, synonym) for synonym in item["synonyms"]])
        context_similarity = similarity(context_text, item["label"])
        type_bonus = _account_type_bonus(account_type, item)
        industry_bonus = _industry_bonus(industry, item)
        chart_bonus = similarity(chart_context, item["label"]) if chart_context else 0.0
        synonym_result = synonym_engine.best_variant_for_label(description, item["label"], industry=industry, account_type=account_type, context=chart_context)
        synonym_score = float(synonym_result["best_score"])
        auditor_similarity = max([similarity(auditor_label, item["label"])] + [similarity(auditor_label, synonym) for synonym in item["synonyms"]]) if auditor_label else 0.0
        auditor_text_similarity = similarity(auditor_label, desc) if auditor_label else 0.0
        asc_score = 0.58 * lexical + 0.20 * synonym_score + 0.10 * context_similarity + 0.12 * type_bonus
        context_score = 0.30 * lexical + 0.25 * synonym_score + 0.20 * context_similarity + 0.16 * type_bonus + 0.06 * min(1.0, industry_bonus * 4) + 0.03 * chart_bonus
        auditor_score = 0.20 * lexical + 0.15 * synonym_score + 0.12 * context_similarity + 0.08 * type_bonus + 0.45 * max(auditor_similarity, auditor_text_similarity) if auditor_label else context_score
        synonym_changes = ", ".join(f"{change['source_term']} -> {change['synonym']}" for change in synonym_result.get("matched_changes", [])[:4])
        synonym_evidence = f"synonym variant: {synonym_result['best_text']} (cosine {synonym_score:.2f}, library {synonym_result['library_version']})"
        if synonym_changes:
            synonym_evidence += f"; changes: {synonym_changes}"
        candidates.append({"label": item["label"], "category": item["category"], "asc_hint": item["asc_hint"], "lexical": lexical, "synonym_score": synonym_score, "synonym_result": synonym_result, "asc_score": asc_score, "context_score": context_score, "auditor_score": auditor_score, "evidence": _evidence(description, item, account_type, industry, chart_context, auditor_label) + [synonym_evidence]})

    def ranked(score_key: str) -> list[dict[str, Any]]:
        rows = sorted(candidates, key=lambda row: row[score_key], reverse=True)[:top_n]
        result = []
        for rank, row in enumerate(rows, 1):
            confidence = min(0.99, max(0.05, row[score_key]))
            synonym_result = row.get("synonym_result", {})
            result.append({"rank": rank, "recommended_label": row["label"], "category": row["category"], "confidence": round(confidence, 3), "asc_hint": row["asc_hint"], "synonym_variant": synonym_result.get("best_text", ""), "synonym_cosine": round(float(synonym_result.get("best_score", 0.0)), 3), "synonym_change_log": "; ".join(f"{change['source_term']} -> {change['synonym']}" for change in synonym_result.get("matched_changes", [])), "synonym_library_version": synonym_result.get("library_version", ""), "llm_refined_label": "", "llm_refinement_status": "not requested", "llm_revalidation": "", "evidence": "; ".join(row["evidence"])})
        return result

    output = {
        "asc": ranked("asc_score"),
        "context": ranked("context_score"),
        "auditor": ranked("auditor_score"),
        "auditor_label_available": bool(normalize_text(auditor_label)),
    }
    if llm_refine:
        seen_inputs: dict[tuple[str, str], dict[str, Any]] = {}
        for layer in ("asc", "context", "auditor"):
            if not output[layer]:
                continue
            top = output[layer][0]
            cache_key = (top["synonym_variant"], top["recommended_label"])
            if cache_key not in seen_inputs:
                seen_inputs[cache_key] = refine_label_with_llm(description, top["synonym_variant"], top["recommended_label"], account_type=account_type, industry=industry, context=chart_context, model=llm_model)
            refinement = seen_inputs[cache_key]
            top["llm_refined_label"] = refinement.get("refined_label", "") if refinement.get("accepted") else ""
            top["llm_refinement_status"] = refinement.get("status", "unknown")
            top["llm_revalidation"] = "; ".join(refinement.get("reasons", []))
            top["evidence"] += "; LLM grammar stage: " + refinement.get("status", "unknown") + "; " + "; ".join(refinement.get("reasons", []))
    return output


def recommend_dataframe(df: pd.DataFrame, industry: str = "", chart_column: str = "", auditor_column: str = "", top_n: int = 3, llm_refine: bool = False, llm_model: str = "gpt-5-mini") -> pd.DataFrame:
    """Generate flattened recommendation columns for a balance-level dataframe."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        description = row.get("account description", row.get("description", ""))
        account_type = row.get("account type", "")
        chart_context = row.get(chart_column, "") if chart_column else ""
        auditor_label = row.get(auditor_column, "") if auditor_column else ""
        amount = row.get("balances", row.get("balance", None))
        recommendation = recommend_account(description, account_type, industry, chart_context, auditor_label, amount, top_n, llm_refine=llm_refine, llm_model=llm_model)
        output = row.to_dict()
        for layer, key in [("asc", "asc"), ("context", "context"), ("auditor", "auditor")]:
            for item in recommendation[key]:
                suffix = f"_{item['rank']}"
                output[f"{layer}_label{suffix}"] = item["recommended_label"]
                output[f"{layer}_confidence{suffix}"] = item["confidence"]
                output[f"{layer}_evidence{suffix}"] = item["evidence"]
                output[f"{layer}_synonym_variant{suffix}"] = item.get("synonym_variant", "")
                output[f"{layer}_synonym_cosine{suffix}"] = item.get("synonym_cosine", 0.0)
                output[f"{layer}_synonym_change_log{suffix}"] = item.get("synonym_change_log", "")
                output[f"{layer}_synonym_library_version{suffix}"] = item.get("synonym_library_version", "")
                output[f"{layer}_top_label"] = recommendation[key][0]["recommended_label"] if recommendation[key] else ""
                output[f"{layer}_top_confidence"] = recommendation[key][0]["confidence"] if recommendation[key] else 0.0
                output[f"{layer}_llm_refined_label"] = recommendation[key][0].get("llm_refined_label", "") if recommendation[key] else ""
                output[f"{layer}_llm_refinement_status"] = recommendation[key][0].get("llm_refinement_status", "not requested") if recommendation[key] else "not requested"
                output[f"{layer}_llm_revalidation"] = recommendation[key][0].get("llm_revalidation", "") if recommendation[key] else ""

        output["review_required"] = bool(recommendation["auditor_label_available"] is False or recommendation["context"][0]["confidence"] < 0.55 or recommendation["auditor"][0]["confidence"] < 0.60)
        rows.append(output)
    return pd.DataFrame(rows)


def materiality_and_quality_flags(df: pd.DataFrame, materiality_ratio: float = 0.01) -> pd.DataFrame:
    """Add heuristic review flags; these are not audit materiality conclusions."""
    result = df.copy()
    balance_col = next((column for column in ["balances", "balance", "amount"] if column in result.columns), None)
    if balance_col:
        values = pd.to_numeric(result[balance_col], errors="coerce").fillna(0.0)
        total_abs = float(values.abs().sum())
        result["absolute_balance"] = values.abs()
        result["balance_share_of_file"] = values.abs() / total_abs if total_abs else 0.0
        result["materiality_flag"] = result["balance_share_of_file"] >= materiality_ratio
    else:
        result["absolute_balance"] = 0.0
        result["balance_share_of_file"] = 0.0
        result["materiality_flag"] = False
    if "review_required" not in result.columns:
        result["review_required"] = True
    asc_confidence = pd.to_numeric(result["asc_top_confidence"], errors="coerce").fillna(0.0) if "asc_top_confidence" in result.columns else pd.Series(0.0, index=result.index)
    context_confidence = pd.to_numeric(result["context_top_confidence"], errors="coerce").fillna(0.0) if "context_top_confidence" in result.columns else pd.Series(0.0, index=result.index)
    result["unusual_account_flag"] = (asc_confidence < 0.45) | (context_confidence < 0.45)
    if "account type" in result.columns and "asc_top_label" in result.columns:
        labels = result["asc_top_label"].astype(str)
        types = result["account type"].astype(str)
        result["inconsistency_flag"] = labels.str.contains("Revenue|Income", case=False, na=False) & types.str.contains("liabil|asset", case=False, na=False)
    else:
        result["inconsistency_flag"] = False
    result["review_reason"] = result.apply(lambda row: "; ".join(reason for reason, flag in [("materiality", row.get("materiality_flag", False)), ("low confidence", row.get("review_required", False)), ("unusual account", row.get("unusual_account_flag", False)), ("classification inconsistency", row.get("inconsistency_flag", False))] if bool(flag)) or "no automatic flag", axis=1)
    return result
