"""Optional LLM wording refinement with deterministic acceptance gates.

The LLM is only a language editor. It does not choose the accounting concept;
the synonym motor chooses the candidate and this module decides whether the
edited wording remains close enough to the deterministic evidence.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from synonym_engine import text_cosine_similarity


DEFAULT_MODEL = "gpt-5-mini"


def _clean_label(value: object) -> str:
    label = re.sub(r"\s+", " ", str(value or "")).strip()
    label = re.sub(r"^[\"'`]+|[\"'`]+$", "", label)
    return label[:160]


def validate_refined_label(refined_label: str, source_description: str, synonym_variant: str, canonical_label: str, connector_words: set[str] | None = None) -> dict[str, Any]:
    """Re-score the LLM output against deterministic evidence."""
    label = _clean_label(refined_label)
    connectors = connector_words or set()
    source_score = text_cosine_similarity(label, source_description, connectors)
    variant_score = text_cosine_similarity(label, synonym_variant, connectors)
    canonical_score = text_cosine_similarity(label, canonical_label, connectors)
    token_count = len(label.split())
    reasons: list[str] = []
    if not label:
        reasons.append("empty label")
    if token_count > 12:
        reasons.append("label is too long")
    if canonical_score < 0.45:
        reasons.append("low similarity to canonical label")
    if variant_score < 0.35 and source_score < 0.35:
        reasons.append("low similarity to source and synonym variant")
    return {
        "refined_label": label,
        "source_cosine": round(source_score, 4),
        "synonym_variant_cosine": round(variant_score, 4),
        "canonical_cosine": round(canonical_score, 4),
        "accepted": not reasons,
        "reasons": reasons or ["passed deterministic revalidation"],
    }


def refine_label_with_llm(source_description: str, synonym_variant: str, canonical_label: str, account_type: str = "", industry: str = "", context: str = "", model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Ask the configured model to polish wording and revalidate the result."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "disabled", "accepted": False, "refined_label": "", "reasons": ["OPENAI_API_KEY is not configured"]}

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an English accounting-label editor. Return one concise professional account label. "
                        "Preserve the accounting meaning of the supplied synonym candidate and canonical label. "
                        "Do not invent an account, change asset/liability/equity/income/expense classification, add a balance, "
                        "or provide an accounting conclusion. Use title case where natural. Output JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "source_description": source_description,
                        "synonym_candidate": synonym_variant,
                        "canonical_label": canonical_label,
                        "account_type": account_type,
                        "industry": industry,
                        "chart_context": context,
                    }),
                },
            ],
            max_completion_tokens=120,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "account_label_refinement",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"refined_label": {"type": "string"}},
                        "required": ["refined_label"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        validation = validate_refined_label(payload.get("refined_label", ""), source_description, synonym_variant, canonical_label)
        validation.update({"status": "accepted" if validation["accepted"] else "rejected", "model": model})
        return validation
    except Exception as error:
        return {"status": "error", "accepted": False, "refined_label": "", "reasons": [f"LLM refinement failed: {error}"]}
