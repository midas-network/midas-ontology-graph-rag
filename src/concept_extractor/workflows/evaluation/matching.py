"""Matching utilities: vector, LLM semantic, string, and date matching."""
from __future__ import annotations

import logging
import re
from typing import Any

from concept_extractor.utils.llm.llm_client import send_to_llm
from concept_extractor.utils.evaluation.vector_similarity import vector_match_tiered

LOGGER = logging.getLogger("midas-llm")


def build_semantic_eval_prompt(
        attribute: str,
        extracted_value: str,
        expected_values: list[str],
) -> str:
    expected_str = ", ".join(f'"{v}"' for v in expected_values)

    return f"""You are evaluating whether an extracted value semantically matches any of the expected values for a scientific metadata field.

Field: {attribute}
Extracted value: "{extracted_value}"
Expected values (any match is acceptable): [{expected_str}]

Does the extracted value mean the same thing as any of the expected values? Consider synonyms, abbreviations, and related terms.

Examples:
- "influenza" matches "flu" or "Influenza A" (same disease)
- "agent-based" matches "ABM" or "individual-based model" (same model type)  
- "USA" matches "United States" or "US" (same country)
- "SARS-CoV-2" matches "COVID-19 virus" (same pathogen)
- "children" matches "pediatric population" or "school-aged children" (similar population)
- "May 2022" matches "2022-05" (same date, different format)
- "December 2023" matches "2023-12" (same date, different format)

Respond with ONLY one word: MATCH or NO_MATCH"""


def evaluate_semantic_match(
        attribute: str,
        extracted_value: str,
        expected_values: list[str],
        llm_model: str,
        llm_host: str,
        logger: logging.Logger,
        timeout_seconds: int = 30,
        api_type: str = "ollama",
) -> tuple[bool, str | None]:
    prompt = build_semantic_eval_prompt(attribute, extracted_value, expected_values)

    try:
        response = send_to_llm(
            prompt=prompt,
            llm_model=llm_model,
            llm_host=llm_host,
            timeout_seconds=timeout_seconds,
            logger=logger,
            api_type=api_type,
        )

        result = response.content.strip().upper()
        is_match = "MATCH" in result and "NO_MATCH" not in result

        if is_match:
            for exp in expected_values:
                if exp.lower() in extracted_value.lower() or extracted_value.lower() in exp.lower():
                    return True, exp
            return True, expected_values[0]

        return False, None

    except Exception as e:
        logger.warning("Semantic eval failed, falling back to string match: %s", e)
        return fallback_string_match(extracted_value, expected_values)


def fallback_string_match(extracted_value: str, expected_values: list[str]) -> tuple[bool, str | None]:
    extracted_norm = extracted_value.lower().strip().replace("-", " ").replace("_", " ")

    for expected in expected_values:
        expected_norm = expected.lower().strip().replace("-", " ").replace("_", " ")
        if extracted_norm == expected_norm:
            return True, expected
        if len(extracted_norm) >= 3 and len(expected_norm) >= 3:
            if expected_norm in extracted_norm or extracted_norm in expected_norm:
                return True, expected
        if dates_match(extracted_value, expected):
            return True, expected

    return False, None


def dates_match(extracted: str, expected: str) -> bool:
    month_names = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12',
    }

    def normalize_date(date_str: str) -> str:
        date_str = date_str.lower().strip().replace('**', '').strip()
        for month_name, month_num in month_names.items():
            if month_name in date_str:
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return f"{year_match.group(1)}-{month_num}"
        ym_match = re.match(r'^(\d{4})-(\d{1,2})(?:-\d{1,2})?$', date_str)
        if ym_match:
            return f"{ym_match.group(1)}-{ym_match.group(2).zfill(2)}"
        return date_str

    return normalize_date(extracted) == normalize_date(expected)


def _vector_model_score_sort_key(entry: dict[str, Any], original_index: int) -> tuple[int, float, int, str, int]:
    """Sort key for vector model score rows (highest similarity first)."""
    score = entry.get("similarity_score")
    numeric_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None

    decision_rank = {
        "MATCH": 0,
        "AMBIGUOUS": 1,
        "NO_MATCH": 2,
        "UNAVAILABLE": 3,
    }.get(str(entry.get("decision", "unknown")).upper(), 99)

    return (
        0 if numeric_score is not None else 1,
        -(numeric_score if numeric_score is not None else 0.0),
        decision_rank,
        str(entry.get("model", "unknown")).lower(),
        original_index,
    )


def _format_vector_model_scores(scores: Any) -> str:
    """Format per-model vector similarity details for logs/reports."""
    if not isinstance(scores, list) or not scores:
        return ""

    valid_entries: list[tuple[int, dict[str, Any]]] = [
        (idx, entry) for idx, entry in enumerate(scores) if isinstance(entry, dict)
    ]
    valid_entries.sort(key=lambda item: _vector_model_score_sort_key(item[1], item[0]))

    parts: list[str] = []
    for _, entry in valid_entries:
        model = str(entry.get("model", "unknown"))
        decision = str(entry.get("decision", "unknown"))
        score = entry.get("similarity_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            part = f"{model}:{decision}:{score:.3f}"
        else:
            part = f"{model}:{decision}"
        best_match = entry.get("best_match")
        if best_match:
            part = f"{part}:{best_match}"
        parts.append(part)
    return "; ".join(parts)


def vector_match_any_model(
    extracted_value: str,
    expected_values: list[str],
    embedding_models: list[str],
    vector_high_threshold: float = 0.85,
    vector_low_threshold: float = 0.50,
) -> tuple[str, str | None, float, list[dict[str, Any]], str | None]:
    """Return (decision, best_match, score, per_model_details, selected_model).

    Decision is one of: MATCH, AMBIGUOUS, NO_MATCH, UNAVAILABLE.
    """
    per_model: list[dict[str, Any]] = []
    per_model_details: list[dict[str, Any]] = []
    for model_name in embedding_models:
        decision, best_match, score = vector_match_tiered(
            extracted_value,
            expected_values,
            high_threshold=vector_high_threshold,
            low_threshold=vector_low_threshold,
            model_name=model_name,
        )
        per_model.append(
            {
                "model": model_name,
                "decision": decision,
                "best_match": best_match,
                "similarity_score_raw": score,
            }
        )
        per_model_details.append(
            {
                "model": model_name,
                "decision": decision,
                "best_match": best_match,
                "similarity_score": round(score, 4),
            }
        )

    match_candidates = [r for r in per_model if r["decision"] == "MATCH"]
    if match_candidates:
        best = max(match_candidates, key=lambda r: r["similarity_score_raw"])
        return (
            best["decision"],
            best["best_match"],
            best["similarity_score_raw"],
            per_model_details,
            best["model"],
        )

    ambiguous_candidates = [r for r in per_model if r["decision"] == "AMBIGUOUS"]
    if ambiguous_candidates:
        best = max(ambiguous_candidates, key=lambda r: r["similarity_score_raw"])
        return (
            best["decision"],
            best["best_match"],
            best["similarity_score_raw"],
            per_model_details,
            best["model"],
        )

    available = [r for r in per_model if r["decision"] != "UNAVAILABLE"]
    if available and all(r["decision"] == "NO_MATCH" for r in available):
        best = max(available, key=lambda r: r["similarity_score_raw"])
        if len(available) == len(per_model):
            return (
                "NO_MATCH",
                None,
                best["similarity_score_raw"],
                per_model_details,
                best["model"],
            )
        # Some models unavailable: defer to LLM tier instead of hard reject.
        return (
            "AMBIGUOUS",
            best["best_match"],
            best["similarity_score_raw"],
            per_model_details,
            best["model"],
        )

    return ("UNAVAILABLE", None, 0.0, per_model_details, None)
