#!/usr/bin/env python
"""Evaluation script for LLM extraction accuracy.

Compares extracted attributes against expected values using LLM-based semantic matching.
Supports constrained decoding via MIDAS structured_vocab JSON schema (--constrained flag).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from concept_extractor.utils.structured_vocab.IdSynonyms import DISEASE_SYNONYMS
from concept_extractor.utils.parsers.extraction_parser import normalize_absent_value
from concept_extractor.utils.config import ExtractionConfig
from concept_extractor.utils.logging.logger import configure_logging
from concept_extractor.utils.llm.llm_client import send_to_llm
from concept_extractor.utils.llm.llm_utils import autodetect_llm_host
from concept_extractor.utils.prompt.builders import prepare_and_display_prompt, build_query
from concept_extractor.utils.reporting.evaluation_reports import (
    generate_evaluation_text_report,
    generate_abstract_evaluation_report,
)
from concept_extractor.utils.evaluation.vector_similarity import vector_match_tiered
from concept_extractor.constants import (
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_RESULTS_FOLDER,
    GOLD_STANDARD_FOLDER,
)

LOGGER = logging.getLogger("midas-llm")

# Repo root for resolving paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_EVALUATION_DATASET_PATH = REPO_ROOT / "resources" / "test_abstracts.json"
LEGACY_EVALUATION_DATASET_PATH = REPO_ROOT / "resources" / GOLD_STANDARD_FOLDER / "datasets" / "current.json"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "resources" / "ontologies" / "midas_data" / "midas-data.owl"
DEFAULT_GOLD_OUTPUT_DIR = (
    REPO_ROOT / DEFAULT_OUTPUT_FOLDER / GOLD_STANDARD_FOLDER / DEFAULT_RESULTS_FOLDER
)
DEFAULT_VALIDATION_SCHEMA_PATH = REPO_ROOT / "midas_schema.json"

# Compiled once, reused
_ANNOTATION_PATTERN = re.compile(r'\s*\((inferred|not mentioned)\)')


# ── NEW: JSON output prompt suffix for constrained mode ──────────────
JSON_OUTPUT_SUFFIX = """

════════════════════════════════════════════════════════════
CRITICAL OUTPUT FORMAT INSTRUCTIONS
════════════════════════════════════════════════════════════

You MUST respond with ONLY a valid JSON object. No markdown, no
backticks, no explanatory text before or after the JSON.

Use this exact structure for EVERY field:

{
  "field_name": {"values": ["value1", "value2"], "reasoning": "Why you chose these values"}
}

Rules:
- Every field MUST be present, even if empty.
- "values" is ALWAYS an array of strings. Use [] for empty.
- For binary fields (intervention_present, calibration_mentioned,
  code_available), use exactly one of: "yes", "no", "not mentioned",
  "not applicable", "not specified".
- "reasoning" is a single string. Prefix with "[Inferred]" when using
  domain knowledge not stated in the abstract.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in markdown code fences.
"""


# ------------------------------------------------------------------
# Output path helpers
# ------------------------------------------------------------------

def _sanitize_model_directory_name(model: str) -> str:
    """Build a filesystem-safe directory name from a model identifier."""
    model_leaf = model.strip().split("/")[-1]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_leaf).strip("._-")
    return sanitized or "unknown-model"


def _results_model_directory_name(config: ExtractionConfig) -> str:
    """Choose a model directory name for grouping evaluation runs."""
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]
    if not models:
        return "unknown-model"
    if len(models) == 1:
        return _sanitize_model_directory_name(models[0])
    return "multi-model"


def _parse_embedding_models_arg(value: str) -> list[str]:
    """Parse comma-separated embedding models from CLI."""
    return [model.strip() for model in value.split(",") if model.strip()]


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


def _log_active_run_configuration(
    *,
    logger: logging.Logger,
    config: ExtractionConfig,
    args: argparse.Namespace,
    use_vector_eval: bool,
    use_llm_eval: bool,
) -> None:
    """Log only configuration entries that are actively used for this run."""
    active_models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]

    logger.info("Active run configuration:")
    logger.info("  llm_api_type: %s", config.llm_api_type)
    logger.info("  active_llm_host: %s", config.active_llm_host)
    logger.info("  active_llm_models: %s", active_models)
    logger.info("  llm_timeout: %s", config.llm_timeout)
    logger.info("  constrained_decoding: %s", args.constrained)
    if args.constrained:
        logger.info("  ontology_path: %s", args.ontology_path)
    if config.llm_api_type == "openai_compatible":
        logger.info(
            "  openai_response_format: %s",
            "json_schema" if args.openai_json_schema else "json_object",
        )
    logger.info("  validate_constrained_json: %s", args.validate_constrained_json)
    if args.validate_constrained_json and args.validation_schema_path:
        logger.info("  validation_schema_path: %s", args.validation_schema_path)
    logger.info("  llm_semantic_matching: %s", use_llm_eval)
    logger.info("  vector_similarity: %s", use_vector_eval)
    if use_vector_eval:
        logger.info("  embedding_models: %s", args.embedding_models)
        logger.info("  vector_high_threshold: %.2f", args.vector_high_threshold)
        logger.info("  vector_low_threshold: %.2f", args.vector_low_threshold)


# ------------------------------------------------------------------
# Evaluation dataset format helpers
# ------------------------------------------------------------------

def get_expected_values(field_data) -> list[str]:
    """
    Return expected values from legacy and constrained gold-standard formats.

    Supported formats:
      - "yes"
      - ["a", "b"]
      - [{"value": "a"}, {"value": "b"}]   # legacy
      - {"values": ["a", "b"], "reasoning": "..."}  # constrained v3.2+
      - {"value": "a", ...}  # tolerate singleton dicts
    """
    # Legacy scalar
    if isinstance(field_data, str):
        return [field_data]

    # NEW: constrained object format
    if isinstance(field_data, dict):
        # Preferred constrained schema shape
        if "values" in field_data:
            vals = field_data.get("values", [])
            if vals is None:
                return []
            if isinstance(vals, list):
                return [str(v) for v in vals if v is not None and str(v) != ""]
            # tolerate malformed single non-list value
            return [str(vals)] if str(vals).strip() else []

        # Tolerate legacy singleton object
        if "value" in field_data:
            v = field_data.get("value")
            if v is None:
                return []
            return [str(v)] if str(v).strip() else []

        return []

    # Legacy list formats
    if isinstance(field_data, list):
        out = []
        for item in field_data:
            if isinstance(item, dict):
                if "value" in item:
                    v = item.get("value")
                    if v is not None and str(v).strip():
                        out.append(str(v))
                elif "values" in item:
                    vals = item.get("values", [])
                    if isinstance(vals, list):
                        out.extend(str(v) for v in vals if v is not None and str(v).strip())
                    elif vals is not None and str(vals).strip():
                        out.append(str(vals))
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out

    return []


def strip_annotations(values: list[str], logger: logging.Logger = LOGGER) -> list[str]:
    result = []
    for v in values:
        if not isinstance(v, str):
            logger.warning(
                "strip_annotations: non-string value, got type %s: %s",
                type(v).__name__, v,
            )
            continue
        result.append(_ANNOTATION_PATTERN.sub('', v).strip())
    return result


def normalize_term(value: str, logger: logging.Logger = LOGGER) -> str:
    if not isinstance(value, str):
        logger.warning("normalize_term: non-string value, got type %s: %s",
                        type(value).__name__, value)
        return str(value)
    return DISEASE_SYNONYMS.get(value.strip().lower(), value)


# ------------------------------------------------------------------
# Evaluation dataset loading
# ------------------------------------------------------------------

def load_evaluation_dataset(path: str | Path | None = None) -> list[dict]:
    if path is None:
        candidate = DEFAULT_EVALUATION_DATASET_PATH
        if not candidate.exists() and LEGACY_EVALUATION_DATASET_PATH.exists():
            LOGGER.warning(
                "Using legacy evaluation dataset path: %s (migrate to %s)",
                LEGACY_EVALUATION_DATASET_PATH,
                DEFAULT_EVALUATION_DATASET_PATH,
            )
            candidate = LEGACY_EVALUATION_DATASET_PATH
    else:
        candidate = Path(path)
    with open(candidate, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("abstracts", [])


# ── NEW: Constrained JSON response parser ────────────────────────────

def _extract_allowed_values(values_items_schema: dict[str, Any]) -> tuple[set[str], bool]:
    """Extract enum values and free-text allowance from a values.items schema."""
    allowed: set[str] = set()
    allows_free_text = False

    if not isinstance(values_items_schema, dict):
        return allowed, allows_free_text

    enum_vals = values_items_schema.get("enum")
    if isinstance(enum_vals, list):
        allowed.update(str(v) for v in enum_vals if v is not None)

    any_of = values_items_schema.get("anyOf")
    if isinstance(any_of, list):
        for option in any_of:
            if not isinstance(option, dict):
                continue
            option_enum = option.get("enum")
            if isinstance(option_enum, list):
                allowed.update(str(v) for v in option_enum if v is not None)
            if option.get("type") == "string":
                allows_free_text = True

    if values_items_schema.get("type") == "string":
        allows_free_text = True

    return allowed, allows_free_text


def _validate_constrained_payload_lightweight(
    payload: dict[str, Any],
    schema: dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """Best-effort schema validation when jsonschema is unavailable."""
    errors: list[str] = []
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []

    if not isinstance(payload, dict):
        logger.error("Constrained payload must be a JSON object.")
        return False

    missing = [key for key in required if key not in payload]
    for key in missing:
        errors.append(f"Missing required top-level field: {key}")

    for field_name, field_payload in payload.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            errors.append(f"Unexpected field not in schema: {field_name}")
            continue

        if not isinstance(field_payload, dict):
            errors.append(f"{field_name}: must be an object with values/reasoning")
            continue

        field_required = field_schema.get("required", [])
        for req in field_required:
            if req not in field_payload:
                errors.append(f"{field_name}: missing required key '{req}'")

        values = field_payload.get("values")
        if not isinstance(values, list):
            errors.append(f"{field_name}.values must be an array")
            continue

        for idx, val in enumerate(values):
            if not isinstance(val, str):
                errors.append(f"{field_name}.values[{idx}] must be a string")

        reasoning = field_payload.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            errors.append(f"{field_name}.reasoning must be a string")

        values_items_schema = (
            field_schema.get("properties", {})
            .get("values", {})
            .get("items", {})
        )
        allowed_values, allows_free_text = _extract_allowed_values(values_items_schema)
        if allowed_values and not allows_free_text:
            for val in values:
                if isinstance(val, str) and val not in allowed_values:
                    errors.append(
                        f"{field_name}.values contains unsupported value '{val}'"
                    )

    if errors:
        logger.error(
            "Constrained JSON failed lightweight schema validation with %d issue(s).",
            len(errors),
        )
        for msg in errors[:10]:
            logger.error("  - %s", msg)
        if len(errors) > 10:
            logger.error("  ... and %d more", len(errors) - 10)
        return False

    logger.info("Constrained JSON passed lightweight schema validation.")
    return True


def validate_constrained_payload(
    payload: dict[str, Any],
    schema: dict[str, Any],
    logger: logging.Logger = LOGGER,
) -> bool:
    """Validate constrained payload using jsonschema if available."""
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
    except ImportError:
        logger.warning(
            "jsonschema is not installed; using lightweight constrained validation."
        )
        return _validate_constrained_payload_lightweight(payload, schema, logger)

    try:
        validate(instance=payload, schema=schema)
        logger.info("Constrained JSON passed JSON Schema validation.")
        return True
    except ValidationError as e:
        logger.error("Constrained JSON failed schema validation: %s", e.message)
        return False


def parse_constrained_response(
    content: str,
    *,
    validation_schema: dict[str, Any] | None = None,
    validate_schema: bool = False,
    logger: logging.Logger = LOGGER,
) -> dict[str, dict[str, Any]]:
    """Parse JSON response from constrained decoding into the dict
    format that evaluate_extraction() expects.

    Handles common LLM quirks: markdown fences, preamble text,
    trailing commas.

    Returns:
        {field_name: {"value": "comma,separated", "reasoning": "..."}, ...}
    """
    raw = content.strip()

    # Strip markdown code fences
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw)
    raw = re.sub(r'\n?```\s*$', '', raw)

    # Strip preamble before first {
    brace_start = raw.find('{')
    if brace_start > 0:
        logger.debug("Stripped preamble: %s", raw[:brace_start][:100])
        raw = raw[brace_start:]

    # Strip postamble after last }
    brace_end = raw.rfind('}')
    if brace_end >= 0 and brace_end < len(raw) - 1:
        raw = raw[:brace_end + 1]

    # Repair trailing commas
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed for constrained response: %s", e)
        return {}

    if not isinstance(parsed, dict):
        logger.error("Parsed JSON is not a dict: %s", type(parsed))
        return {}

    if validate_schema:
        if validation_schema is None:
            logger.error(
                "Schema validation was requested but no validation schema is available."
            )
            return {}
        if not validate_constrained_payload(parsed, validation_schema, logger=logger):
            logger.error("Dropping constrained response due to schema validation failure.")
            return {}

    # Convert to evaluate_extraction format
    extracted = {}
    for field_name, field_data in parsed.items():
        if not isinstance(field_data, dict):
            extracted[field_name] = {
                "value": str(field_data),
                "reasoning": None,
                "provenance": None,
                "definition": None,
                "parent_class": None,
                "differentia": None,
                "key_relationships": None,
                "ontologies": [],
                "domains": [],
            }
            continue

        values = field_data.get("values", [])
        if isinstance(values, list):
            value_str = ", ".join(str(v) for v in values)
        else:
            value_str = str(values)

        extracted[field_name] = {
            "value": value_str,
            "reasoning": field_data.get("reasoning"),
            "provenance": field_data.get("provenance"),
            "definition": field_data.get("definition"),
            "parent_class": None,
            "differentia": None,
            "key_relationships": None,
            "ontologies": [],
            "domains": [],
        }

    logger.debug("Parsed %d fields from constrained JSON response", len(extracted))
    for attr, data in extracted.items():
        logger.debug("[%s]: %s", attr, data["value"])
        if data.get("reasoning"):
            logger.debug("   Reasoning: %s", data["reasoning"])

    return extracted
# ------------------------------------------------------------------
# Matching functions
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Core evaluation engine (unchanged)
# ------------------------------------------------------------------

def evaluate_extraction(
        extracted: dict[str, Any],
        expected: dict[str, Any],
        llm_model: str,
        llm_host: str,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_models: list[str] | None = None,
        config: Any = None,
        timeout_seconds: int = 30,
) -> dict[str, Any]:
    resolved_config = config if isinstance(config, ExtractionConfig) else ExtractionConfig.from_yaml()

    if embedding_models is None:
        embedding_models = resolved_config.embedding_models

    if not embedding_models:
        embedding_models = [resolved_config.embedding_model]

    if use_vector_eval:
        logger.info(
            "Vector similarity embedding models (%d): %s",
            len(embedding_models),
            ", ".join(embedding_models),
        )

    def vector_match_any_model(
        extracted_value: str,
        expected_values: list[str],
    ) -> tuple[str, str | None, float, list[dict[str, Any]], str | None]:
        """Return MATCH if any model crosses threshold; otherwise aggregate conservatively."""
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

    results = {
        "hits": [],
        "misses": [],
        "false_positives": [],
        "not_expected": [],
        "scores": {},
        "vector_stats": {
            "auto_matches": 0,
            "auto_rejects": 0,
            "ambiguous_to_llm": 0,
            "vector_unavailable": 0,
        },
    }

    matched_expected: dict[str, set[str]] = {attr: set() for attr in expected}

    for attr, data in extracted.items():
        logger.debug("Processing attribute: %s, data type: %s", attr, type(data).__name__)

        value = data.get("value", "") if isinstance(data, dict) else str(data)

        if attr not in expected:
            results["not_expected"].append({
                "attribute": attr,
                "extracted_value": value,
            })
            continue

        expected_values = get_expected_values(expected[attr])
        if not expected_values:
            continue

        expected_values = strip_annotations(expected_values, logger)
        expected_values = [normalize_term(v, logger) for v in expected_values]

        if not isinstance(value, str):
            logger.error("Attribute: %s - value is not a string: %s", attr, type(value).__name__)
            continue

        extracted_values = [v.strip() for v in value.split(",")]

        for ext_val in extracted_values:
            if not ext_val:
                continue

            matched = False
            matched_val = None
            similarity_score = None
            vector_model_scores: list[dict[str, Any]] = []
            vector_selected_model: str | None = None
            match_method = None

            is_date_field = attr in ("study_dates_start", "study_dates_end")

            extracted_normalized = normalize_term(ext_val, logger)
            extracted_normalized = normalize_absent_value(extracted_normalized)

            # Tier 1: Vector similarity
            decision = None
            if use_vector_eval and not is_date_field:
                try:
                    decision, best_match, similarity_score, vector_model_scores, vector_selected_model = vector_match_any_model(
                        extracted_normalized,
                        expected_values,
                    )
                except Exception as e:
                    logger.error("Vector match error for %s: %s", attr, e)
                    decision = "UNAVAILABLE"

                if decision == "MATCH":
                    matched, matched_val, match_method = True, best_match, "vector"
                    results["vector_stats"]["auto_matches"] += 1
                elif decision == "NO_MATCH":
                    if dates_match(extracted_normalized, expected_values[0] if expected_values else ""):
                        matched, matched_val, match_method = True, expected_values[0], "date_format"
                    else:
                        results["vector_stats"]["auto_rejects"] += 1
                elif decision == "AMBIGUOUS":
                    results["vector_stats"]["ambiguous_to_llm"] += 1
                else:
                    results["vector_stats"]["vector_unavailable"] += 1

                if vector_model_scores:
                    ranked_vector_scores = [
                        (idx, entry)
                        for idx, entry in enumerate(vector_model_scores)
                        if isinstance(entry, dict)
                    ]
                    ranked_vector_scores.sort(
                        key=lambda item: _vector_model_score_sort_key(item[1], item[0])
                    )
                    per_model_summary = "; ".join(
                        (
                            f"{entry['model']}:{entry['decision']}:"
                            f"{entry['similarity_score']:.4f}"
                            + (
                                f":{entry['best_match']}"
                                if entry.get("best_match")
                                else ""
                            )
                        )
                        for _, entry in ranked_vector_scores
                    )
                    logger.info(
                        "Vector similarity decision | attr=%s value=%s result=%s best_score=%.4f details=[%s]",
                        attr,
                        extracted_normalized,
                        decision or "UNAVAILABLE",
                        similarity_score or 0.0,
                        per_model_summary,
                    )

            # Tier 2: LLM semantic evaluation
            if not matched and matched_val is None and use_llm_eval:
                if not use_vector_eval or similarity_score is None or (
                        vector_low_threshold < (similarity_score or 0) < vector_high_threshold
                ):
                    try:
                        api_type = config.llm_api_type if config else "ollama"
                        matched, matched_val = evaluate_semantic_match(
                            attr, extracted_normalized, expected_values,
                            llm_model, llm_host,
                            logger=logger, api_type=api_type,
                            timeout_seconds=timeout_seconds,
                        )
                        if matched:
                            match_method = "llm"
                    except Exception as e:
                        logger.warning("LLM eval failed for '%s': %s", extracted_normalized, e)

            # Tier 3: Fallback string matching
            if not matched and matched_val is None:
                matched, matched_val = fallback_string_match(extracted_normalized, expected_values)
                if matched:
                    match_method = "string"

            if matched:
                hit_record = {
                    "attribute": attr,
                    "extracted_value": extracted_normalized,
                    "matched_expected": matched_val,
                    "match_method": match_method,
                }
                if similarity_score is not None:
                    hit_record["similarity_score"] = round(similarity_score, 4)
                if vector_model_scores:
                    hit_record["vector_model_scores"] = vector_model_scores
                if vector_selected_model:
                    hit_record["vector_selected_model"] = vector_selected_model
                results["hits"].append(hit_record)
                if matched_val:
                    matched_expected[attr].add(matched_val)
            else:
                fp_record = {
                    "attribute": attr,
                    "extracted_value": extracted_normalized,
                    "expected_values": expected_values,
                }
                if similarity_score is not None:
                    fp_record["similarity_score"] = round(similarity_score, 4)
                if vector_model_scores:
                    fp_record["vector_model_scores"] = vector_model_scores
                if vector_selected_model:
                    fp_record["vector_selected_model"] = vector_selected_model
                results["false_positives"].append(fp_record)

    # Check for misses
    for attr, raw_expected in expected.items():
        for exp_val in strip_annotations(get_expected_values(raw_expected), logger):
            normalized_exp = normalize_term(exp_val, logger)
            if normalized_exp not in matched_expected.get(attr, set()):
                results["misses"].append({
                    "attribute": attr,
                    "expected_value": exp_val,
                })

    total_expected = sum(len(get_expected_values(v)) for v in expected.values())
    total_hits = len(results["hits"])
    total_misses = len(results["misses"])
    total_false_positives = len(results["false_positives"])

    results["scores"] = {
        "total_expected": total_expected,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_false_positives": total_false_positives,
        "recall": total_hits / total_expected if total_expected > 0 else 0,
        "precision": total_hits / (total_hits + total_false_positives)
        if (total_hits + total_false_positives) > 0 else 0,
    }

    p, r = results["scores"]["precision"], results["scores"]["recall"]
    results["scores"]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return results


# ------------------------------------------------------------------
# Evaluation runner
# ------------------------------------------------------------------

def run_evaluation(
        abstracts: list[dict],
        config: ExtractionConfig,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_models: list[str] | None = None,
        run_folder: str | None = None,
        json_schema: dict[str, Any] | None = None,  # ── NEW
        openai_allow_json_schema: bool = False,
        validate_constrained_json: bool = False,
        validation_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embedding_models is None:
        embedding_models = config.embedding_models

    all_results = []
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]

    # ── NEW: Log constrained decoding status ──
    if json_schema is not None:
        logger.info(
            "Constrained decoding: ENABLED (%d schema properties)",
            len(json_schema.get("properties", {})),
        )
    else:
        logger.info("Constrained decoding: DISABLED")

    for abstract_data in abstracts:
        abstract_id = abstract_data["id"]
        title = abstract_data["title"]
        abstract = abstract_data["abstract"]
        expected = abstract_data["expected"]

        logger.info("=" * 60)
        logger.info("Evaluating: %s", title[:60])
        logger.info("=" * 60)

        query = build_query(
            abstract,
            include_examples=config.prompt_include_format_examples,
            include_reminders=config.prompt_include_reminders,
            include_few_shot=config.prompt_include_few_shot,
            include_fields=config.prompt_include_fields,
            include_ontologies=config.prompt_include_ontologies,
            simple_prompt=config.prompt_simple_prompt,
        )
        prompt = prepare_and_display_prompt(query, "", logger=logger)

        # ── NEW: Append JSON instructions when using constrained mode ──
        if json_schema is not None:
            prompt = prompt + JSON_OUTPUT_SUFFIX

        if run_folder:
            prompt_file = Path(run_folder) / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            logger.info("Saved prompt to: %s", prompt_file)

        abstract_results = {
            "id": abstract_id,
            "title": title,
            "models": {},
        }

        if run_folder:
            abstract_dir = Path(run_folder) / abstract_id
            os.makedirs(abstract_dir, exist_ok=True)
            logger.info("Created abstract directory: %s", abstract_dir)

        for model in models:
            logger.info("Running model: %s", model)

            try:
                # ── NEW: Pass json_schema to send_to_llm ──
                response = send_to_llm(
                    prompt=prompt,
                    llm_model=model,
                    llm_host=config.active_llm_host,
                    timeout_seconds=config.llm_timeout,
                    logger=logger,
                    api_type=config.llm_api_type,
                    json_schema=json_schema,  # None → unconstrained
                    allow_json_schema=openai_allow_json_schema,
                )
                timing = {
                    "request_duration_s": response.request_duration_s,
                    "total_duration_s": response.total_duration_s,
                    "load_duration_s": response.load_duration_s,
                    "prompt_eval_duration_s": response.prompt_eval_duration_s,
                    "eval_duration_s": response.eval_duration_s,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                }
                logger.info(
                    "LLM timing | request=%.2fs backend_total=%s prompt_eval=%s eval=%s",
                    timing["request_duration_s"] or 0.0,
                    (
                        f"{timing['total_duration_s']:.2f}s"
                        if isinstance(timing["total_duration_s"], (int, float))
                        else "n/a"
                    ),
                    (
                        f"{timing['prompt_eval_duration_s']:.2f}s"
                        if isinstance(timing["prompt_eval_duration_s"], (int, float))
                        else "n/a"
                    ),
                    (
                        f"{timing['eval_duration_s']:.2f}s"
                        if isinstance(timing["eval_duration_s"], (int, float))
                        else "n/a"
                    ),
                )

                if response.constrained:
                    logger.info("Parsing constrained JSON response")
                else:
                    logger.warning(
                        "Response was not marked constrained; attempting strict JSON parse."
                    )
                extracted = parse_constrained_response(
                    response.content,
                    validation_schema=validation_schema,
                    validate_schema=validate_constrained_json,
                    logger=logger,
                )
                if not extracted:
                    raise ValueError(
                        "Failed to parse constrained JSON response. "
                        "Free-text responses are unsupported."
                    )

                logger.info("Evaluating extraction with semantic matching...")
                evaluation = evaluate_extraction(
                    extracted, expected,
                    model, config.active_llm_host, logger,
                    use_llm_eval=use_llm_eval,
                    use_vector_eval=use_vector_eval,
                    vector_high_threshold=vector_high_threshold,
                    vector_low_threshold=vector_low_threshold,
                    embedding_models=embedding_models,
                    config=config,
                    timeout_seconds=config.llm_timeout,
                )

                # Save artifacts
                if run_folder:
                    safe_model_name = model.replace("/", "-").replace("\\", "-")

                    response_file = abstract_dir / f"{safe_model_name}_response.json"
                    response_file.write_text(response.content, encoding="utf-8")

                    # ── NEW: Save constrained status ──
                    meta_file = abstract_dir / f"{safe_model_name}_meta.json"
                    meta_file.write_text(json.dumps({
                        "model": model,
                        "constrained": response.constrained,
                        "response_length": len(response.content),
                        "timing": timing,
                    }, indent=2), encoding="utf-8")

                abstract_results["models"][model] = {
                    "extracted": extracted,
                    "evaluation": evaluation,
                    "raw_response": response.content[:2000],
                    "constrained": response.constrained,  # ── NEW
                    "timing": timing,
                }

                # Log legend
                logger.info("")
                logger.info("EVALUATION LEGEND:")
                logger.info("  HIT = LLM extracted a value that matches something in the evaluation dataset")
                logger.info("  MISS = Evaluation dataset expects a value, but LLM didn't extract it")
                logger.info("  FALSE POSITIVE = LLM extracted a value not in the evaluation dataset")
                logger.info("")

                # Log vector stats
                vector_stats = evaluation.get("vector_stats", {})
                if use_vector_eval and any(vector_stats.values()):
                    logger.info("VECTOR EVALUATION STATS:")
                    logger.info("  Auto-matches (score >= %.2f): %d",
                                vector_high_threshold, vector_stats.get("auto_matches", 0))
                    logger.info("  Auto-rejects (score <= %.2f): %d",
                                vector_low_threshold, vector_stats.get("auto_rejects", 0))
                    logger.info("  Ambiguous -> LLM: %d", vector_stats.get("ambiguous_to_llm", 0))
                    if vector_stats.get("vector_unavailable", 0) > 0:
                        logger.info("  Vector unavailable: %d", vector_stats.get("vector_unavailable", 0))
                    logger.info("")

                # Log summary
                scores = evaluation["scores"]
                logger.info(
                    "Model %s: Recall=%.2f, Precision=%.2f, F1=%.2f "
                    "(hits=%d, misses=%d, fp=%d, constrained=%s)",
                    model, scores["recall"], scores["precision"], scores["f1"],
                    scores["total_hits"], scores["total_misses"],
                    scores["total_false_positives"], response.constrained,
                )

                # Log detailed results
                if evaluation["hits"]:
                    logger.info("  HITS (LLM extracted value matches evaluation dataset):")
                    for hit in evaluation["hits"]:
                        score_str = f" [sim={hit['similarity_score']:.3f}]" if "similarity_score" in hit else ""
                        method_str = f" ({hit['match_method']})" if "match_method" in hit else ""
                        vector_model_str = (
                            f" [vector_model={hit['vector_selected_model']}]"
                            if hit.get("vector_selected_model")
                            else ""
                        )
                        vector_details = _format_vector_model_scores(hit.get("vector_model_scores"))
                        vector_details_str = (
                            f" [vector_details={vector_details}]"
                            if vector_details
                            else ""
                        )
                        logger.info("    + %s: LLM='%s' -> matched dataset='%s'%s%s%s%s",
                                    hit["attribute"], hit["extracted_value"],
                                    hit["matched_expected"], method_str, score_str,
                                    vector_model_str, vector_details_str)

                if evaluation["misses"]:
                    logger.info("  MISSES (dataset value NOT extracted by LLM):")
                    for miss in evaluation["misses"]:
                        logger.info("    x %s: dataset='%s' was not extracted",
                                    miss["attribute"], miss["expected_value"])

                if evaluation["false_positives"]:
                    logger.info("  FALSE POSITIVES (LLM extracted value NOT in evaluation dataset):")
                    for fp in evaluation["false_positives"]:
                        score_str = f" [best_sim={fp['similarity_score']:.3f}]" if "similarity_score" in fp else ""
                        vector_model_str = (
                            f" [vector_model={fp['vector_selected_model']}]"
                            if fp.get("vector_selected_model")
                            else ""
                        )
                        vector_details = _format_vector_model_scores(fp.get("vector_model_scores"))
                        vector_details_str = (
                            f" [vector_details={vector_details}]"
                            if vector_details
                            else ""
                        )
                        logger.info("    ? %s: LLM='%s' not in gold=%s%s%s%s",
                                    fp["attribute"], fp["extracted_value"],
                                    fp["expected_values"], score_str,
                                    vector_model_str, vector_details_str)

            except Exception as e:
                logger.error("Model %s failed: %s", model, e)
                abstract_results["models"][model] = {"error": str(e)}

            # Save per-abstract evaluation report
            if run_folder and "error" not in abstract_results["models"].get(model, {}):
                generate_abstract_evaluation_report(
                    abstract_id=abstract_id,
                    title=title,
                    model=model,
                    evaluation=evaluation,
                    output_dir=str(abstract_dir),
                    logger=logger,
                )

        all_results.append(abstract_results)

    return {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "abstracts": all_results,
    }


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def _add_optional_float(accumulator: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, (int, float)):
        accumulator[key] += float(value)
        accumulator[f"{key}_count"] += 1


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def print_summary(results: dict, logger: logging.Logger) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    model_scores = {}

    for abstract_result in results["abstracts"]:
        for model, model_result in abstract_result["models"].items():
            if "error" in model_result:
                continue

            if model not in model_scores:
                model_scores[model] = {
                    "total_hits": 0,
                    "total_misses": 0,
                    "total_false_positives": 0,
                    "total_expected": 0,
                    "abstracts_evaluated": 0,
                    "request_duration_s": 0.0,
                    "request_duration_s_count": 0,
                    "total_duration_s": 0.0,
                    "total_duration_s_count": 0,
                    "load_duration_s": 0.0,
                    "load_duration_s_count": 0,
                    "prompt_eval_duration_s": 0.0,
                    "prompt_eval_duration_s_count": 0,
                    "eval_duration_s": 0.0,
                    "eval_duration_s_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                }

            scores = model_result["evaluation"]["scores"]
            model_scores[model]["total_hits"] += scores["total_hits"]
            model_scores[model]["total_misses"] += scores["total_misses"]
            model_scores[model]["total_false_positives"] += scores["total_false_positives"]
            model_scores[model]["total_expected"] += scores["total_expected"]
            model_scores[model]["abstracts_evaluated"] += 1
            timing = model_result.get("timing", {})
            if isinstance(timing, dict):
                _add_optional_float(
                    model_scores[model], "request_duration_s", timing.get("request_duration_s")
                )
                _add_optional_float(
                    model_scores[model], "total_duration_s", timing.get("total_duration_s")
                )
                _add_optional_float(
                    model_scores[model], "load_duration_s", timing.get("load_duration_s")
                )
                _add_optional_float(
                    model_scores[model], "prompt_eval_duration_s",
                    timing.get("prompt_eval_duration_s"),
                )
                _add_optional_float(
                    model_scores[model], "eval_duration_s", timing.get("eval_duration_s")
                )
                if isinstance(timing.get("prompt_tokens"), int):
                    model_scores[model]["prompt_tokens"] += timing["prompt_tokens"]
                if isinstance(timing.get("completion_tokens"), int):
                    model_scores[model]["completion_tokens"] += timing["completion_tokens"]
                if isinstance(timing.get("reasoning_tokens"), int):
                    model_scores[model]["reasoning_tokens"] += timing["reasoning_tokens"]

    for model, scores in model_scores.items():
        total = scores["total_hits"] + scores["total_false_positives"]
        recall = scores["total_hits"] / scores["total_expected"] if scores["total_expected"] > 0 else 0
        precision = scores["total_hits"] / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logger.info("")
        logger.info("Model: %s", model)
        logger.info("  Abstracts evaluated: %d", scores["abstracts_evaluated"])
        logger.info("  Total expected values: %d", scores["total_expected"])
        logger.info("  Hits: %d", scores["total_hits"])
        logger.info("  Misses: %d", scores["total_misses"])
        logger.info("  False positives: %d", scores["total_false_positives"])
        logger.info("  Recall: %.2f%%", recall * 100)
        logger.info("  Precision: %.2f%%", precision * 100)
        logger.info("  F1 Score: %.2f%%", f1 * 100)

        avg_request = (
            scores["request_duration_s"] / scores["request_duration_s_count"]
            if scores["request_duration_s_count"] > 0 else None
        )
        avg_total = (
            scores["total_duration_s"] / scores["total_duration_s_count"]
            if scores["total_duration_s_count"] > 0 else None
        )
        avg_prompt_eval = (
            scores["prompt_eval_duration_s"] / scores["prompt_eval_duration_s_count"]
            if scores["prompt_eval_duration_s_count"] > 0 else None
        )
        avg_eval = (
            scores["eval_duration_s"] / scores["eval_duration_s_count"]
            if scores["eval_duration_s_count"] > 0 else None
        )
        avg_load = (
            scores["load_duration_s"] / scores["load_duration_s_count"]
            if scores["load_duration_s_count"] > 0 else None
        )
        total_request = (
            scores["request_duration_s"]
            if scores["request_duration_s_count"] > 0 else None
        )
        total_backend = (
            scores["total_duration_s"]
            if scores["total_duration_s_count"] > 0 else None
        )
        total_prompt_eval = (
            scores["prompt_eval_duration_s"]
            if scores["prompt_eval_duration_s_count"] > 0 else None
        )
        total_eval = (
            scores["eval_duration_s"]
            if scores["eval_duration_s_count"] > 0 else None
        )
        total_load = (
            scores["load_duration_s"]
            if scores["load_duration_s_count"] > 0 else None
        )

        logger.info("  Timing:")
        logger.info(
            "    Request round-trip: total=%s avg=%s over %d call(s)",
            _fmt_seconds(total_request),
            _fmt_seconds(avg_request),
            scores["request_duration_s_count"],
        )
        logger.info(
            "    Backend total: total=%s avg=%s over %d call(s)",
            _fmt_seconds(total_backend),
            _fmt_seconds(avg_total),
            scores["total_duration_s_count"],
        )
        logger.info(
            "    Prompt eval (input processing): total=%s avg=%s over %d call(s)",
            _fmt_seconds(total_prompt_eval),
            _fmt_seconds(avg_prompt_eval),
            scores["prompt_eval_duration_s_count"],
        )
        logger.info(
            "    Eval/generation (\"thinking\"): total=%s avg=%s over %d call(s)",
            _fmt_seconds(total_eval),
            _fmt_seconds(avg_eval),
            scores["eval_duration_s_count"],
        )
        logger.info(
            "    Model load: total=%s avg=%s over %d call(s)",
            _fmt_seconds(total_load),
            _fmt_seconds(avg_load),
            scores["load_duration_s_count"],
        )
        logger.info(
            "    Tokens: prompt=%d completion=%d reasoning=%d",
            scores["prompt_tokens"],
            scores["completion_tokens"],
            scores["reasoning_tokens"],
        )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM extraction against an evaluation dataset"
    )
    parser.add_argument("-n", "--num-papers", type=int, default=20,
                        help="Number of papers to evaluate (default: 1, use -1 for all)")
    parser.add_argument("--paper-id", type=str, default=None,
                        help="Evaluate a specific paper by ID")
    parser.add_argument(
        "--evaluation-dataset-path",
        "--gold-standard-path",
        dest="evaluation_dataset_path",
        type=str,
        default=str(DEFAULT_EVALUATION_DATASET_PATH),
        help=(
            "Path to evaluation dataset JSON "
            f"(default: {DEFAULT_EVALUATION_DATASET_PATH})"
        ),
    )
    parser.add_argument("--no-llm-eval", action="store_true",
                        help="Use string matching instead of LLM semantic evaluation")
    parser.add_argument("--no-vector-eval", action="store_true",
                        help="Disable vector similarity evaluation")
    parser.add_argument("--vector-high-threshold", type=float, default=0.85,
                        help="Vector similarity threshold for auto-match (default: 0.85)")
    parser.add_argument("--vector-low-threshold", type=float, default=0.50,
                        help="Vector similarity threshold for auto-reject (default: 0.50)")
    parser.add_argument(
        "--embedding-models",
        type=str,
        default=None,
        help=(
            "Comma-separated HuggingFace embedding models for vector similarity "
            "(default: use config embedding_models array)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Deprecated alias for a single embedding model.",
    )
    parser.add_argument("--list-papers", action="store_true",
                        help="List available paper IDs and exit")
    parser.add_argument("--validate-thresholds", action="store_true",
                        help="Run domain validation to find optimal thresholds and exit")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_GOLD_OUTPUT_DIR),
        help=(
            "Output root directory for timestamped evaluation runs "
            f"(default: {DEFAULT_GOLD_OUTPUT_DIR})"
        ),
    )

    # ── NEW: Constrained decoding flags ──
    parser.add_argument(
        "--constrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable constrained decoding with MIDAS structured_vocab JSON schema (default: enabled)",
    )
    parser.add_argument(
        "--structured_vocab-path",
        "--ontology-path",
        dest="ontology_path",
        type=str,
        default=str(DEFAULT_ONTOLOGY_PATH),
        help=f"Path to MIDAS OWL structured_vocab file (default: {DEFAULT_ONTOLOGY_PATH})",
    )
    parser.add_argument(
        "--validate-constrained-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Validate constrained JSON responses against schema "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--validation-schema-path",
        type=str,
        default=None,
        help=(
            "Optional JSON schema file path used for constrained response "
            f"validation (default fallback: {DEFAULT_VALIDATION_SCHEMA_PATH})"
        ),
    )
    parser.add_argument(
        "--openai-json-schema",
        action="store_true",
        default=False,
        help=(
            "For openai_compatible API type, attempt response_format=json_schema. "
            "Default is json_object mode."
        ),
    )

    args = parser.parse_args()

    config = ExtractionConfig.from_yaml()
    logger = configure_logging()

    if args.embedding_models:
        args.embedding_models = _parse_embedding_models_arg(args.embedding_models)
    elif args.embedding_model:
        args.embedding_models = _parse_embedding_models_arg(args.embedding_model)
    else:
        args.embedding_models = config.embedding_models

    # Handle --validate-thresholds
    if args.validate_thresholds:
        from concept_extractor.utils.evaluation.vector_similarity import run_domain_validation
        validation_model = args.embedding_models[0]
        logger.info("Running threshold validation with domain-specific test pairs...")
        logger.info("Threshold validation model: %s", validation_model)
        validation_results = run_domain_validation(model_name=validation_model, logger=logger)

        if "error" in validation_results:
            logger.error("Validation failed: %s", validation_results["error"])
            return

        print("\n" + "=" * 60)
        print("THRESHOLD VALIDATION RESULTS")
        print("=" * 60)
        print(f"\nEmbedding Model: {validation_results['model']}")
        print(f"Optimal Threshold: {validation_results['optimal_threshold']:.2f}")
        opt = validation_results["optimal_metrics"]
        print(f"\nAt optimal threshold:")
        print(f"  Precision: {opt['precision']:.2%}")
        print(f"  Recall:    {opt['recall']:.2%}")
        print(f"  F1 Score:  {opt['f1']:.2%}")
        print(f"  Accuracy:  {opt['accuracy']:.2%}")

        print("\n" + "-" * 60)
        print("Threshold sweep results:")
        for key, metrics in sorted(validation_results.get("threshold_results", {}).items()):
            t = metrics.get("threshold", 0)
            print(f"  {t:.2f}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

        print("\n" + "-" * 60)
        print("Individual pair results at optimal threshold:")
        for pair in validation_results.get("pair_results", []):
            status = "+" if pair["correct"] else "x"
            match_str = "should match" if pair["expected_match"] else "should NOT match"
            print(f"  {status} '{pair['text_a']}' <-> '{pair['text_b']}' ({match_str})")
            print(f"      similarity={pair['similarity']:.3f}, "
                  f"predicted={'MATCH' if pair['predicted_match'] else 'NO_MATCH'}")
        return

    # Log resolved path-related options used for this run
    evaluation_dataset_path = Path(args.evaluation_dataset_path)
    output_root_path = Path(args.output_dir)
    logger.info("Resolved run paths/options:")
    logger.info("  repo_root: %s", REPO_ROOT)
    logger.info(
        "  evaluation_dataset_path: %s%s",
        evaluation_dataset_path,
        " (default)" if evaluation_dataset_path == DEFAULT_EVALUATION_DATASET_PATH
        else (" (legacy)" if evaluation_dataset_path == LEGACY_EVALUATION_DATASET_PATH else " (override)"),
    )
    logger.info(
        "  output_dir: %s%s",
        output_root_path,
        " (default)" if output_root_path == DEFAULT_GOLD_OUTPUT_DIR else " (override)",
    )
    if args.constrained:
        ontology_path = Path(args.ontology_path)
        logger.info(
            "  ontology_path: %s%s",
            ontology_path,
            " (default)" if ontology_path == DEFAULT_ONTOLOGY_PATH else " (override)",
        )
    if args.validate_constrained_json:
        if args.validation_schema_path:
            logger.info("  validation_schema_path: %s (override)", Path(args.validation_schema_path))
        else:
            logger.info("  validation_schema_path: <runtime structured_vocab schema>")

    # Load evaluation dataset
    try:
        all_abstracts = load_evaluation_dataset(args.evaluation_dataset_path)
    except FileNotFoundError:
        logger.error("Evaluation dataset not found: %s", args.evaluation_dataset_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Invalid evaluation dataset JSON at %s: %s", args.evaluation_dataset_path, e)
        return

    if args.list_papers:
        print("Available paper IDs:")
        for abstract in all_abstracts:
            print(f"  {abstract['id']}: {abstract['title'][:60]}...")
        return

    logger.info("Starting evaluation...")

    detected_host = autodetect_llm_host(
        config.active_llm_host,
        api_type=config.llm_api_type,
        logger=logger,
    )
    if detected_host:
        config.active_llm_host = detected_host

    use_llm_eval = not args.no_llm_eval
    use_vector_eval = not args.no_vector_eval

    if config.show_config:
        _log_active_run_configuration(
            logger=logger,
            config=config,
            args=args,
            use_vector_eval=use_vector_eval,
            use_llm_eval=use_llm_eval,
        )

    # ── NEW: Load MIDAS structured_vocab for constrained decoding ──
    json_schema = None
    if args.constrained:
        try:
            from concept_extractor.utils.structured_vocab.midas_vocabulary import MIDASVocabulary

            owl_path = args.ontology_path or str(DEFAULT_ONTOLOGY_PATH)
            logger.info("Loading MIDAS structured_vocab from: %s", owl_path)
            vocab = MIDASVocabulary.from_owl(owl_path)
            json_schema = vocab.build_json_schema()

            logger.info(
                "MIDAS structured_vocab loaded: %d schema properties",
                len(json_schema.get("properties", {})),
            )
            logger.info("Constrained decoding: ENABLED")

            # Also append structured_vocab vocabulary to prompt
            # (helps even if constrained decoding falls back)
            vocab_text = vocab.build_prompt_section()
            logger.info("Ontology vocabulary section: %d chars", len(vocab_text))

        except FileNotFoundError:
            logger.error(
                "Ontology file not found: %s — run with --no-constrained "
                "or download with:\n"
                "  curl -o resources/ontologies/midas_data/midas-data.owl "
                "https://raw.githubusercontent.com/midas-network/midas-data/"
                "refs/heads/main/midas-data.owl",
                owl_path,
            )
            return
        except ImportError:
            logger.error(
                "midas_vocabulary module not found. Ensure "
                "midas_vocabulary.py is in src/concept_extractor/utils/structured_vocab/"
            )
            return
        except Exception as e:
            logger.error("Failed to load MIDAS structured_vocab: %s", e)
            return
    else:
        logger.error(
            "Free-text parsing is disabled. Run with --constrained "
            "(or remove --no-constrained)."
        )
        return

    validation_schema = None
    resolved_validation_schema_path = None
    if args.validate_constrained_json:
        if args.validation_schema_path:
            candidate = Path(args.validation_schema_path)
            if not candidate.exists():
                logger.error("Validation schema file not found: %s", candidate)
                return
            try:
                validation_schema = json.loads(candidate.read_text(encoding="utf-8"))
                resolved_validation_schema_path = str(candidate)
                logger.info(
                    "Constrained response validation enabled using schema file: %s",
                    candidate,
                )
            except json.JSONDecodeError as e:
                logger.error("Invalid schema JSON at %s: %s", candidate, e)
                return
        elif json_schema is not None:
            validation_schema = json_schema
            logger.info(
                "Constrained response validation enabled using runtime structured_vocab schema."
            )
        else:
            if DEFAULT_VALIDATION_SCHEMA_PATH.exists():
                try:
                    validation_schema = json.loads(
                        DEFAULT_VALIDATION_SCHEMA_PATH.read_text(encoding="utf-8")
                    )
                    resolved_validation_schema_path = str(DEFAULT_VALIDATION_SCHEMA_PATH)
                    logger.info(
                        "Constrained response validation enabled using schema file: %s",
                        DEFAULT_VALIDATION_SCHEMA_PATH,
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid schema JSON at %s: %s",
                        DEFAULT_VALIDATION_SCHEMA_PATH,
                        e,
                    )
                    return
            else:
                logger.error(
                    "Constrained response validation requested, but no schema found. "
                    "Set --validation-schema-path or ensure %s exists.",
                    DEFAULT_VALIDATION_SCHEMA_PATH,
                )
                return

    # Filter abstracts
    if args.paper_id:
        abstracts = [a for a in all_abstracts if a["id"] == args.paper_id]
        if not abstracts:
            logger.error("Paper ID '%s' not found. Use --list-papers.", args.paper_id)
            return
    elif args.num_papers == -1:
        abstracts = all_abstracts
    else:
        abstracts = all_abstracts[:args.num_papers]

    logger.info("Evaluating %d of %d dataset abstracts", len(abstracts), len(all_abstracts))

    if use_vector_eval:
        logger.info(
            "Vector similarity evaluation: ENABLED (models=%s)",
            ", ".join(args.embedding_models),
        )
        logger.info("  Auto-match threshold: >= %.2f", args.vector_high_threshold)
        logger.info("  Auto-reject threshold: <= %.2f", args.vector_low_threshold)
    else:
        logger.info("Vector similarity evaluation: DISABLED")

    if use_llm_eval:
        logger.info("LLM semantic matching: ENABLED")
    else:
        logger.info("LLM semantic matching: DISABLED")
    if config.llm_api_type == "openai_compatible":
        logger.info(
            "OpenAI response_format mode: %s",
            "json_schema" if args.openai_json_schema else "json_object",
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = _results_model_directory_name(config)
    run_root = Path(args.output_dir) / model_dir
    run_folder = str(run_root / timestamp)
    os.makedirs(run_folder, exist_ok=True)
    logger.info("Run folder: %s", run_folder)

    # ── NEW: Save schema to run folder for reproducibility ──
    if json_schema is not None:
        schema_file = os.path.join(run_folder, "constrained_schema.json")
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(json_schema, f, indent=2)
        logger.info("Saved constrained schema to: %s", schema_file)

    results = run_evaluation(
        abstracts, config, logger,
        use_llm_eval=use_llm_eval,
        use_vector_eval=use_vector_eval,
        vector_high_threshold=args.vector_high_threshold,
        vector_low_threshold=args.vector_low_threshold,
        embedding_models=args.embedding_models,
        run_folder=run_folder,
        json_schema=json_schema,  # ── NEW: passed through
        openai_allow_json_schema=args.openai_json_schema,
        validate_constrained_json=args.validate_constrained_json,
        validation_schema=validation_schema,
    )

    results["evaluation_config"] = {
        "use_vector_eval": use_vector_eval,
        "use_llm_eval": use_llm_eval,
        "vector_high_threshold": args.vector_high_threshold,
        "vector_low_threshold": args.vector_low_threshold,
        "embedding_models": args.embedding_models,
        "embedding_model": args.embedding_models[0] if args.embedding_models else None,  # backward-compatible key
        "constrained": args.constrained,  # ── NEW
        "ontology_path": args.ontology_path,  # ── NEW
        "evaluation_dataset_path": args.evaluation_dataset_path,
        "output_dir": args.output_dir,
        "validate_constrained_json": args.validate_constrained_json,
        "openai_json_schema": args.openai_json_schema,
        "validation_schema_path": (
            args.validation_schema_path
            or resolved_validation_schema_path
            or ("<runtime_ontology_schema>" if validation_schema is json_schema and json_schema is not None else None)
        ),
    }

    output_file = os.path.join(run_folder, "evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    report_file = generate_evaluation_text_report(
        results, run_folder=run_folder, logger=logger,
    )
    logger.info("Human-readable evaluation report: %s", report_file)

    print_summary(results, logger)


if __name__ == "__main__":
    main()
