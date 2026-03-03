#!/usr/bin/env python
"""Gold standard evaluation script for LLM extraction accuracy.

Compares extracted attributes against expected values using LLM-based semantic matching.
Supports constrained decoding via MIDAS ontology JSON schema (--constrained flag).
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

from src.midas_llm.data.IdSynonyms import DISEASE_SYNONYMS
from src.midas_llm.utils.parsers.extraction_parser import normalize_absent_value

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.config import ExtractionConfig
from midas_llm.utils.logging.logger import configure_logging
from midas_llm.utils.llm.llm_client import send_to_llm
from midas_llm.utils.llm.llm_utils import autodetect_llm_host
from midas_llm.utils.prompt.builders import prepare_and_display_prompt, build_query
from midas_llm.utils.parsers.extraction_parser import (
    parse_and_display_extracted_data,
    save_response_json,
    normalize_llm_format,
)
from midas_llm.utils.reporting.evaluation_reports import (
    generate_evaluation_text_report,
    generate_evaluation_html_report,
    generate_abstract_evaluation_report,
)
from midas_llm.utils.evaluation.vector_similarity import vector_match_tiered

LOGGER = logging.getLogger("midas-llm")

# Repo root for resolving paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_GOLD_STANDARD_PATH = REPO_ROOT / "resources" / "gold_standard" / "datasets" / "current.json"
LEGACY_GOLD_STANDARD_PATH = REPO_ROOT / "resources" / "test_abstracts.json"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "resources" / "ontologies" / "midas_data" / "midas-data.owl"
DEFAULT_GOLD_OUTPUT_DIR = REPO_ROOT / "output" / "gold_standard" / "results"
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
# Gold standard format helpers
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
# Gold standard loading
# ------------------------------------------------------------------

def load_gold_standard(path: str | Path | None = None) -> list[dict]:
    if path is None:
        candidate = DEFAULT_GOLD_STANDARD_PATH
        if not candidate.exists() and LEGACY_GOLD_STANDARD_PATH.exists():
            LOGGER.warning(
                "Using legacy gold standard path: %s (migrate to %s)",
                LEGACY_GOLD_STANDARD_PATH,
                DEFAULT_GOLD_STANDARD_PATH,
            )
            candidate = LEGACY_GOLD_STANDARD_PATH
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
        logger.warning("JSON parse failed: %s — falling back to regex", e)
        parsed = _regex_json_fallback(raw, logger)

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

    logger.info("Parsed %d fields from constrained JSON response", len(extracted))
    for attr, data in extracted.items():
        logger.info("[%s]: %s", attr, data["value"])
        if data.get("reasoning"):
            logger.info("   Reasoning: %s", data["reasoning"])

    return extracted


def _regex_json_fallback(raw: str, logger: logging.Logger) -> dict:
    """Last-resort extraction when JSON parsing fails."""
    result = {}
    pattern = re.compile(
        r'"(\w+)"\s*:\s*\{[^}]*"values"\s*:\s*\[([^\]]*)\]',
        re.DOTALL,
    )
    for match in pattern.finditer(raw):
        field_name = match.group(1)
        values = re.findall(r'"([^"]*)"', match.group(2))
        result[field_name] = {"values": values}

    if result:
        logger.info("Regex fallback recovered %d fields", len(result))
    else:
        logger.error("Regex fallback also failed")
    return result


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
        embedding_model: str | None = None,
        config: Any = None,
        timeout_seconds: int = 30,
) -> dict[str, Any]:
    if embedding_model is None:
        embedding_model = ExtractionConfig().embedding_model

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
            match_method = None

            is_date_field = attr in ("study_dates_start", "study_dates_end")

            extracted_normalized = normalize_term(ext_val, logger)
            extracted_normalized = normalize_absent_value(extracted_normalized)

            # Tier 1: Vector similarity
            decision = None
            if use_vector_eval and not is_date_field:
                try:
                    decision, best_match, similarity_score = vector_match_tiered(
                        extracted_normalized, expected_values,
                        high_threshold=vector_high_threshold,
                        low_threshold=vector_low_threshold,
                        model_name=embedding_model,
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
        embedding_model: str | None = None,
        run_folder: str | None = None,
        json_schema: dict[str, Any] | None = None,  # ── NEW
        validate_constrained_json: bool = False,
        validation_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embedding_model is None:
        embedding_model = config.embedding_model

    all_results = []
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]

    # ── NEW: Log constrained decoding status ──
    if json_schema is not None:
        logger.info(
            "Constrained decoding: ENABLED (%d schema properties)",
            len(json_schema.get("properties", {})),
        )
    else:
        logger.info("Constrained decoding: DISABLED (free-text mode)")

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
                )

                # ── NEW: Branch on constrained vs unconstrained ──
                if response.constrained:
                    logger.info("Parsing constrained JSON response")
                    extracted = parse_constrained_response(
                        response.content,
                        validation_schema=validation_schema,
                        validate_schema=validate_constrained_json,
                        logger=logger,
                    )
                else:
                    if json_schema is not None:
                        logger.warning(
                            "Constrained decoding was requested but server "
                            "fell back to unconstrained — using text parser"
                        )
                    extracted = parse_and_display_extracted_data(
                        response.content, logger=logger
                    )

                logger.info("Evaluating extraction with semantic matching...")
                evaluation = evaluate_extraction(
                    extracted, expected,
                    model, config.active_llm_host, logger,
                    use_llm_eval=use_llm_eval,
                    use_vector_eval=use_vector_eval,
                    vector_high_threshold=vector_high_threshold,
                    vector_low_threshold=vector_low_threshold,
                    embedding_model=embedding_model,
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
                    }, indent=2), encoding="utf-8")

                    if not response.constrained:
                        normalized = normalize_llm_format(response.content)
                        normalized_file = abstract_dir / f"{safe_model_name}_response_normalized.txt"
                        normalized_file.write_text(normalized, encoding="utf-8")


                abstract_results["models"][model] = {
                    "extracted": extracted,
                    "evaluation": evaluation,
                    "raw_response": response.content[:2000],
                    "constrained": response.constrained,  # ── NEW
                }

                # Log legend
                logger.info("")
                logger.info("EVALUATION LEGEND:")
                logger.info("  HIT = LLM extracted a value that matches something in the gold standard")
                logger.info("  MISS = Gold standard expects a value, but LLM didn't extract it")
                logger.info("  FALSE POSITIVE = LLM extracted a value not in the gold standard")
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
                    logger.info("  HITS (LLM extracted value matches gold standard):")
                    for hit in evaluation["hits"]:
                        score_str = f" [sim={hit['similarity_score']:.3f}]" if "similarity_score" in hit else ""
                        method_str = f" ({hit['match_method']})" if "match_method" in hit else ""
                        logger.info("    + %s: LLM='%s' -> matched gold='%s'%s%s",
                                    hit["attribute"], hit["extracted_value"],
                                    hit["matched_expected"], method_str, score_str)

                if evaluation["misses"]:
                    logger.info("  MISSES (gold standard value NOT extracted by LLM):")
                    for miss in evaluation["misses"]:
                        logger.info("    x %s: gold='%s' was not extracted",
                                    miss["attribute"], miss["expected_value"])

                if evaluation["false_positives"]:
                    logger.info("  FALSE POSITIVES (LLM extracted value NOT in gold standard):")
                    for fp in evaluation["false_positives"]:
                        score_str = f" [best_sim={fp['similarity_score']:.3f}]" if "similarity_score" in fp else ""
                        logger.info("    ? %s: LLM='%s' not in gold=%s%s",
                                    fp["attribute"], fp["extracted_value"],
                                    fp["expected_values"], score_str)

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
                }

            scores = model_result["evaluation"]["scores"]
            model_scores[model]["total_hits"] += scores["total_hits"]
            model_scores[model]["total_misses"] += scores["total_misses"]
            model_scores[model]["total_false_positives"] += scores["total_false_positives"]
            model_scores[model]["total_expected"] += scores["total_expected"]
            model_scores[model]["abstracts_evaluated"] += 1

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


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM extraction against gold standard"
    )
    parser.add_argument("-n", "--num-papers", type=int, default=1,
                        help="Number of papers to evaluate (default: 1, use -1 for all)")
    parser.add_argument("--paper-id", type=str, default=None,
                        help="Evaluate a specific paper by ID")
    parser.add_argument(
        "--gold-standard-path",
        type=str,
        default=str(DEFAULT_GOLD_STANDARD_PATH),
        help=(
            "Path to gold standard JSON dataset "
            f"(default: {DEFAULT_GOLD_STANDARD_PATH})"
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
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="HuggingFace embedding model for vector similarity")
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
        help="Enable constrained decoding with MIDAS ontology JSON schema (default: enabled)",
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default=str(DEFAULT_ONTOLOGY_PATH),
        help=f"Path to MIDAS OWL ontology file (default: {DEFAULT_ONTOLOGY_PATH})",
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

    args = parser.parse_args()

    config = ExtractionConfig()
    logger = configure_logging(debug=config.debug)

    if args.embedding_model is None:
        args.embedding_model = config.embedding_model

    # Handle --validate-thresholds
    if args.validate_thresholds:
        from midas_llm.utils.evaluation.vector_similarity import run_domain_validation
        logger.info("Running threshold validation with domain-specific test pairs...")
        validation_results = run_domain_validation(model_name=args.embedding_model, logger=logger)

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

    # Load gold standard
    try:
        all_abstracts = load_gold_standard(args.gold_standard_path)
    except FileNotFoundError:
        logger.error("Gold standard dataset not found: %s", args.gold_standard_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Invalid gold standard JSON at %s: %s", args.gold_standard_path, e)
        return

    if args.list_papers:
        print("Available paper IDs:")
        for abstract in all_abstracts:
            print(f"  {abstract['id']}: {abstract['title'][:60]}...")
        return

    logger.info("Starting gold standard evaluation...")

    detected_host = autodetect_llm_host(config.active_llm_host, logger=logger)
    if detected_host:
        config.llm_host = detected_host

    if config.show_config:
        config.log_config(logger)

    # ── NEW: Load MIDAS ontology for constrained decoding ──
    json_schema = None
    if args.constrained:
        try:
            from midas_llm.utils.ontology.midas_vocabulary import MIDASVocabulary

            owl_path = args.ontology_path or str(DEFAULT_ONTOLOGY_PATH)
            logger.info("Loading MIDAS ontology from: %s", owl_path)
            vocab = MIDASVocabulary.from_owl(owl_path)
            json_schema = vocab.build_json_schema()

            logger.info(
                "MIDAS ontology loaded: %d schema properties",
                len(json_schema.get("properties", {})),
            )
            logger.info("Constrained decoding: ENABLED")

            # Also append ontology vocabulary to prompt
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
                "midas_vocabulary.py is in src/midas_llm/utils/ontology/"
            )
            return
        except Exception as e:
            logger.error("Failed to load MIDAS ontology: %s", e)
            return
    else:
        logger.info("Constrained decoding: DISABLED (use --constrained to enable)")

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
                "Constrained response validation enabled using runtime ontology schema."
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

    logger.info("Evaluating %d of %d gold standard abstracts", len(abstracts), len(all_abstracts))

    use_llm_eval = not args.no_llm_eval
    use_vector_eval = not args.no_vector_eval

    if use_vector_eval:
        logger.info("Vector similarity evaluation: ENABLED (model=%s)", args.embedding_model)
        logger.info("  Auto-match threshold: >= %.2f", args.vector_high_threshold)
        logger.info("  Auto-reject threshold: <= %.2f", args.vector_low_threshold)
    else:
        logger.info("Vector similarity evaluation: DISABLED")

    if use_llm_eval:
        logger.info("LLM semantic matching: ENABLED")
    else:
        logger.info("LLM semantic matching: DISABLED")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.output_dir)
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
        embedding_model=args.embedding_model,
        run_folder=run_folder,
        json_schema=json_schema,  # ── NEW: passed through
        validate_constrained_json=args.validate_constrained_json,
        validation_schema=validation_schema,
    )

    results["evaluation_config"] = {
        "use_vector_eval": use_vector_eval,
        "use_llm_eval": use_llm_eval,
        "vector_high_threshold": args.vector_high_threshold,
        "vector_low_threshold": args.vector_low_threshold,
        "embedding_model": args.embedding_model,
        "constrained": args.constrained,  # ── NEW
        "ontology_path": args.ontology_path,  # ── NEW
        "gold_standard_path": args.gold_standard_path,
        "output_dir": args.output_dir,
        "validate_constrained_json": args.validate_constrained_json,
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

    if config.generate_evaluation_report:
        report_file = generate_evaluation_text_report(
            results, run_folder=run_folder, logger=logger,
        )
        html_report_file = generate_evaluation_html_report(
            results, run_folder=run_folder, logger=logger,
        )
        logger.info("Human-readable evaluation report: %s", report_file)
        logger.info("HTML evaluation report: %s", html_report_file)

    print_summary(results, logger)


if __name__ == "__main__":
    main()
