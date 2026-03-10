"""Dataset loading and constrained JSON parsing/validation utilities."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from concept_extractor.utils.structured_vocab.IdSynonyms import DISEASE_SYNONYMS
from concept_extractor.utils.parsers.extraction_parser import normalize_absent_value
from concept_extractor.constants import (
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_RESULTS_FOLDER,
    GOLD_STANDARD_FOLDER,
)

LOGGER = logging.getLogger("midas-llm")

# Repo root for resolving paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
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
