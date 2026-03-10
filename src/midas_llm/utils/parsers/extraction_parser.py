from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.midas_llm.utils.ontology.modeling_domains import identify_modeling_domains

LOGGER = logging.getLogger("midas-llm")

# Known constrained-schema fields
KNOWN_SCHEMA_FIELDS = {
    "model_type", "model_determinism",
    "pathogen_name", "pathogen_type",
    "disease_name",
    "host_species",
    "primary_population", "population_setting_type",
    "geographic_scope", "geographic_units",
    "study_dates_start", "study_dates_end",
    "historical_vs_hypothetical",
    "study_goal_category",
    "intervention_present", "intervention_types",
    "data_used", "data_source",
    "calibration_mentioned", "calibration_techniques",
    "key_outcome_measures",
    "code_available",
    "extraction_notes",
}

ABSENT_VALUE_SYNONYMS = {
    "unspecified": "not specified",
    "unknown": "not specified",
    "n/a": "not applicable",
    "none": "not mentioned",
    "not available": "not mentioned",
    "not stated": "not mentioned",
    "not reported": "not mentioned",
}


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences if present."""
    s = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return s


def _try_load_json_response(text: str) -> dict[str, Any] | None:
    """Parse response text as a single constrained JSON object."""
    candidate = _strip_code_fences(text).strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except JSONDecodeError:
        return None


def normalize_absent_value(value: str) -> str:
    return ABSENT_VALUE_SYNONYMS.get(value.strip().lower(), value)


def normalize_field_name(field: str) -> str:
    field = re.sub(r'^\*+', '', field)
    field = re.sub(r'\*+$', '', field)
    field = re.sub(r'^_+', '', field)
    field = re.sub(r'_+$', '', field)
    field = re.sub(r'^\#+\s*', '', field)
    return field.strip().lower().replace(' ', '_').replace('-', '_')


def _normalize_json_values(values: Any) -> list[str]:
    """Normalize constrained-schema values into a clean list[str]."""
    if values is None:
        return []

    raw_vals = values if isinstance(values, list) else [values]
    out: list[str] = []
    for v in raw_vals:
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, ensure_ascii=False)
        else:
            v_str = str(v).strip()
        if not v_str:
            continue
        out.append(normalize_absent_value(v_str))
    return out


def parse_llm_json_response(response_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse constrained JSON response into internal extracted-data structure."""
    extracted: dict[str, dict[str, Any]] = {}

    for raw_field, payload in response_obj.items():
        normalized_field = normalize_field_name(str(raw_field))
        if normalized_field not in KNOWN_SCHEMA_FIELDS:
            continue

        domains = identify_modeling_domains(normalized_field)

        if isinstance(payload, dict):
            values = _normalize_json_values(payload.get("values"))
            reasoning_raw = payload.get("reasoning")
            reasoning = str(reasoning_raw).strip() if reasoning_raw is not None else None
        else:
            values = _normalize_json_values(payload)
            reasoning = None

        extracted[normalized_field] = {
            "value": ", ".join(values),  # backward compatibility for downstream evaluators
            "values": values,
            "provenance": None,
            "parent_class": None,
            "differentia": None,
            "key_relationships": None,
            "definition": None,
            "reasoning": reasoning,
            "ontologies": [],
            "domains": domains,
            # Legacy placeholders kept for report compatibility
            "provenances": {},
            "parent_classes": {},
            "differentias": {},
            "key_relationships_all": {},
            "definitions": {},
            "reasonings": ({1: reasoning} if reasoning else {}),
        }

    return extracted


def normalize_llm_format(raw: str) -> str:
    """Deprecated no-op kept for compatibility.

    Free-text parsing is no longer supported.
    """
    return raw


def parse_llm_output(response_text: str) -> dict[str, dict[str, Any]]:
    """Parse constrained JSON output only.

    Raises:
        ValueError: if response is not a valid constrained JSON object.
    """
    json_obj = _try_load_json_response(response_text)
    if json_obj is None:
        raise ValueError(
            "Free-text LLM responses are not supported. Expected constrained JSON object."
        )
    return parse_llm_json_response(json_obj)


def parse_and_display_extracted_data(
    response_content: str,
    *,
    logger: logging.Logger = LOGGER,
) -> dict[str, dict[str, Any]]:
    extracted_data = parse_llm_output(response_content)

    logger.debug("Parsed constrained JSON response with %d attributes", len(extracted_data))
    for attr, data in extracted_data.items():
        logger.debug("[%s]: %s", attr, data.get("values", []))
        if data.get("reasoning"):
            logger.debug("   Reasoning: %s", data["reasoning"])
    return extracted_data


def create_response_json(
    response_content: str,
    model: str,
    abstract_id: str | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a response JSON object from constrained response content."""
    extracted = parse_llm_output(response_content)

    return {
        "content": response_content,
        "model": model,
        "done": True,
        "timestamp": datetime.now().isoformat(),
        "abstract_id": abstract_id,
        "extracted": extracted,
        "evaluation": evaluation,
        "raw_response": response_content[:2000],
    }


def save_response_json(
    response_content: str,
    model: str,
    output_path: Path | str,
    abstract_id: str | None = None,
    evaluation: dict[str, Any] | None = None,
    logger: logging.Logger = LOGGER,
) -> Path:
    """Create and save a JSON object from constrained response content."""
    output_path = Path(output_path)
    response_json = create_response_json(
        response_content=response_content,
        model=model,
        abstract_id=abstract_id,
        evaluation=evaluation,
    )

    safe_model_name = model.replace("/", "-").replace("\\", "-")
    json_file = output_path / f"{safe_model_name}_response.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2, ensure_ascii=False)

    logger.info("Saved response JSON to: %s", json_file)
    return json_file
