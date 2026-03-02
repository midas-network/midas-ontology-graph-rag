#!/usr/bin/env python3
"""
Schema-driven few-shot generator for MIDAS constrained extraction prompts.

Goals:
- Generate a few-shot JSON object that conforms to the constrained schema
- Ensure enum values use exact schema spellings
- Ensure controlled fields use preferred MIDAS vocab terms
- Prevent prompt drift between schema/vocab and few-shot examples

Repo assumptions (based on user's structure):
- midas_schema.json                (repo root)
- midas_prompt_vocab.txt           (repo root)
- midas_synonyms.json              (repo root, optional)
- resources/prompts/               (output location)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the MIDAS repo root (identified by midas_schema.json).
    """
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "midas_schema.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root from {start}. Expected to find 'midas_schema.json' in a parent directory."
    )


# If this script lives in utils/prompt/resources/
# __file__ -> .../utils/prompt/resources/generate_few_shot_from_schema.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)

SCHEMA_PATH = REPO_ROOT / "midas_schema.json"
VOCAB_PATH = REPO_ROOT / "midas_prompt_vocab.txt"
SYNONYMS_PATH = REPO_ROOT / "midas_synonyms.json"  # optional / non-blocking

PROMPTS_DIR = REPO_ROOT / "resources" / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON_PATH = PROMPTS_DIR / "few_shot_generated.json"
OUTPUT_TXT_PATH = PROMPTS_DIR / "few-shot.generated.txt"

CONTROLLED_FIELDS: Set[str] = {"data_used", "intervention_types", "model_type"}


# ---------------------------------------------------------------------
# Example values (ILLUSTRATIVE) -- validated against schema/vocab
# IMPORTANT: if your MIDAS vocab uses different exact terms, change only
# the values below; the script will validate and fail fast if mismatched.
# ---------------------------------------------------------------------

PREFERRED_EXAMPLE_VALUES: Dict[str, List[str]] = {
    # Enum-like / categorical examples (validated against schema enum if enum-constrained)
    "model_determinism": ["not mentioned"],
    "pathogen_type": ["virus"],
    "geographic_scope": ["national"],
    "historical_vs_hypothetical": ["historical"],
    "study_goal_category": ["forecast/nowcast"],
    "intervention_present": ["not mentioned"],
    "calibration_mentioned": ["not mentioned"],
    "code_available": ["not mentioned"],

    # Controlled vocab examples (validated against midas_prompt_vocab.txt)
    # NOTE: these must match your vocab file exactly
    "data_used": ["Case Count"],
    "intervention_types": [],
    "model_type": ["Forecasting"],

    # Open/free-text-ish fields
    "pathogen_name": ["monkeypox virus", "MPXV"],
    "disease_name": ["mpox"],
    "host_species": ["human"],
    "primary_population": ["general population"],
    "population_setting_type": ["community"],
    "geographic_units": ["United States"],
    "data_source": ["not specified"],
    "calibration_techniques": [],
    "key_outcome_measures": ["cases"],
    "study_dates_start": ["2022"],
    "study_dates_end": ["2022"],

    # Optional if present in your schema
    "extraction_notes": [],
}


DEFAULT_REASONING: Dict[str, str] = {
    "model_determinism": "The title/abstract does not explicitly state whether the model is deterministic or stochastic.",
    "pathogen_type": "[Inferred] mpox is caused by a virus.",
    "geographic_scope": "The study is framed at the U.S. national level.",
    "historical_vs_hypothetical": "The study concerns an observed outbreak period.",
    "study_goal_category": "The study explicitly focuses on nowcasting/forecasting.",
    "intervention_present": "No intervention evaluation is explicitly described in the example.",
    "calibration_mentioned": "No fitting or estimation method is explicitly described in the example.",
    "code_available": "No code repository or code availability statement is mentioned in the example.",

    "data_used": "[Inferred] Case count data are a minimally supported data type for outbreak nowcasting/forecasting examples.",
    "intervention_types": "No intervention types are described in the example.",
    "model_type": "The study is framed as a forecasting analysis.",

    "pathogen_name": "[Inferred] mpox refers to monkeypox virus (MPXV) in this context.",
    "disease_name": "The disease name is explicitly given as mpox.",
    "host_species": "The example concerns a human public health outbreak.",
    "primary_population": "No narrower subgroup is identified in the example.",
    "population_setting_type": "The study is population/community-level rather than a single facility.",
    "geographic_units": "The example names the United States.",
    "data_source": "No specific data provider or repository is named in the example.",
    "calibration_techniques": "No calibration techniques are explicitly described in the example.",
    "key_outcome_measures": "[Inferred] Cases are a standard minimally committal outcome for outbreak nowcasting/forecasting.",
    "study_dates_start": "The example references 2022 as the study period.",
    "study_dates_end": "The example references 2022 as the study period.",
    "extraction_notes": "No additional notes.",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_midas_prompt_vocab(vocab_text: str) -> Dict[str, List[str]]:
    """
    Parse midas_prompt_vocab.txt into:
      { "data_used": [...], "intervention_types": [...], "model_type": [...] }

    Expected flexible format examples:
        data_used:
          - "Case Count"
          - "Hospital Admissions"

        model_type:
          - "Forecasting"
    """
    sections: Dict[str, List[str]] = {}
    current_field: Optional[str] = None

    field_header_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*$")
    quoted_bullet_re = re.compile(r'^\s*-\s*"(.+)"\s*$')
    plain_bullet_re = re.compile(r"^\s*-\s*(.+?)\s*$")

    for raw_line in vocab_text.splitlines():
        line = raw_line.rstrip()

        m_field = field_header_re.match(line)
        if m_field:
            field_name = m_field.group(1)
            if field_name in CONTROLLED_FIELDS:
                current_field = field_name
                sections.setdefault(current_field, [])
            else:
                current_field = None
            continue

        if current_field is None:
            continue

        m_q = quoted_bullet_re.match(line)
        if m_q:
            sections[current_field].append(m_q.group(1))
            continue

        m_p = plain_bullet_re.match(line)
        if m_p:
            value = m_p.group(1).strip()
            # ignore comments / separators
            if value and not value.startswith("#"):
                sections[current_field].append(value)

    return sections


def find_extraction_field_specs(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Find top-level extraction fields assuming schema root object with properties.
    Skips any top-level keys that don't look like constrained field objects.
    """
    if schema.get("type") != "object":
        raise ValueError("Top-level schema must be an object")

    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("Schema root missing object 'properties'")

    field_specs: Dict[str, Dict[str, Any]] = {}
    for field_name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        # Must look like constrained field object with {values, reasoning}
        if field_has_values_array(spec) and field_has_reasoning(spec):
            field_specs[field_name] = spec

    if not field_specs:
        raise ValueError(
            "No constrained extraction fields found at schema root. "
            "Check schema shape or update parser for nested/$ref fields."
        )

    return field_specs


def field_has_values_array(field_schema: Dict[str, Any]) -> bool:
    props = field_schema.get("properties", {})
    if not isinstance(props, dict):
        return False
    values_spec = props.get("values")
    return isinstance(values_spec, dict) and values_spec.get("type") == "array"


def field_has_reasoning(field_schema: Dict[str, Any]) -> bool:
    props = field_schema.get("properties", {})
    if not isinstance(props, dict):
        return False
    reasoning_spec = props.get("reasoning")
    return isinstance(reasoning_spec, dict) and reasoning_spec.get("type") == "string"


def get_values_items_schema(field_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    props = field_schema.get("properties", {})
    values_spec = props.get("values")
    if not isinstance(values_spec, dict):
        return None
    if values_spec.get("type") != "array":
        return None
    items = values_spec.get("items")
    return items if isinstance(items, dict) else None


def get_field_enum(field_schema: Dict[str, Any]) -> Optional[List[str]]:
    """
    Returns enum list if field values[] items are enum-constrained strings.
    """
    items = get_values_items_schema(field_schema)
    if not items:
        return None
    enum_vals = items.get("enum")
    if isinstance(enum_vals, list) and all(isinstance(x, str) for x in enum_vals):
        return enum_vals
    return None


def get_field_min_items(field_schema: Dict[str, Any]) -> int:
    props = field_schema.get("properties", {})
    values_spec = props.get("values", {})
    if isinstance(values_spec, dict):
        mi = values_spec.get("minItems", 0)
        if isinstance(mi, int):
            return mi
    return 0


def choose_default_enum_value(enum_vals: List[str]) -> str:
    """
    Conservative fallback if no explicit example value is configured.
    Prefer a substantive value if available; otherwise use an allowed absent value.
    """
    preferred_order = [
        "historical",
        "national",
        "yes",
        "no",
        "not mentioned",
        "not specified",
        "not applicable",
    ]
    enum_set = set(enum_vals)
    for val in preferred_order:
        if val in enum_set:
            return val
    return enum_vals[0]


def validate_values_against_schema_enum(field_name: str, values: List[str], field_schema: Dict[str, Any]) -> None:
    enum_vals = get_field_enum(field_schema)
    if enum_vals is None:
        return
    enum_set = set(enum_vals)
    for v in values:
        if v not in enum_set:
            raise ValueError(
                f"Schema enum violation for '{field_name}': {v!r} not in {sorted(enum_set)}"
            )


def validate_controlled_vocab(field_name: str, values: List[str], controlled_vocab: Dict[str, List[str]]) -> None:
    if field_name not in CONTROLLED_FIELDS:
        return
    allowed = set(controlled_vocab.get(field_name, []))
    for v in values:
        if v not in allowed:
            raise ValueError(
                f"Controlled vocab violation for '{field_name}': {v!r} not found in midas_prompt_vocab.txt"
            )


def build_reasoning(field_name: str, values: List[str]) -> str:
    if field_name in DEFAULT_REASONING:
        return DEFAULT_REASONING[field_name]
    if not values:
        return "No value is provided in this illustrative example."
    return "Illustrative schema-valid example value."


def build_field_object(field_name: str, field_schema: Dict[str, Any], controlled_vocab: Dict[str, List[str]]) -> Dict[str, Any]:
    min_items = get_field_min_items(field_schema)
    enum_vals = get_field_enum(field_schema)

    # Pick configured values if provided
    if field_name in PREFERRED_EXAMPLE_VALUES:
        chosen = list(PREFERRED_EXAMPLE_VALUES[field_name])
    else:
        # Fallback generation
        if enum_vals:
            chosen = [choose_default_enum_value(enum_vals)]
        else:
            # Open field default:
            # if minItems requires at least one item, use a generic placeholder
            chosen = ["example value"] if min_items >= 1 else []

    # Validate enum conformance
    validate_values_against_schema_enum(field_name, chosen, field_schema)

    # Validate controlled vocab conformance
    validate_controlled_vocab(field_name, chosen, controlled_vocab)

    # Validate minItems
    if len(chosen) < min_items:
        raise ValueError(
            f"Schema minItems violation for '{field_name}': chose {len(chosen)} values, need >= {min_items}"
        )

    return {
        "values": chosen,
        "reasoning": build_reasoning(field_name, chosen),
    }


def build_few_shot_example(schema: Dict[str, Any], controlled_vocab: Dict[str, List[str]]) -> Dict[str, Any]:
    field_specs = find_extraction_field_specs(schema)

    out: Dict[str, Any] = {}
    for field_name, field_schema in field_specs.items():
        out[field_name] = build_field_object(field_name, field_schema, controlled_vocab)

    return out


def render_few_shot_txt_block(example_json: Dict[str, Any]) -> str:
    return (
        "=== FEW-SHOT EXAMPLE (Generated, Schema-Validated) ===\n\n"
        "Title:\n"
        "Nowcasting and forecasting the 2022 U.S. mpox outbreak: Support for public health decision making and lessons learned\n\n"
        "Abstract:\n"
        "[Abstract text omitted here; generated example is for schema shape and controlled-vocab alignment]\n\n"
        "Example output:\n\n"
        f"{json.dumps(example_json, indent=2, ensure_ascii=False)}\n\n"
        "=== END FEW-SHOT EXAMPLE ===\n"
    )


def main() -> None:
    # Load schema
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    schema = load_json(SCHEMA_PATH)

    # Load vocab
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}")
    vocab_text = VOCAB_PATH.read_text(encoding="utf-8")
    controlled_vocab = parse_midas_prompt_vocab(vocab_text)

    # Ensure controlled field sections exist
    for field in CONTROLLED_FIELDS:
        if field not in controlled_vocab:
            raise ValueError(
                f"Controlled vocab section missing in {VOCAB_PATH.name}: {field}\n"
                f"Expected a section like:\n{field}:\n  - \"Preferred Term\"\n"
            )
        if not controlled_vocab[field]:
            raise ValueError(f"Controlled vocab section is empty: {field}")

    # Optional synonyms file load (non-blocking; placeholder for future strict checks)
    if SYNONYMS_PATH.exists():
        try:
            _ = load_json(SYNONYMS_PATH)
            print(f"[INFO] Loaded synonyms file: {SYNONYMS_PATH.name} (no strict validation in this template)")
        except Exception as e:
            print(f"[WARN] Could not parse {SYNONYMS_PATH.name}: {e}")

    # Build and validate example
    few_shot_json = build_few_shot_example(schema, controlled_vocab)

    # Write outputs
    OUTPUT_JSON_PATH.write_text(json.dumps(few_shot_json, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_TXT_PATH.write_text(render_few_shot_txt_block(few_shot_json), encoding="utf-8")

    print(f"[OK] Wrote JSON: {OUTPUT_JSON_PATH}")
    print(f"[OK] Wrote text block: {OUTPUT_TXT_PATH}")
    print(f"[INFO] Fields generated: {len(few_shot_json)}")
    print("[INFO] Guarantees enforced:")
    print("  - exact schema enum spellings (where enum-constrained)")
    print("  - preferred MIDAS terms for data_used / intervention_types / model_type (from vocab file)")


if __name__ == "__main__":
    main()