from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..modeling_domains import identify_modeling_domains

LOGGER = logging.getLogger("midas-llm")

# ── Known schema fields ────────────────────────────────────────────────
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

METADATA_SUBFIELDS = {
    "provenance", "parent_class", "differentia",
    "key_relationships", "definition", "reasoning",
}

SECTION_HEADER_NAMES = {
    "model_characteristics", "pathogen_disease", "pathogen_&_disease",
    "pathogen_and_disease",
    "population_setting", "population_&_setting", "population_and_setting",
    "geography_time", "geography_&_time", "geography_and_time",
    "study_purpose", "interventions",
    "data_methods", "data_&_methods", "data_and_methods",
    "outcomes_outputs", "outcomes_&_outputs", "outcomes_and_outputs",
    "reproducibility", "additional_notes",
    "final_checklist_verification", "final_checklist",
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

def normalize_absent_value(value: str) -> str:
    return ABSENT_VALUE_SYNONYMS.get(value.strip().lower(), value)


def _is_known_field(normalized: str) -> bool:
    return normalized in KNOWN_SCHEMA_FIELDS or normalized in METADATA_SUBFIELDS


# ═══════════════════════════════════════════════════════════════════════
# Format normalizer — rewrites non-standard LLM output conventions into
# the canonical format that parse_llm_response() expects.
# ═══════════════════════════════════════════════════════════════════════

def _is_nemotron_format(raw: str) -> bool:
    """Detect **field**: name / **value**: val pattern (Nemotron-style)."""
    field_lines = re.findall(
        r'^\*\*field\*\*\s*:', raw, re.MULTILINE | re.IGNORECASE
    )
    value_lines = re.findall(
        r'^\*\*value\*\*\s*:', raw, re.MULTILINE | re.IGNORECASE
    )
    return len(field_lines) >= 3 and len(value_lines) >= 3


def _normalize_nemotron(raw: str) -> str:
    """Rewrite Nemotron **field**/**value** blocks into canonical format.

    Input:
        **field**: model_type
        **value**: stochastic
        **provenance1**: "some quote"
        **reasoning1**: Inferred because...

    Output:
        **model_type**
        stochastic
        provenance1: "some quote"
        reasoning1: Inferred because...
    """
    lines = raw.split('\n')
    output: list[str] = []
    i = 0

    # Regex for metadata sub-fields in bold: **provenance1**: ...
    _meta_bold_re = re.compile(
        r'^\*\*((?:provenance|definition|reasoning|parent_class|'
        r'differentia|key_relationships)\d*)\*\*\s*:\s*(.*)',
        re.IGNORECASE,
    )
    # Regex for metadata sub-fields without bold: provenance1: ...
    _meta_plain_re = re.compile(
        r'^((?:provenance|definition|reasoning|parent_class|'
        r'differentia|key_relationships)\d*)\s*:\s*(.*)',
        re.IGNORECASE,
    )

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Match: **field**: field_name ──
        field_match = re.match(
            r'^\*\*field\*\*\s*:\s*(.+)$', stripped, re.IGNORECASE
        )
        if field_match:
            field_name = field_match.group(1).strip()
            j = i + 1
            value_text = ''
            metadata_lines: list[str] = []

            while j < len(lines):
                ns = lines[j].strip()

                # Next **field**: → stop
                if re.match(r'^\*\*field\*\*\s*:', ns, re.IGNORECASE):
                    break

                # Skip headers / separators / empties
                if not ns or ns.startswith('#') or ns.startswith('---'):
                    j += 1
                    continue

                # **value**: ...
                val_m = re.match(
                    r'^\*\*value\*\*\s*:\s*(.*)', ns, re.IGNORECASE
                )
                if val_m:
                    value_text = val_m.group(1).strip()
                    j += 1
                    continue

                # Bold metadata: **provenance1**: ...
                meta_m = _meta_bold_re.match(ns)
                if meta_m:
                    metadata_lines.append(
                        f"  {meta_m.group(1)}: {meta_m.group(2).strip()}"
                    )
                    j += 1
                    continue

                # Plain metadata: provenance1: ...
                meta_p = _meta_plain_re.match(ns)
                if meta_p:
                    metadata_lines.append(f"  {ns}")
                    j += 1
                    continue

                # Anything else — treat as value if we don't have one yet
                if not value_text:
                    value_text = ns
                j += 1

            # Emit canonical block
            output.append(f'**{field_name}**')
            if value_text:
                output.append(value_text)
            output.extend(metadata_lines)
            output.append('')
            i = j
            continue

        # ── Pass through section headers ──
        if stripped.startswith('#') or stripped.startswith('---'):
            output.append(line)
            i += 1
            continue

        # Bold section header without ### (e.g. **Pathogen & Disease**)
        if re.match(r'^\*\*[A-Z][a-zA-Z\s&]+\*\*$', stripped):
            output.append(f'### {stripped}')
            i += 1
            continue

        # Skip known non-schema bold fields like **study_id**: ...
        bold_kv = re.match(r'^\*\*(\w+)\*\*\s*:\s*(.+)$', stripped)
        if bold_kv and bold_kv.group(1).lower() in ('study_id',):
            i += 1
            continue

        # ── FINAL CHECKLIST block — skip entirely ──
        if re.match(r'^\*?\*?-?\s*\[[ X]\]', stripped) or \
           re.match(r'^\*\*FINAL\s+CHECKLIST', stripped, re.IGNORECASE):
            i += 1
            continue

        # Default pass-through
        output.append(line)
        i += 1

    return '\n'.join(output)


def normalize_llm_format(raw: str) -> str:
    """Detect LLM output format and normalize to canonical form."""
    if _is_nemotron_format(raw):
        return _normalize_nemotron(raw)
    return raw


# ═══════════════════════════════════════════════════════════════════════
# Core parser
# ═══════════════════════════════════════════════════════════════════════

def normalize_field_name(field: str) -> str:
    field = re.sub(r'^\*+', '', field)
    field = re.sub(r'\*+$', '', field)
    field = re.sub(r'^_+', '', field)
    field = re.sub(r'_+$', '', field)
    field = re.sub(r'^\#+\s*', '', field)
    return field.strip().lower().replace(' ', '_').replace('-', '_')


def clean_value(value: str) -> str:
    value = re.sub(r'^\*+\s*', '', value)
    value = re.sub(r'\s*\*+$', '', value)
    return value.strip()


def _looks_like_new_field(stripped: str, field_with_value, field_alone) -> bool:
    m = field_with_value.match(stripped)
    if m:
        norm = normalize_field_name(m.group(1))
        return _is_known_field(norm) and norm not in SECTION_HEADER_NAMES

    m = field_alone.match(stripped)
    if m:
        norm = normalize_field_name(m.group(1))
        return _is_known_field(norm) and norm not in SECTION_HEADER_NAMES

    return False


def parse_llm_response(response_text: str) -> dict[str, dict[str, Any]]:
    """Parse LLM response text into structured attributes.

    Handles various formats:
    - field_name: value
    - *field_name*: value
    - **field_name**: value
    - **field_name** (on own line, value follows)
    - - field_name: value (bullet point)

    Format-normalised input (Nemotron **field**/**value** style) is
    handled by normalize_llm_format() which runs before this function.
    """
    attributes: dict[str, dict[str, Any]] = {}
    lines = response_text.split('\n')
    current_attr = None
    current_content: list[str] = []

    field_with_value = re.compile(
        r'^[-\*•]?\s*[\*_]{0,2}([a-zA-Z][a-zA-Z0-9_\s\-]*)[\*_]{0,2}:\s*(.+)$'
    )
    field_alone = re.compile(
        r'^[-\*•]?\s*[\*_]{0,2}([a-zA-Z][a-zA-Z0-9_\s\-]*)[\*_]{0,2}:?\s*$'
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith('#') or stripped.startswith('---'):
            if current_attr:
                current_content.append(line)
            i += 1
            continue

        if re.match(r'^\*\*[A-Z][a-zA-Z\s&]+\*\*$', stripped):
            i += 1
            continue

        # ── FINAL CHECKLIST lines — skip ──
        if re.match(r'^\*?\*?-?\s*\[[ Xx]\]', stripped) or \
           re.match(r'^\*\*FINAL\s+CHECKLIST', stripped, re.IGNORECASE):
            i += 1
            continue

        # ── Case A: field with value on same line ──
        match = field_with_value.match(stripped)
        if match and not line.startswith((' ', '\t')):
            raw_field = match.group(1)
            normalized = normalize_field_name(raw_field)

            if len(normalized) > 50 or normalized.startswith(
                ('here_are', 'the_following', 'based_on', 'below_is')
            ):
                i += 1
                continue

            if current_attr:
                attributes[current_attr] = parse_attribute_block(
                    current_attr, '\n'.join(current_content)
                )

            current_attr = normalized
            current_content = [match.group(2)]
            i += 1
            continue

        # ── Case B: field name on its own line ─────────────────────
        match = field_alone.match(stripped)
        if match and not line.startswith((' ', '\t')):
            raw_field = match.group(1)
            normalized = normalize_field_name(raw_field)

            if len(normalized) > 50 or normalized.startswith(
                ('here_are', 'the_following', 'based_on', 'below_is')
            ):
                i += 1
                continue

            if normalized in SECTION_HEADER_NAMES:
                i += 1
                continue

            # Only treat as new field if it's a known schema field
            if not _is_known_field(normalized):
                if current_attr:
                    current_content.append(stripped)
                i += 1
                continue

            if current_attr:
                attributes[current_attr] = parse_attribute_block(
                    current_attr, '\n'.join(current_content)
                )

            current_attr = normalized
            current_content = []
            i += 1

            # Look-ahead: collect until next known field
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()

                if re.match(
                    r'^\s*(provenance|definition|reasoning|parent_class|'
                    r'differentia|key_relationships)\d*:',
                    next_line,
                    re.IGNORECASE,
                ):
                    current_content.append(next_line)
                    i += 1
                    continue

                if next_stripped and not next_line.startswith((' ', '\t')):
                    if _looks_like_new_field(
                        next_stripped, field_with_value, field_alone
                    ):
                        break

                if next_stripped.startswith('#') or next_stripped.startswith('---'):
                    i += 1
                    continue

                # Skip checklist lines inside look-ahead
                if re.match(r'^\*?\*?-?\s*\[[ Xx]\]', next_stripped):
                    i += 1
                    continue

                current_content.append(next_line)
                i += 1
            continue

        # Default: add to current content
        if current_attr:
            current_content.append(line)
        i += 1

    if current_attr:
        attributes[current_attr] = parse_attribute_block(
            current_attr, '\n'.join(current_content)
        )
    return attributes


# ═══════════════════════════════════════════════════════════════════════
# Attribute block parser (unchanged logic, reformatted)
# ═══════════════════════════════════════════════════════════════════════

def parse_attribute_block(attr: str, content: str) -> dict[str, Any]:
    domains = identify_modeling_domains(attr)
    lines = content.split('\n')

    value = ''
    rest_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped and not value:
            continue

        if re.match(
            r'^[-\*•]?\s*(provenance|definition|reasoning|parent_class|'
            r'differentia|key_relationships)\d*:',
            stripped,
            re.IGNORECASE,
        ):
            rest_lines = lines[i:]
            break

        if not value:
            # Strip "value:" prefix from non-normalized LLM output
            val_prefix = re.match(r'^value\s*:\s*(.+)$', stripped, re.IGNORECASE)
            if val_prefix:
                value = val_prefix.group(1).strip()
            else:
                value = stripped

            is_match = re.search(
                r'\bis\s+(?:a\s+)?([A-Za-z][A-Za-z0-9\s\-\/]+?)(?:\.|,|$)',
                stripped,
            )
            if is_match:
                extracted = is_match.group(1).strip()
                if len(extracted) < 50 and extracted.lower() not in (
                    'not', 'the', 'a', 'an',
                ):
                    value = extracted
        else:
            rest_lines.append(line)

    rest_content = '\n'.join(rest_lines)
    rest_content_normalized = re.sub(
        r'\n(\s*)[-\*•]\s*(provenance|definition|reasoning|parent_class|'
        r'differentia|key_relationships)',
        r'\n\1\2',
        rest_content,
    )
    has_numbered = bool(re.search(
        r'\n\s*(?:provenance|parent_class|differentia|key_relationships|'
        r'definition|reasoning)1:',
        '\n' + rest_content_normalized,
        re.IGNORECASE,
    ))

    if has_numbered:
        provenances: dict[int, str] = {}
        parent_classes: dict[int, str] = {}
        differentias: dict[int, str] = {}
        key_relationships_dict: dict[int, str] = {}
        definitions: dict[int, str] = {}
        reasonings: dict[int, str] = {}

        for idx in range(1, 20):
            pats = {
                'prov': rf'\n\s+provenance{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
                'parent': rf'\n\s+parent_class{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
                'diff': rf'\n\s+differentia{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
                'rel': rf'\n\s+key_relationships{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
                'defn': rf'\n\s+definition{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
                'reas': rf'\n\s+reasoning{idx}:\s*(.+?)(?=\n\s+\w+\d*:|$)',
            }
            found_any = False
            for key, pat in pats.items():
                m = re.search(pat, '\n' + rest_content_normalized, re.DOTALL)
                if m:
                    val = m.group(1).strip()
                    found_any = True
                    if key == 'prov':
                        provenances[idx] = val
                    elif key == 'parent':
                        parent_classes[idx] = val
                    elif key == 'diff':
                        differentias[idx] = val
                    elif key == 'rel':
                        key_relationships_dict[idx] = val
                    elif key == 'defn':
                        definitions[idx] = val
                    elif key == 'reas':
                        reasonings[idx] = val
            if not found_any:
                break

        definition = '; '.join(
            f"{definitions[k]}" for k in sorted(definitions.keys())
        ) if definitions else None

        return {
            'value': value,
            'provenance': provenances.get(1),
            'parent_class': parent_classes.get(1),
            'differentia': differentias.get(1),
            'key_relationships': key_relationships_dict.get(1),
            'definition': definition,
            'reasoning': reasonings.get(1),
            'provenances': provenances,
            'parent_classes': parent_classes,
            'differentias': differentias,
            'key_relationships_all': key_relationships_dict,
            'definitions': definitions,
            'reasonings': reasonings,
            'ontologies': [],
            'domains': domains,
        }

    # ── Un-numbered sub-fields ──
    provenance_match = re.search(
        r'\n\s*(?:provenance|citation):\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    parent_class_match = re.search(
        r'\n\s*parent_class:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    differentia_match = re.search(
        r'\n\s*differentia:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    key_relationships_match = re.search(
        r'\n\s*key_relationships:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    definition_match = re.search(
        r'\n\s*definition:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    reasoning_match = re.search(
        r'\n\s*reasoning:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )
    ontologies_match = re.search(
        r'\n\s*ontologies:\s*(.+?)(?=\n\s*\w+:|$)',
        '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE,
    )

    provenance = provenance_match.group(1).strip() if provenance_match else None
    parent_class = parent_class_match.group(1).strip() if parent_class_match else None
    differentia = differentia_match.group(1).strip() if differentia_match else None
    key_relationships = key_relationships_match.group(1).strip() if key_relationships_match else None
    definition = definition_match.group(1).strip() if definition_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    ontologies_str = ontologies_match.group(1).strip() if ontologies_match else None

    ontologies: list[str] = []
    if ontologies_str:
        ontologies_str = ontologies_str.strip('[]"\'')
        ontologies = [
            ont.strip().strip('"\'')
            for ont in ontologies_str.split(',')
            if ont.strip()
        ]

    is_multi_value = ',' in value
    if is_multi_value and (provenance or definition or reasoning):
        num_values = len([v.strip() for v in value.split(',') if v.strip()])
        LOGGER.warning(
            "FORMAT ERROR: '%s' has %d values (%s) but uses un-numbered "
            "fields. LLM should use provenance1/provenance2/etc for each "
            "value. Only the first value's metadata will be used.",
            attr,
            num_values,
            value,
        )

    return {
        'value': value,
        'provenance': provenance,
        'parent_class': parent_class,
        'differentia': differentia,
        'key_relationships': key_relationships,
        'definition': definition,
        'reasoning': reasoning,
        'ontologies': ontologies,
        'domains': domains,
    }


# ═══════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════

def parse_and_display_extracted_data(
    response_content: str, *, logger: logging.Logger = LOGGER
):
    # ── Step 0: normalize non-standard LLM output formats ──
    normalized = normalize_llm_format(response_content)
    if normalized != response_content:
        logger.info(
            "Detected non-standard LLM output format; "
            "normalized before parsing"
        )

    logger.info("LLM response:\n%s", response_content)
    extracted_data = parse_llm_response(normalized)
    logger.info("Extracted %d attributes", len(extracted_data))
    for attr, data in extracted_data.items():
        logger.info("[%s]: %s", attr, data['value'])
        if data.get('provenance'):
            logger.info("   Provenance: %s", data['provenance'])
        if data.get('definition'):
            logger.info("   Definition: %s", data['definition'])
        if data.get('reasoning'):
            logger.info("   Reasoning: %s", data['reasoning'])
    return extracted_data


def create_response_json(
    response_content: str,
    model: str,
    abstract_id: str | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON object from the LLM response.

    Args:
        response_content: The raw LLM response text
        model: The model name used
        abstract_id: Optional abstract identifier
        evaluation: Optional evaluation results

    Returns:
        A dictionary with the response data ready for JSON serialization
    """
    # Parse the response to extract structured data
    normalized = normalize_llm_format(response_content)


    extracted = parse_llm_response(normalized)

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
    """Create and save a JSON object from the LLM response.

    Args:
        response_content: The raw LLM response text
        model: The model name used
        output_path: Path where the JSON file should be saved
        abstract_id: Optional abstract identifier
        evaluation: Optional evaluation results
        logger: Logger instance

    Returns:
        The path to the saved JSON file
    """
    output_path = Path(output_path)

    # Create the JSON object
    response_json = create_response_json(
        response_content=response_content,
        model=model,
        abstract_id=abstract_id,
        evaluation=evaluation,
    )

    # Sanitize model name for filename
    safe_model_name = model.replace("/", "-").replace("\\", "-")
    json_file = output_path / f"{safe_model_name}_response.json"

    # Write the JSON file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=2, ensure_ascii=False)

    logger.info("Saved response JSON to: %s", json_file)
    return json_file