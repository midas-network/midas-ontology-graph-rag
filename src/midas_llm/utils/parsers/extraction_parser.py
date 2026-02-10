from __future__ import annotations

import logging
import re
from typing import Any

from ..modeling_domains import identify_modeling_domains

LOGGER = logging.getLogger("midas-llm")


def normalize_field_name(field: str) -> str:
    """Normalize field name by removing markdown formatting."""
    # Remove markdown bold/italic markers
    field = re.sub(r'^\*+', '', field)
    field = re.sub(r'\*+$', '', field)
    field = re.sub(r'^_+', '', field)
    field = re.sub(r'_+$', '', field)
    field = re.sub(r'^\#+\s*', '', field)  # Remove heading markers
    return field.strip().lower().replace(' ', '_').replace('-', '_')


def clean_value(value: str) -> str:
    """Clean extracted value by removing markdown formatting."""
    # Remove leading/trailing ** or *
    value = re.sub(r'^\*+\s*', '', value)
    value = re.sub(r'\s*\*+$', '', value)
    return value.strip()


def parse_llm_response(response_text: str) -> dict[str, dict[str, Any]]:
    """Parse LLM response text into structured attributes.

    Handles various formats:
    - field_name: value
    - *field_name*: value (markdown italic)
    - **field_name**: value (markdown bold)
    - **field_name** (on own line, value follows)
    - - field_name: value (bullet point)
    - field_name: value (with various whitespace)
    """
    attributes: dict[str, dict[str, Any]] = {}
    lines = response_text.split('\n')
    current_attr = None
    current_content = []

    # Pattern to match field names with optional markdown formatting and value on same line
    # Also handles bullet points: - field_name: value
    # Matches: field_name: value, *field_name*: value, **field_name**: value, - field_name: value
    field_with_value = re.compile(r'^[-\*•]?\s*[\*_]{0,2}([a-zA-Z][a-zA-Z0-9_\s\-]*)[\*_]{0,2}:\s*(.+)$')

    # Pattern to match sub-fields (provenance, definition, etc.) - possibly with bullet points
    sub_field_pattern = re.compile(r'^\s*[-\*•]?\s*(provenance|definition|reasoning|parent_class|differentia|key_relationships)\d*:\s*(.*)$', re.IGNORECASE)

    # Pattern for field name on its own line (no colon or empty after colon)
    # Matches: **field_name**, *field_name*, field_name, - field_name
    field_alone = re.compile(r'^[-\*•]?\s*[\*_]{0,2}([a-zA-Z][a-zA-Z0-9_\s\-]*)[\*_]{0,2}:?\s*$')

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and section headers (including markdown bold section headers)
        if not stripped or stripped.startswith('#') or stripped.startswith('---'):
            if current_attr:
                current_content.append(line)
            i += 1
            continue

        # Skip section headers like **Model Characteristics** or **Pathogen & Disease**
        if re.match(r'^\*\*[A-Z][a-zA-Z\s&]+\*\*$', stripped):
            i += 1
            continue

        # Check for field with value on same line
        match = field_with_value.match(stripped)
        if match and not line.startswith(' ') and not line.startswith('\t'):
            raw_field = match.group(1)
            normalized = normalize_field_name(raw_field)

            # Skip preamble/intro lines (long field names that are actually sentences)
            if len(normalized) > 50 or normalized.startswith('here_are') or normalized.startswith('the_following') or normalized.startswith('based_on'):
                i += 1
                continue

            # Save previous attribute
            if current_attr:
                attributes[current_attr] = parse_attribute_block(current_attr, '\n'.join(current_content))

            current_attr = normalized
            current_content = [match.group(2)]
            i += 1
            continue

        # Check for field name on its own line
        match = field_alone.match(stripped)
        if match and not line.startswith(' ') and not line.startswith('\t'):
            raw_field = match.group(1)
            normalized = normalize_field_name(raw_field)

            # Skip preamble/intro lines
            if len(normalized) > 50 or normalized.startswith('here_are') or normalized.startswith('the_following') or normalized.startswith('based_on'):
                i += 1
                continue

            # Skip section header-like names
            if normalized in ('model_characteristics', 'pathogen_disease', 'pathogen_&_disease',
                             'population_setting', 'population_&_setting', 'geography_time',
                             'geography_&_time', 'study_purpose', 'interventions',
                             'data_methods', 'data_&_methods', 'outcomes_outputs',
                             'outcomes_&_outputs', 'reproducibility', 'additional_notes'):
                i += 1
                continue

            # Save previous attribute
            if current_attr:
                attributes[current_attr] = parse_attribute_block(current_attr, '\n'.join(current_content))

            current_attr = normalized
            # Look ahead for value - might be description text on next line(s) before provenance
            current_content = []
            i += 1

            # Collect content until we hit provenance: or next field
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()

                # Check if this is a sub-field like provenance:, definition:, etc.
                if re.match(r'^\s*(provenance|definition|reasoning|parent_class|differentia|key_relationships)\d*:', next_line, re.IGNORECASE):
                    current_content.append(next_line)
                    i += 1
                    continue

                # Check if we hit a new top-level field
                if (field_with_value.match(next_stripped) or field_alone.match(next_stripped)) and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    break

                # Check for section header
                if next_stripped.startswith('#') or next_stripped.startswith('---'):
                    i += 1
                    continue

                current_content.append(next_line)
                i += 1
            continue

        # Otherwise, add to current content
        if current_attr:
            current_content.append(line)
        i += 1

    if current_attr:
        attributes[current_attr] = parse_attribute_block(current_attr, '\n'.join(current_content))
    return attributes


def parse_attribute_block(attr: str, content: str) -> dict[str, Any]:
    domains = identify_modeling_domains(attr)
    lines = content.split('\n')

    # Find the value - might be first line, or might need to extract from description
    value = ''
    rest_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines at start
        if not stripped and not value:
            continue

        # Check if this looks like a sub-field (provenance:, definition:, etc.)
        # Handles both "  provenance:" and "  - provenance:" formats
        if re.match(r'^[-\*•]?\s*(provenance|definition|reasoning|parent_class|differentia|key_relationships)\d*:', stripped, re.IGNORECASE):
            rest_lines = lines[i:]
            break

        # If we haven't found a value yet, this might be the value or description
        if not value:
            # Check for patterns like "is X" or "The X is Y" to extract actual value
            # For mistral format like "The computational/mathematical framework used is not explicitly mentioned"
            value = stripped

            # Try to extract a more specific value from description
            # Pattern: "is X" where X is the actual value
            is_match = re.search(r'\bis\s+(?:a\s+)?([A-Za-z][A-Za-z0-9\s\-\/]+?)(?:\.|,|$)', stripped)
            if is_match:
                extracted = is_match.group(1).strip()
                # Only use if it's a reasonable length (not a full sentence)
                if len(extracted) < 50 and extracted.lower() not in ('not', 'the', 'a', 'an'):
                    value = extracted
        else:
            rest_lines.append(line)

    rest_content = '\n'.join(rest_lines)
    # Normalize: convert "  - provenance:" to "  provenance:" for consistent parsing
    rest_content_normalized = re.sub(r'\n(\s*)[-\*•]\s*(provenance|definition|reasoning|parent_class|differentia|key_relationships)', r'\n\1\2', rest_content)
    has_numbered = bool(re.search(r'\n\s*(?:provenance|parent_class|differentia|key_relationships|definition|reasoning)1:', '\n' + rest_content_normalized, re.IGNORECASE))

    if has_numbered:
        provenances = {}
        parent_classes = {}
        differentias = {}
        key_relationships_dict = {}
        definitions = {}
        reasonings = {}

        for i in range(1, 20):
            prov_pattern = rf'\n\s+provenance{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'
            parent_pattern = rf'\n\s+parent_class{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'
            diff_pattern = rf'\n\s+differentia{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'
            rel_pattern = rf'\n\s+key_relationships{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'
            defn_pattern = rf'\n\s+definition{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'
            reas_pattern = rf'\n\s+reasoning{i}:\s*(.+?)(?=\n\s+\w+\d*:|$)'

            prov_match = re.search(prov_pattern, '\n' + rest_content_normalized, re.DOTALL)
            parent_match = re.search(parent_pattern, '\n' + rest_content_normalized, re.DOTALL)
            diff_match = re.search(diff_pattern, '\n' + rest_content_normalized, re.DOTALL)
            rel_match = re.search(rel_pattern, '\n' + rest_content_normalized, re.DOTALL)
            defn_match = re.search(defn_pattern, '\n' + rest_content_normalized, re.DOTALL)
            reas_match = re.search(reas_pattern, '\n' + rest_content_normalized, re.DOTALL)

            if prov_match:
                provenances[i] = prov_match.group(1).strip()
            if parent_match:
                parent_classes[i] = parent_match.group(1).strip()
            if diff_match:
                differentias[i] = diff_match.group(1).strip()
            if rel_match:
                key_relationships_dict[i] = rel_match.group(1).strip()
            if defn_match:
                definitions[i] = defn_match.group(1).strip()
            if reas_match:
                reasonings[i] = reas_match.group(1).strip()

            if not (prov_match or parent_match or diff_match or rel_match or defn_match or reas_match):
                break

        definition = '; '.join([f"{definitions[k]}" for k in sorted(definitions.keys())]) if definitions else None
        provenance = provenances.get(1) if provenances else None
        parent_class = parent_classes.get(1) if parent_classes else None
        differentia = differentias.get(1) if differentias else None
        key_relationships = key_relationships_dict.get(1) if key_relationships_dict else None
        reasoning = reasonings.get(1) if reasonings else None

        return {
            'value': value,
            'provenance': provenance,
            'parent_class': parent_class,
            'differentia': differentia,
            'key_relationships': key_relationships,
            'definition': definition,
            'reasoning': reasoning,
            'provenances': provenances,
            'parent_classes': parent_classes,
            'differentias': differentias,
            'key_relationships_all': key_relationships_dict,
            'definitions': definitions,
            'reasonings': reasonings,
            'ontologies': [],
            'domains': domains,
        }

    provenance_match = re.search(r'\n\s*(?:provenance|citation):\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    parent_class_match = re.search(r'\n\s*parent_class:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    differentia_match = re.search(r'\n\s*differentia:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    key_relationships_match = re.search(r'\n\s*key_relationships:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    definition_match = re.search(r'\n\s*definition:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    reasoning_match = re.search(r'\n\s*reasoning:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)
    ontologies_match = re.search(r'\n\s*ontologies:\s*(.+?)(?=\n\s*\w+:|$)', '\n' + rest_content_normalized, re.DOTALL | re.IGNORECASE)

    provenance = provenance_match.group(1).strip() if provenance_match else None
    parent_class = parent_class_match.group(1).strip() if parent_class_match else None
    differentia = differentia_match.group(1).strip() if differentia_match else None
    key_relationships = key_relationships_match.group(1).strip() if key_relationships_match else None
    definition = definition_match.group(1).strip() if definition_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    ontologies_str = ontologies_match.group(1).strip() if ontologies_match else None

    ontologies = []
    if ontologies_str:
        ontologies_str = ontologies_str.strip('[]"\'')
        ontologies = [ont.strip().strip('"\'') for ont in ontologies_str.split(',') if ont.strip()]

    is_multi_value = ',' in value
    if is_multi_value and (provenance or definition or reasoning):
        num_values = len([v.strip() for v in value.split(',') if v.strip()])
        LOGGER.warning(
            "FORMAT ERROR: '%s' has %d values (%s) but uses un-numbered fields. "
            "LLM should use provenance1/provenance2/etc for each value. "
            "Only the first value's metadata will be used.",
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


def parse_and_display_extracted_data(response_content: str, *, logger: logging.Logger = LOGGER):
    logger.info("LLM response:\n%s", response_content)
    extracted_data = parse_llm_response(response_content)
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
