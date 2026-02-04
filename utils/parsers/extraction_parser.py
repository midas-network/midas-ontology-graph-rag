from __future__ import annotations
import re
from typing import Dict
from utils.modeling_domains import identify_modeling_domains
import logging

LOGGER = logging.getLogger("midas-llm")


def parse_llm_response(response_text: str) -> Dict[str, Dict[str, str]]:
    attributes: Dict[str, Dict[str, str]] = {}
    lines = response_text.split('\n')
    current_attr = None
    current_content = []

    for line in lines:
        if line and not line[0].isspace() and re.match(r'^[a-z_]+:\s', line):
            if current_attr:
                attributes[current_attr] = parse_attribute_block(current_attr, '\n'.join(current_content))
            match = re.match(r'^([a-z_]+):\s(.*)$', line)
            if match:
                current_attr = match.group(1)
                current_content = [match.group(2)]
        elif current_attr:
            current_content.append(line)

    if current_attr:
        attributes[current_attr] = parse_attribute_block(current_attr, '\n'.join(current_content))
    return attributes


def parse_attribute_block(attr: str, content: str) -> Dict:
    domains = identify_modeling_domains(attr)
    lines = content.split('\n')
    value = lines[0].strip() if lines else ''
    rest_content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
    has_numbered = bool(re.search(r'\n\s+(?:provenance|parent_class|differentia|key_relationships|definition|reasoning)1:', '\n' + rest_content))

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

            prov_match = re.search(prov_pattern, '\n' + rest_content, re.DOTALL)
            parent_match = re.search(parent_pattern, '\n' + rest_content, re.DOTALL)
            diff_match = re.search(diff_pattern, '\n' + rest_content, re.DOTALL)
            rel_match = re.search(rel_pattern, '\n' + rest_content, re.DOTALL)
            defn_match = re.search(defn_pattern, '\n' + rest_content, re.DOTALL)
            reas_match = re.search(reas_pattern, '\n' + rest_content, re.DOTALL)

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

    provenance_match = re.search(r'\n\s+(?:provenance|citation):\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    parent_class_match = re.search(r'\n\s+parent_class:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    differentia_match = re.search(r'\n\s+differentia:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    key_relationships_match = re.search(r'\n\s+key_relationships:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    definition_match = re.search(r'\n\s+definition:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    reasoning_match = re.search(r'\n\s+reasoning:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)
    ontologies_match = re.search(r'\n\s+ontologies:\s*(.+?)(?=\n\s+\w+:|$)', '\n' + rest_content, re.DOTALL)

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
        LOGGER.warning(
            "Attribute '%s' has multiple values but missing numbered provenance/definition/reasoning. Values: %s",
            attr,
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
        domains_str = ', '.join(data['domains'])
        ontologies_str = ', '.join(data.get('ontologies', [])) if data.get('ontologies') else 'None'
        logger.info("[%s]: %s", attr, data['value'])
        logger.info("   Domains: %s", domains_str)
        logger.info("   Recommended Ontologies: %s", ontologies_str)
        if data.get('provenance'):
            logger.info("   Provenance: %s", data['provenance'])
        if data.get('definition'):
            logger.info("   Definition: %s", data['definition'])
        if data.get('reasoning'):
            logger.info("   Reasoning: %s", data['reasoning'])
    return extracted_data
