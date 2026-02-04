from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any
import logging

from utils.html_reports import generate_html_report

LOGGER = logging.getLogger("midas-llm")


def generate_reports(
    extracted_data: Dict,
    lookup_results: List[Dict],
    response_content: str,
    llm_model: str,
    abstract_path: str,
    *,
    logger: logging.Logger = LOGGER,
    output_dir: str = "data/output/runs",
) -> Tuple[str, str, Dict[str, Any]]:
    log_dir = output_dir
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_html = os.path.join(log_dir, f"{timestamp}_extraction_report.html")
    logger.info("Generating HTML report: %s", output_html)
    generate_html_report(extracted_data, lookup_results, response_content, output_html)
    logger.info("HTML report generated: %s", output_html)

    json_output_file = os.path.join(log_dir, f"{timestamp}.json")
    logger.info("Generating JSON output: %s", json_output_file)

    json_output: Dict[str, Any] = {
        "extraction_timestamp": datetime.now().isoformat(),
        "llm_model": llm_model,
        "abstract_source": abstract_path,
        "total_attributes": len(extracted_data),
        "attributes": {},
    }

    for attr, data in extracted_data.items():
        json_output["attributes"][attr] = {
            "value": data.get('value'),
            "domains": data.get('domains', []),
            "recommended_ontologies": data.get('ontologies', []),
            "provenance": data.get('provenance'),
            "concept": data.get('concept'),
            "reasoning": data.get('reasoning'),
            "ontology_matches": [],
        }

        attr_matches = [r for r in lookup_results if r.get('attribute') == attr]
        for match in attr_matches:
            ont_def = match.get('ontology_definition') or match.get('definition')
            match_entry = {
                "ontology": match.get('ontology'),
                "identifier": match.get('identifier'),
                "matched_term": match.get('matched_term'),
                "status": match.get('status', 'found'),
                "ontology_definition": ont_def,
                "llm_definition": match.get('llm_definition'),
                "similarity": match.get('similarity'),
                "similarity_method": match.get('similarity_method'),
            }
            json_output["attributes"][attr]["ontology_matches"].append(match_entry)

    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    logger.info("JSON output written to: %s", json_output_file)
    return output_html, json_output_file, json_output
