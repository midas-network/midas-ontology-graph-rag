from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger("midas-llm")


def _update_latest_folder(output_dir: str, run_dir: str, logger: logging.Logger) -> None:
    """Update the /latest/ folder to contain copies of the latest run output."""
    latest_dir = os.path.join(output_dir, "latest")

    # Remove existing latest folder
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)

    # Copy run_dir contents to latest
    shutil.copytree(run_dir, latest_dir)
    logger.info("Latest output copied to: %s", latest_dir)


def generate_reports(
    extracted_data: dict[str, Any],
    lookup_results: list[dict[str, Any]],
    response_content: str,
    llm_model: str,
    abstract_path: str,
    *,
    logger: logging.Logger = LOGGER,
    output_dir: str = "output/extract_concepts/runs",
    all_model_results: list[dict[str, Any]] | None = None,
    prompt_text: str = "",
    generate_html: bool = True,
    generate_json: bool = True,
) -> tuple[str, str, dict[str, Any]]:
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run output directory: %s", run_dir)

    # Save prompt
    if prompt_text:
        prompt_file = os.path.join(run_dir, "prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        logger.info("Prompt saved to: %s", prompt_file)

    # Save each model's response
    if all_model_results:
        for result in all_model_results:
            model_name = result["model"].replace(":", "_").replace("/", "_")
            response_file = os.path.join(run_dir, f"{model_name}-response.txt")
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(result.get("response", ""))
            logger.info("Model response saved to: %s", response_file)
    else:
        # Single model case
        model_name = llm_model.replace(":", "_").replace("/", "_")
        response_file = os.path.join(run_dir, f"{model_name}-response.txt")
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_content)
        logger.info("Model response saved to: %s", response_file)

    output_html = ""
    if generate_html:
        output_html = os.path.join(run_dir, "extraction_report.html")
        logger.info("Generating HTML report: %s", output_html)
        generate_html_report(extracted_data, lookup_results, response_content, output_html)
        logger.info("HTML report generated: %s", output_html)
    else:
        logger.info("HTML report generation disabled")

    json_output_file = ""
    json_output: dict[str, Any] = {}

    if generate_json:
        json_output_file = os.path.join(run_dir, "extraction.json")
        logger.info("Generating JSON output: %s", json_output_file)

        json_output = {
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

        # Add multi-model comparison if available
        if all_model_results and len(all_model_results) > 1:
            json_output["multi_model_comparison"] = []
            for result in all_model_results:
                model_summary = {
                    "model": result["model"],
                    "success": result["success"],
                    "attributes_extracted": len(result["extracted_data"]) if result["success"] else 0,
                    "raw_response": result["response"][:2000] if result["response"] else "",  # truncate
                }
                if not result["success"]:
                    model_summary["error"] = result.get("error", "Unknown error")
                if result["success"]:
                    model_summary["extracted_attributes"] = result["extracted_data"]
                json_output["multi_model_comparison"].append(model_summary)

        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        logger.info("JSON output written to: %s", json_output_file)
    else:
        logger.info("JSON output generation disabled")

    # Update /latest/ folder with current run output
    _update_latest_folder(output_dir, run_dir, logger)

    return output_html, json_output_file, json_output
