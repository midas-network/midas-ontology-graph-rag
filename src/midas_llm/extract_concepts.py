from __future__ import annotations

"""Package entrypoint for concept extraction."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:  # allow `python src/midas_llm/extract_concepts.py`
    # This file is at src/midas_llm/extract_concepts.py
    # parents[0] = midas_llm/, parents[1] = src/
    _this_file = Path(__file__).resolve()
    _src = _this_file.parent.parent  # go up to src/
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    __package__ = "midas_llm"

from .utils.config import ExtractionConfig
from .utils.logging.logger import configure_logging
from .utils.loaders.ontology_loader import (
    load_midas_ontology,
    start_background_ontology_loading,
    finalize_ontology_loading,
)
from .utils.llm.llm_utils import probe_llm_host, autodetect_llm_host
from .utils.llm.llm_client import send_to_llm, test_respond_ok
from .utils.ontology_linker.link_concepts import link_concepts_to_ontologies
from .utils.prompt.builders import prepare_and_display_prompt, load_and_prepare_abstract
from .utils.parsers.extraction_parser import parse_and_display_extracted_data
from .utils.reporting.run_reports import generate_reports


def _log_config(logger: logging.Logger, config: ExtractionConfig) -> None:
    """Log current configuration."""
    if config.show_config:
        config.log_config(logger)


def _run_sanity_checks(logger: logging.Logger, config: ExtractionConfig) -> bool:
    """Run pre-flight sanity checks on LLM connectivity."""
    logger.info("Running sanity check (test_respond_ok)...")
    ok, raw = test_respond_ok(config.llm_model, config.llm_host, timeout_seconds=30, logger=logger)
    logger.info("Sanity check OK: %s; response: %s", ok, raw[:120])
    if not ok:
        logger.error("Sanity check failed; probing host for diagnostics...")
        probe_llm_host(config.llm_host)
        return False
    return True


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract concepts from scientific abstracts using LLMs.")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Show the prompt only without sending to LLM",
    )
    return parser.parse_args()


def main() -> None:
    """Main extraction workflow."""
    args = _parse_args()
    config = ExtractionConfig()

    # Suppress logging if only showing prompt
    if args.prompt_only:
        logger = logging.getLogger("midas-llm")
        logger.setLevel(logging.CRITICAL)
    else:
        logger = configure_logging(debug=config.debug)
        logger.info("Starting attribute extraction...")

    # Skip LLM-related initialization if only showing prompt
    if not args.prompt_only:
        detected_host = autodetect_llm_host(config.llm_host, logger=logger)
        if detected_host:
            config.llm_host = detected_host

        _log_config(logger, config)

        probe_llm_host(config.llm_host, logger=logger)

        if config.run_sanity_check and not config.test_mode:
            if not _run_sanity_checks(logger, config):
                logger.error("Aborting run due to failed sanity check.")
                return

        if config.test_mode:
            logger.info("TEST_MODE enabled; performing LLM ping only.")
            ok, raw = test_respond_ok(config.llm_model, config.llm_host, timeout_seconds=30, logger=logger)
            logger.info("Test mode result ok=%s response=%s", ok, raw[:120])
            return

    if config.enable_midas_ontology:
        midas_graph, ontology_context, label_map, classes, properties = load_midas_ontology(
            str(config.ontology_path)
        )
    else:
        logger.info("MIDAS ontology loading disabled")
        midas_graph, ontology_context, label_map, classes, properties = None, "", {}, [], []

    abstract, query = load_and_prepare_abstract(
        str(config.abstract_path),
        include_examples=config.prompt_include_format_examples,
        include_reminders=config.prompt_include_reminders,
        include_few_shot=config.prompt_include_few_shot,
        include_fields=config.prompt_include_fields,
        include_ontologies=config.prompt_include_ontologies,
        simple_prompt=config.prompt_simple_prompt,
        logger=logger,
    )

    prompt = prepare_and_display_prompt(query, ontology_context, logger=logger)

    if args.prompt_only:
        print(prompt)
        return

    repo_root = Path(__file__).resolve().parents[2]
    schema_candidates = [
        repo_root / "resources" / "schemas" / "midas_schema.json",
        repo_root / "midas_schema.json",
    ]
    constrained_schema = None
    for schema_path in schema_candidates:
        if schema_path.is_file():
            try:
                constrained_schema = json.loads(schema_path.read_text(encoding="utf-8"))
                logger.info("Using constrained schema: %s", schema_path)
                break
            except json.JSONDecodeError as e:
                logger.error("Invalid constrained schema JSON at %s: %s", schema_path, e)
                return
    if constrained_schema is None:
        logger.error(
            "Constrained schema not found. Expected one of: %s",
            ", ".join(str(p) for p in schema_candidates),
        )
        return

    if config.enable_ontology_linking:
        executor, background_tasks = start_background_ontology_loading()
    else:
        executor, background_tasks = None, None

    # Determine which models to use
    models_to_run = config.llm_marodels if config.llm_models else [config.llm_model]
    logger.info("Running extraction with %d model(s): %s", len(models_to_run), ", ".join(models_to_run))

    all_results = []
    for model in models_to_run:
        logger.info("=" * 60)
        logger.info("Running model: %s", model)
        logger.info("=" * 60)

        try:
            response = send_to_llm(
                prompt=prompt,
                llm_model=config.active_llm_model,
                llm_host=config.active_llm_host,
                timeout_seconds=config.llm_timeout,
                api_type=config.llm_api_type,
                json_schema=constrained_schema,
            )

            extracted_data = parse_and_display_extracted_data(response.content, logger=logger)

            all_results.append({
                "model": model,
                "response": response.content,
                "extracted_data": extracted_data,
                "success": True,
            })
        except Exception as e:
            logger.error("Model %s failed: %s", model, e)
            all_results.append({
                "model": model,
                "response": "",
                "extracted_data": {},
                "success": False,
                "error": str(e),
            })

    if config.enable_ontology_linking:
        ontologies = finalize_ontology_loading(background_tasks, executor, midas_graph)
    else:
        logger.info("Ontology linking disabled")
        ontologies = None

    # Use the first successful result for reports, or the first result if all failed
    primary_result = next((r for r in all_results if r["success"]), all_results[0] if all_results else None)

    if primary_result:
        extracted_data = primary_result["extracted_data"]
        response_content = primary_result["response"]
        primary_model = primary_result["model"]
    else:
        extracted_data = {}
        response_content = ""
        primary_model = models_to_run[0] if models_to_run else config.llm_model

    if config.enable_ontology_linking and ontologies:
        lookup_results = link_concepts_to_ontologies(extracted_data, ontologies, logger=logger)
    else:
        logger.info("Skipping ontology linking (disabled)")
        lookup_results = []

    output_html, json_output_file, _ = generate_reports(
        extracted_data,
        lookup_results,
        response_content,
        primary_model,
        str(config.abstract_path),
        logger=logger,
        output_dir=str(config.output_dir),
        all_model_results=all_results,
        prompt_text=prompt,
        generate_html=config.generate_html_report,
        generate_json=config.generate_json_output,
    )

    logger.info("EXTRACTION COMPLETE | Run directory: %s", config.output_dir)

    # Log summary of all model results
    if len(models_to_run) > 1:
        logger.info("=" * 60)
        logger.info("MULTI-MODEL SUMMARY:")
        for result in all_results:
            status = "✓" if result["success"] else "✗"
            attr_count = len(result["extracted_data"]) if result["success"] else 0
            logger.info("  %s %s: %d attributes extracted", status, result["model"], attr_count)
        logger.info("=" * 60)


def cli() -> None:
    """CLI entrypoint."""
    main()
