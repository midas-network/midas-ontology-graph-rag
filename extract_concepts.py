from __future__ import annotations

"""
Extract concepts from an abstract using ontology context and few-shot examples.
No RAG needed - ontology provided as static context, papers as examples.
Enhanced with concept-to-ontology linking and HTML report generation.
"""
import time
import os
import sys
import contextlib
import json
import logging

# Ensure project root is on sys.path for package imports when run as script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import ExtractionConfig
from utils.logging.logger import configure_logging
from utils.loaders.ontology_loader import (
    load_midas_ontology,
    start_background_ontology_loading,
    finalize_ontology_loading,
)
from utils.llm.llm_utils import probe_llm_host
from utils.llm.llm_client import send_to_llm, test_respond_ok
from utils.ontology_linker.link_concepts import link_concepts_to_ontologies
from utils.prompt.builders import (
    prepare_and_display_prompt,
    load_and_prepare_abstract,
)
from utils.prompt.examples import load_few_shot_examples
from utils.parsers.extraction_parser import (
    parse_and_display_extracted_data,
)
from utils.reporting.run_reports import generate_reports


def _log_config(logger: logging.Logger, config: ExtractionConfig) -> None:
    logger.info("Configuration: %s", config)


def _run_sanity_checks(logger: logging.Logger, config: ExtractionConfig) -> bool:
    logger.info("Running sanity check (test_respond_ok)...")
    ok, raw = test_respond_ok(config.llm_model, config.llm_host, timeout_seconds=30, logger=logger)
    logger.info("Sanity check OK: %s; response: %s", ok, raw[:120])
    if not ok:
        logger.error("Sanity check failed; probing host for diagnostics...")
        probe_llm_host(config.llm_host)
        return False
    return True


def main():
    """Entry point for attribute extraction run."""
    config = ExtractionConfig()
    logger = configure_logging(debug=config.debug)
    logger.info("Starting attribute extraction...")
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
        midas_graph, ontology_context, label_map, classes, properties = load_midas_ontology(str(config.ontology_path))
    else:
        logger.info("MIDAS ontology loading disabled")
        midas_graph, ontology_context, label_map, classes, properties = None, "", {}, [], []

    examples_context = load_few_shot_examples(str(config.papers_path), config.num_examples, config.use_examples, logger=logger)

    abstract, query = load_and_prepare_abstract(
        str(config.abstract_path),
        include_examples=config.include_format_examples,
        include_reminders=config.include_reminders,
        include_few_shot=config.include_few_shot,
        include_fields=config.include_fields,
        include_ontologies=config.include_ontologies,
        logger=logger,
    )

    prompt = prepare_and_display_prompt(query, ontology_context, examples_context, logger=logger)

    if config.enable_ontology_linking:
        executor, background_tasks = start_background_ontology_loading()
    else:
        executor, background_tasks = None, None

    response = send_to_llm(
        prompt=prompt,
        llm_model=config.llm_model,
        ollama_host=config.llm_host,
        timeout_seconds=config.llm_timeout,
        logger=logger,
    )

    if config.enable_ontology_linking:
        ontologies = finalize_ontology_loading(background_tasks, executor, midas_graph)
    else:
        logger.info("Ontology linking disabled")
        ontologies = None

    extracted_data = parse_and_display_extracted_data(response.content, logger=logger)

    if config.enable_ontology_linking and ontologies:
        lookup_results = link_concepts_to_ontologies(extracted_data, ontologies, logger=logger)
    else:
        logger.info("Skipping ontology linking (disabled)")
        lookup_results = []

    output_html, json_output_file, json_output = generate_reports(
        extracted_data, lookup_results, response.content, config.llm_model, str(config.abstract_path), logger=logger, output_dir=str(config.output_dir)
    )

    logger.info("EXTRACTION COMPLETE | HTML: %s | JSON: %s", output_html, json_output_file)


if __name__ == "__main__":
    from datetime import datetime

    class Tee:
        """Write stdout to multiple streams (console + log file)."""
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    log_dir = os.path.join("data", "output", "runs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            main()
            print(f"\nRun log captured at {log_path}")
