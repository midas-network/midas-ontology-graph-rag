#!/usr/bin/env python3
"""Generate the exact full prompt that would be sent to the LLM.

This script is display-only: it does not call any LLM endpoint.
It reuses ExtractionConfig and existing prompt-building utilities.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.config import ExtractionConfig
from midas_llm.utils.loaders.ontology_loader import load_midas_ontology
from midas_llm.utils.logging.logger import configure_logging
from midas_llm.utils.prompt.builders import (
    load_and_prepare_abstract,
    prepare_and_display_prompt,
)

LOGGER = logging.getLogger("midas-llm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and display the full LLM prompt using current config settings."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to write the generated prompt text.",
    )
    parser.add_argument(
        "--print-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether config is logged (default: use SHOW_CONFIG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig()
    logger = configure_logging(debug=config.debug)

    should_print_config = config.show_config if args.print_config is None else args.print_config
    if should_print_config:
        config.log_config(logger)

    ontology_context = ""
    if config.enable_midas_ontology:
        try:
            _, ontology_context, _, _, _ = load_midas_ontology(str(config.ontology_path))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load MIDAS ontology context: %s", e)
            raise SystemExit(1)
    else:
        logger.info("MIDAS ontology loading disabled")

    _, query = load_and_prepare_abstract(
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

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(prompt, encoding="utf-8")
        logger.info("Prompt written to: %s", args.output_file)

    print(prompt)


if __name__ == "__main__":
    main()
