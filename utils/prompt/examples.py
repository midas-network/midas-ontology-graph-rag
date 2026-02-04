from __future__ import annotations
import json
from formatters.papers import format_paper_examples
import logging

LOGGER = logging.getLogger("midas-llm")


def load_few_shot_examples(papers_path: str, num_examples: int, use_examples: bool, *, logger: logging.Logger = LOGGER):
    if not use_examples:
        logger.info("Running without few-shot examples")
        return None

    logger.info("Loading %d example papers from %s", num_examples, papers_path)
    with open(papers_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    examples_context = format_paper_examples(papers, num_examples)
    logger.info("Prepared %d few-shot examples", num_examples)
    return examples_context
