from __future__ import annotations
from typing import Optional, List
import os
import logging

LOGGER = logging.getLogger("midas-llm")


def read_abstract(abstract_path: str) -> str:
    with open(abstract_path, "r", encoding="utf-8") as f:
        return f.read()


def build_query(
    abstract: str,
    prompt_dir: str = "data/prompt_text",
    *,
    include_examples: bool = True,
    include_reminders: bool = True,
    include_few_shot: bool = False,
    include_fields: bool = True,
    include_ontologies: bool = False,
) -> str:
    section_files: List[str] = ["instructions.txt"]
    if include_examples:
        section_files.append("format_examples.txt")
    if include_few_shot:
        section_files.append("few-shot.txt")
    if include_fields:
        section_files.append("fields.txt")
    if include_ontologies:
        section_files.append("ontologies.txt")
    if include_reminders:
        section_files.append("reminders.txt")

    prompt_parts = []
    for filename in section_files:
        path = os.path.join(prompt_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                prompt_parts.append(f.read().rstrip())

    query_template = "\n\n".join(part for part in prompt_parts if part)
    return query_template + "\n\n" + abstract


def prepare_complete_prompt(query: str, ontology_context: str, examples_context: Optional[str] = None) -> str:
    prompt_parts = []
    if examples_context:
        prompt_parts.extend(["", "=== Few-Shot Examples (for reference) ===", examples_context])
    prompt_parts.extend(["", "=== Your Task ===", query])
    return "\n".join(prompt_parts)


def prepare_and_display_prompt(query: str, ontology_context: str, examples_context: Optional[str], *, logger: logging.Logger = LOGGER):
    logger.info("Building prompt_text...")
    prompt = prepare_complete_prompt(query, ontology_context, examples_context)
    logger.info("Prompt prepared (%d characters)", len(prompt))
    logger.debug("ACTUAL PROMPT:\n%s", prompt)
    return prompt


def load_and_prepare_abstract(
    abstract_path: str,
    include_examples: bool = True,
    include_reminders: bool = True,
    include_few_shot: bool = False,
    include_fields: bool = True,
    include_ontologies: bool = False,
    *,
    logger: logging.Logger = LOGGER,
):
    logger.info("Loading abstract from %s...", abstract_path)
    abstract = read_abstract(abstract_path)
    query = build_query(
        abstract,
        include_examples=include_examples,
        include_reminders=include_reminders,
        include_few_shot=include_few_shot,
        include_fields=include_fields,
        include_ontologies=include_ontologies,
    )
    logger.info("Loaded abstract (%d characters)", len(abstract))
    return abstract, query
