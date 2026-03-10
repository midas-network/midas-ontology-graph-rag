from __future__ import annotations
from typing import List
import os
from pathlib import Path
import logging

# formatters now under concept_extractor if needed


LOGGER = logging.getLogger("midas-llm")


def _resolve_prompt_dir(prompt_dir: str) -> Path:
    """Resolve prompt directory, preferring absolute; fall back to repo-relative."""
    candidate = Path(prompt_dir)
    if candidate.is_dir():
        return candidate

    # Try relative to repo root (src/../../..)
    repo_root = Path(__file__).resolve().parents[4]
    fallback = repo_root / prompt_dir
    return fallback if fallback.is_dir() else candidate


def _read_section(path: Path) -> str:
    if not path.is_file():
        LOGGER.warning("Prompt section not found: %s", path)
        return ""
    with path.open("r", encoding="utf-8") as f:
        return f.read().rstrip()


def read_abstract(abstract_path: str) -> str:
    with open(abstract_path, "r", encoding="utf-8") as f:
        return f.read()


def build_query(
    abstract: str,
    prompt_dir: str = "resources/prompts/extract_concepts",
    *,
    include_examples: bool = True,
    include_reminders: bool = True,
    include_few_shot: bool = False,
    include_fields: bool = True,
    include_ontologies: bool = False,
    simple_prompt: bool = False,
) -> str:
    # Simple prompt mode: use condensed instructions, skip extra sections
    if simple_prompt:
        # Direct completion-style prompt
        prompt = f"""Extract the following information from this abstract about an infectious disease modeling study.

ABSTRACT:
{abstract}

Please provide:
1. Model type (e.g., agent-based, SEIR, compartmental, network, statistical)
2. Disease or pathogen studied
3. Population studied
4. Geographic scope
5. Main study goal
6. Interventions evaluated (if any)
7. Key outcomes measured

Format each answer on its own line."""
        LOGGER.info("Prompt files used: [simple_prompt] (no external files)")
        return prompt
        # Fall through to normal mode if simple instructions don't exist

    prompt_base = _resolve_prompt_dir(prompt_dir)

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

    # Log which files are actually used (after checking existence)
    used_files = []
    prompt_parts = []
    for filename in section_files:
        path = prompt_base / filename
        content = _read_section(path)
        if content:
            used_files.append(str(path))
            prompt_parts.append(content)
        else:
            LOGGER.warning("Prompt section not found (skipped): %s", str(path))

    LOGGER.info("Prompt files used: [%s]", ", ".join(used_files))
    LOGGER.info("Prompt directory: %s", prompt_base)

    query_template = "\n\n".join(part for part in prompt_parts if part)
    return query_template + "\n\n" + abstract


def prepare_complete_prompt(query: str, ontology_context: str) -> str:
    prompt_parts = []
    prompt_parts.extend(["=== Your Task ===", query])
    return "\n".join(prompt_parts)


def prepare_and_display_prompt(query: str, ontology_context: str, *, logger: logging.Logger = LOGGER):
    logger.info("Building prompt_text...")
    prompt = prepare_complete_prompt(query, ontology_context)
    logger.info("Prompt prepared (%d characters)", len(prompt))
    logger.debug("ACTUAL PROMPT:\n%s", prompt)
    return prompt


def load_and_prepare_abstract(
    abstract_path: str,
    include_examples: bool = True,
    include_reminders: bool = True,
    include_few_shot: bool = True,
    include_fields: bool = True,
    include_ontologies: bool = True,
    simple_prompt: bool = False,
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
        simple_prompt=simple_prompt,
    )
    logger.info("Loaded abstract (%d characters)", len(abstract))
    return abstract, query
