from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("midas-llm")

# Default path to few-shot examples file
DEFAULT_FEW_SHOT_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "resources" / "prompts" / "extract_concepts" / "few-shot.txt"


def load_few_shot_examples(
    use_examples: bool,
    few_shot_path: str | Path | None = None,
    *,
    logger: logging.Logger = LOGGER,
) -> str | None:
    """Load few-shot examples from the few-shot.txt file.

    Args:
        use_examples: Whether to load examples or skip.
        few_shot_path: Path to the few-shot examples file. Defaults to resources/prompts/extract_concepts/few-shot.txt.
        logger: Logger instance.

    Returns:
        Content of few-shot.txt or None if use_examples is False.
    """
    if not use_examples:
        logger.info("Running without few-shot examples")
        return None

    path = Path(few_shot_path) if few_shot_path else DEFAULT_FEW_SHOT_PATH
    logger.info("Loading few-shot examples from %s", path)

    if not path.exists():
        logger.warning("Few-shot examples file not found: %s", path)
        return None

    content = path.read_text(encoding="utf-8")
    logger.info("Loaded few-shot examples (%d characters)", len(content))
    return content
