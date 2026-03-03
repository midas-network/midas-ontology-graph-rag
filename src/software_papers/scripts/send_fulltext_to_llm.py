#!/usr/bin/env python3
"""
Batch sender: read JSON fulltext files, build prompts, send to LLM, and save responses.
Uses the prompt template at resources/prompts/software_extract/example_prompt.txt by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.llm.llm_client import send_to_llm
from midas_llm.utils.llm.llm_utils import autodetect_llm_host

LOGGER = logging.getLogger("midas-llm")

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class SafeDict(dict):
    """Dictionary that returns empty string for missing keys during format_map."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - simple safety
        return ""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_prompt_template(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def build_prompt(template: str, paper: Dict[str, Any], max_body_chars: int | None = None) -> str:
    body_text = paper.get("body") or ""
    if max_body_chars and len(body_text) > max_body_chars:
        body_text = body_text[: max_body_chars] + "\n...[truncated]"

    result = template
    result = result.replace("{paperID}", str(paper.get("paperID") or ""))
    result = result.replace("{pmcid}", str(paper.get("pmcid") or ""))
    result = result.replace("{title}", str(paper.get("title") or paper.get("original_title") or ""))
    result = result.replace("{abstract}", str(paper.get("abstract") or ""))
    result = result.replace("{body}", body_text)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send fulltext JSON files to an LLM for concept extraction.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "resources" / "software_papers_fulltext",
        help="Directory containing JSON fulltext files.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Process a single specific JSON file instead of the entire input directory.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=REPO_ROOT / "resources" / "prompts" / "software_extract" / "example_prompt.txt",
        help="Prompt template file to use (supports {paperID}, {pmcid}, {title}, {abstract}, {body}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "runs" / "software_extract",
        help="Where to write LLM responses as JSON.",
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1:70b"), help="LLM model name.")
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host (http://host:port). Will auto-detect if unreachable.",
    )
    parser.add_argument("--timeout", type=int, default=300000, help="LLM timeout seconds per call.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of files to process.")
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=0,
        help="If set, truncate body text to this many characters before sending.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build prompts only; do not call the LLM.")
    return parser.parse_args()


def save_result(output_dir: Path, source_file: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source_file.stem}_llm.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Saved result -> %s", out_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Handle single file mode
    if args.file:
        if not args.file.is_file():
            LOGGER.error("File not found: %s", args.file)
            raise SystemExit(1)
        json_files = [args.file]
    else:
        if not args.input_dir.is_dir():
            LOGGER.error("Input directory not found: %s", args.input_dir)
            raise SystemExit(1)
        json_files = sorted(p for p in args.input_dir.glob("*.json") if not p.name.startswith("_"))
        if args.limit:
            json_files = json_files[: args.limit]

    if not args.prompt_file.is_file():
        LOGGER.error("Prompt file not found: %s", args.prompt_file)
        raise SystemExit(1)

    prompt_template = load_prompt_template(args.prompt_file)

    detected_host = autodetect_llm_host(args.host, logger=LOGGER)
    llm_host = detected_host or args.host
    LOGGER.info("Using LLM host: %s", llm_host)
    LOGGER.info("Using model: %s", args.model)
    LOGGER.info("Found %d files to process", len(json_files))

    for idx, path in enumerate(json_files, start=1):
        LOGGER.info("[%d/%d] Processing %s", idx, len(json_files), path.name)
        try:
            paper = load_json(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to read %s: %s", path, exc)
            continue

        prompt = build_prompt(prompt_template, paper, max_body_chars=args.max_body_chars or None)

        result: Dict[str, Any] = {
            "paper_id": paper.get("paperID"),
            "pmcid": paper.get("pmcid"),
            "source_file": str(path),
            "llm_model": args.model,
            "llm_host": llm_host,
            "prompt": prompt,
        }

        if args.dry_run:
            LOGGER.info("Dry-run: skipping LLM call for %s", path.name)
            result.update({"response": None, "success": True, "error": None})
            save_result(args.output_dir, path, result)
            continue

        try:
            response = send_to_llm(prompt=prompt, llm_model=args.model, llm_host=llm_host, timeout_seconds=args.timeout, logger=LOGGER)
            result.update({"response": response.content, "success": True, "error": None})
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("LLM call failed for %s: %s", path.name, exc)
            result.update({"response": None, "success": False, "error": str(exc)})

        save_result(args.output_dir, path, result)


if __name__ == "__main__":
    main()
