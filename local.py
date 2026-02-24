#!/usr/bin/env python3
"""
Test script to send a sample prompt to an Ollama server at 192.168.68.55.
"""
import logging
import sys
from pathlib import Path

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.llm.llm_client import send_to_llm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("midas-llm-test")

OLLAMA_HOST = "http://192.168.68.55:11434"
MODEL = "qwen3-coder:30b-a3b-q4_K_M"
TIMEOUT = 60  # seconds

PROMPT = """
Extract the main concepts from the following text.\n\nText: The quick brown fox jumps over the lazy dog.
"""

def main():
    LOGGER.info(f"Sending test prompt to Ollama server at {OLLAMA_HOST} using model {MODEL}")
    try:
        response = send_to_llm(prompt=PROMPT, llm_model=MODEL, ollama_host=OLLAMA_HOST, timeout_seconds=TIMEOUT, logger=LOGGER)
        print("Response from LLM:\n", response.content)
    except Exception as exc:
        LOGGER.error(f"Failed to get response from LLM: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()

