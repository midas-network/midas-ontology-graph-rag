#!/usr/bin/env python3
"""Convenience wrapper to run send_fulltext_to_llm from repo root."""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from midas_llm.scripts.send_fulltext_to_llm import main

if __name__ == "__main__":
    main()

