#!/usr/bin/env python3
"""Convenience wrapper to generate and display the full prompt from repo root."""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from midas_llm.scripts.generate_full_prompt import main

if __name__ == "__main__":
    main()
