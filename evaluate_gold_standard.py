#!/usr/bin/env python3
"""Convenience wrapper to run gold-standard evaluation from repo root.

Preferred entrypoint after installation:
    midas-eval-gold
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from midas_llm.scripts.evaluate_gold_standard import main

if __name__ == "__main__":
    main()
