#!/usr/bin/env python
"""Convenience script to run extraction.

For proper usage, install the package and run:
    pip install -e .
    midas-extract

Or use the module directly:
    PYTHONPATH=src python -m midas_llm
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for development without install
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from midas_llm.extract_concepts import cli

if __name__ == "__main__":
    cli()
