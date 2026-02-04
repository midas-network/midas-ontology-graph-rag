"""midas-llm package shim for src/ layout.

This injects the project root onto sys.path so existing modules (utils, model, etc.)
continue to resolve while we transition to a full src/ package layout.
"""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
