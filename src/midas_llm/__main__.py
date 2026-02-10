"""Allow running the package with `python -m midas_llm`."""

import sys
from pathlib import Path

# Add src/ to path if package not installed
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from midas_llm.extract_concepts import cli

if __name__ == "__main__":
    cli()
