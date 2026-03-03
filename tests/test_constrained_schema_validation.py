from __future__ import annotations

import sys
from pathlib import Path

# Allow tests to import from src/ without requiring editable install.
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from midas_llm.scripts.evaluate_gold_standard import (  # noqa: E402
    parse_constrained_response,
    validate_constrained_payload,
)


SCHEMA = {
    "type": "object",
    "properties": {
        "model_type": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"enum": ["agent-based", "compartmental"]},
                },
                "reasoning": {"type": "string"},
            },
            "required": ["values", "reasoning"],
        }
    },
    "required": ["model_type"],
}


def test_validate_constrained_payload_accepts_valid_payload() -> None:
    payload = {
        "model_type": {
            "values": ["agent-based"],
            "reasoning": "The abstract states this directly.",
        }
    }
    assert validate_constrained_payload(payload, SCHEMA) is True


def test_validate_constrained_payload_rejects_invalid_enum() -> None:
    payload = {
        "model_type": {
            "values": ["unsupported-model-type"],
            "reasoning": "Invalid value for this schema.",
        }
    }
    assert validate_constrained_payload(payload, SCHEMA) is False


def test_parse_constrained_response_schema_validation_optional() -> None:
    invalid_content = """
    {
      "model_type": {
        "values": ["unsupported-model-type"],
        "reasoning": "Invalid under schema"
      }
    }
    """

    # Validation enabled: invalid payload is rejected.
    rejected = parse_constrained_response(
        invalid_content,
        validation_schema=SCHEMA,
        validate_schema=True,
    )
    assert rejected == {}

    # Validation disabled: parser still returns structured output.
    accepted = parse_constrained_response(
        invalid_content,
        validation_schema=SCHEMA,
        validate_schema=False,
    )
    assert accepted["model_type"]["value"] == "unsupported-model-type"
