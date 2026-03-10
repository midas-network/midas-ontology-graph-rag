from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from .env import parse_model_list

LOGGER = logging.getLogger("midas-llm")


def find_config_file() -> Path | None:
    """Find config file in priority order.

    Search order:
    1. MIDAS_CONFIG (if set)
    2. ./config.yaml
    3. <repo_root>/config.yaml
    4. ~/.midas-llm.yaml
    """
    candidates: list[Path] = []
    midas_config = os.getenv("MIDAS_CONFIG")
    if midas_config:
        candidates.append(Path(midas_config).expanduser())

    candidates.extend(
        [
            Path("config.yaml"),
            Path(__file__).resolve().parents[4] / "config.yaml",
            Path.home() / ".midas-llm.yaml",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    """Load YAML config from disk and return a flat mapping."""
    if path is None or not path.is_file():
        return {}

    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.warning("Failed to load config YAML at %s: %s", path, exc)
        return {}

    if config is None:
        return {}
    if not isinstance(config, dict):
        LOGGER.warning("Config YAML at %s must be a mapping; ignoring file.", path)
        return {}
    return config


def yaml_bool(value: Any) -> bool:
    """Parse booleans from YAML-native bools or common string forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"expected bool-compatible value, got {value!r}")


def yaml_string_list(value: Any) -> list[str]:
    """Parse list[str] from YAML list or comma-separated string."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return parse_model_list(value)
    raise ValueError(f"expected list[str] or comma-separated string, got {value!r}")


def yaml_coerce_value(field_name: str, value: Any, annotation: Any) -> Any:
    """Coerce YAML field values to ExtractionConfig field types."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is bool:
        return yaml_bool(value)
    if annotation is int:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid int override")
        return int(value)
    if annotation is Path:
        return value if isinstance(value, Path) else Path(str(value))
    if annotation is str:
        return str(value)
    if origin is list and args == (str,):
        return yaml_string_list(value)

    LOGGER.warning(
        "Unhandled annotation for YAML field '%s' (%s); using raw value.",
        field_name,
        annotation,
    )
    return value


def has_env_override(field_name: str) -> bool:
    """Return True if canonical env var for a field is explicitly set."""
    return os.getenv(field_name.upper()) is not None


# Backward-compatible internal helper names.
_find_config_file = find_config_file
_load_yaml_config = load_yaml_config
_yaml_bool = yaml_bool
_yaml_string_list = yaml_string_list
_yaml_coerce_value = yaml_coerce_value
_has_env_override = has_env_override
