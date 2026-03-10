from __future__ import annotations

import os

DEFAULT_EMBEDDING_MODELS = [
    "BAAI/bge-m3",
    "FremyCompany/BioLORD-2023-C",
]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def parse_model_list(value: str) -> list[str]:
    """Parse comma-separated model list, filtering empty strings."""
    if not value:
        return []
    return [model.strip() for model in value.split(",") if model.strip()]


def default_embedding_models() -> list[str]:
    """Resolve embedding model list from env with backward-compatible fallback."""
    multi = parse_model_list(os.getenv("EMBEDDING_MODELS", ""))
    if multi:
        return multi

    legacy = parse_model_list(os.getenv("EMBEDDING_MODEL", ""))
    if legacy:
        return legacy

    return list(DEFAULT_EMBEDDING_MODELS)


# Backward-compatible internal helper names.
_env_bool = env_bool
_env_int = env_int
_parse_model_list = parse_model_list
_default_embedding_models = default_embedding_models
