from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

from .env import DEFAULT_EMBEDDING_MODELS, default_embedding_models, env_bool, env_int, parse_model_list
from .yaml import find_config_file, has_env_override, load_yaml_config, yaml_coerce_value

LOGGER = logging.getLogger("midas-llm")


@dataclass
class ExtractionConfig:
    """Application configuration with env var overrides."""

    # Paths and I/O
    ontology_path: Path = field(default_factory=lambda: Path(os.getenv("ONTOLOGY_PATH", "resources/ontologies")))

    # Prompt options
    prompt_include_format_examples: bool = field(default_factory=lambda: env_bool("PROMPT_INCLUDE_FORMAT_EXAMPLES", False))
    prompt_include_reminders: bool = field(default_factory=lambda: env_bool("PROMPT_INCLUDE_REMINDERS", True))
    prompt_include_few_shot: bool = field(default_factory=lambda: env_bool("PROMPT_INCLUDE_FEW_SHOT", True))
    prompt_include_fields: bool = field(default_factory=lambda: env_bool("PROMPT_INCLUDE_FIELDS", True))
    prompt_include_ontologies: bool = field(default_factory=lambda: env_bool("PROMPT_INCLUDE_ONTOLOGIES", False))
    prompt_simple_prompt: bool = field(default_factory=lambda: env_bool("PROMPT_SIMPLE_PROMPT", False))

    # LLM settings - Ollama
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral:7b"))
    ollama_models: list[str] = field(default_factory=lambda: parse_model_list(os.getenv("OLLAMA_MODELS", "")))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # LLM settings - NIM / OpenAI-compatible
    nim_models: list[str] = field(default_factory=lambda: parse_model_list(
        os.getenv("NIM_MODELS", "nvidia/llama-3.1-nemotron-nano-8b-v1")
    ))
    nim_host: str = field(default_factory=lambda: os.getenv("NIM_HOST", "http://localhost:8000"))

    # LLM settings - shared
    llm_timeout: int = field(default_factory=lambda: env_int("LLM_TIMEOUT", 300000))
    llm_api_type: str = field(default_factory=lambda: os.getenv("LLM_API_TYPE", "ollama"))  # "ollama" or "openai_compatible"

    # Reporting/output
    show_config: bool = field(default_factory=lambda: env_bool("SHOW_CONFIG", True))

    # Embeddings / Sentence Transformers
    embedding_models: list[str] = field(default_factory=default_embedding_models)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ExtractionConfig:
        """Construct config with precedence: env vars > YAML > code defaults."""
        resolved_path = Path(path).expanduser() if path is not None else find_config_file()
        yaml_config = load_yaml_config(resolved_path)
        field_types = get_type_hints(cls)

        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(cls):
            field_name = dataclass_field.name
            if has_env_override(field_name):
                continue
            if field_name not in yaml_config:
                continue

            annotation = field_types.get(field_name, dataclass_field.type)
            try:
                kwargs[field_name] = yaml_coerce_value(field_name, yaml_config[field_name], annotation)
            except (TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Ignoring YAML value for '%s' in %s: %s",
                    field_name,
                    resolved_path,
                    exc,
                )

        config = cls(**kwargs)
        config._loaded_config_path = resolved_path
        LOGGER.info("Loaded config from: %s", resolved_path or "(defaults only)")
        return config

    def __post_init__(self) -> None:
        self.ontology_path.mkdir(parents=True, exist_ok=True)
        if not self.ollama_host.startswith(("http://", "https://")):
            self.ollama_host = f"http://{self.ollama_host}"
        if not self.nim_host.startswith(("http://", "https://")):
            self.nim_host = f"http://{self.nim_host}"
        if not self.embedding_models:
            self.embedding_models = list(DEFAULT_EMBEDDING_MODELS)

    @property
    def active_llm_model(self) -> str:
        """Return the model for the currently configured API type."""
        if self.llm_api_type == "openai_compatible":
            return self.nim_models[0] if self.nim_models else "meta/llama-3.1-8b-instruct"
        return self.ollama_models[0] if self.ollama_models else self.ollama_model

    @property
    def active_llm_models(self) -> list[str]:
        """Return the model list for the currently configured API type."""
        if self.llm_api_type == "openai_compatible":
            return self.nim_models
        return self.ollama_models

    @property
    def active_llm_host(self) -> str:
        """Return the host for the currently configured API type."""
        if self.llm_api_type == "openai_compatible":
            return self.nim_host
        return self.ollama_host

    @property
    def embedding_model(self) -> str:
        """Backward-compatible single-model accessor."""
        return self.embedding_models[0]

    def log_config(self, logger: logging.Logger | None = None) -> None:
        """Log all configuration options in a nicely formatted table."""
        logger = logger or LOGGER
        loaded_path = getattr(self, "_loaded_config_path", None)
        llm_provider_prefix = "nim_" if self.llm_api_type == "openai_compatible" else "ollama_"
        hidden_provider_prefix = "ollama_" if self.llm_api_type == "openai_compatible" else "nim_"
        visible_field_names = [
            config_field.name
            for config_field in fields(self)
            if not config_field.name.startswith(hidden_provider_prefix)
        ]

        max_name = max(len(name) for name in visible_field_names)
        header = f"{'Option':<{max_name}}  Value"
        separator = "-" * len(header)
        logger.info("Configuration:")
        logger.info("Loaded config from: %s", loaded_path or "(defaults only)")
        logger.info(separator)
        logger.info(header)
        logger.info(separator)
        for field_name in visible_field_names:
            value = getattr(self, field_name)
            logger.info("%-*s  %s", max_name, field_name, value)
        logger.info(separator)
        logger.info(
            "LLM provider-specific options shown for active provider: %s",
            llm_provider_prefix.rstrip("_"),
        )
        logger.info(separator)
        logger.info("Active LLM settings:")
        logger.info("%-*s  %s", max_name, "active_llm_model", self.active_llm_model)
        logger.info("%-*s  %s", max_name, "active_llm_models", self.active_llm_models)
        logger.info("%-*s  %s", max_name, "active_llm_host", self.active_llm_host)
        logger.info(separator)

    @active_llm_host.setter
    def active_llm_host(self, value: str) -> None:
        """Set the host for the currently configured API type."""
        if self.llm_api_type == "openai_compatible":
            self.nim_host = value
        else:
            self.ollama_host = value
