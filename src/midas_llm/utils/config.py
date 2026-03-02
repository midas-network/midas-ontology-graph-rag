from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

LOGGER = logging.getLogger("midas-llm")


def _load_ollama_cfg() -> None:
    """Load ollama.cfg if present, setting env vars for OLLAMA_MODEL/OLLAMA_HOST."""
    cfg_paths = [
        Path("ollama.cfg"),  # current dir
        Path(__file__).resolve().parents[3] / "ollama.cfg",  # repo root
        Path.home() / ".ollama.cfg",  # home dir
    ]
    for cfg in cfg_paths:
        if cfg.is_file():
            with open(cfg, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip()
                        if key and key not in os.environ:
                            os.environ[key] = value
            break


_load_ollama_cfg()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _parse_model_list(value: str) -> list[str]:
    """Parse comma-separated model list, filtering empty strings."""
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


@dataclass
class ExtractionConfig:
    """Application configuration with env var overrides."""

    # Paths and I/O
    ontology_path: Path = field(default_factory=lambda: Path(os.getenv("ONTOLOGY_PATH", "resources/ontologies")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output/extract_concepts/runs")))
    papers_path: Path = field(default_factory=lambda: Path(os.getenv("PAPERS_PATH", "resources/modeling_papers.json")))
    abstract_path: Path = field(default_factory=lambda: Path(os.getenv("ABSTRACT_PATH", "resources/fred-abstract.txt")))

    # Feature toggles
    enable_ontology_linking: bool = field(default_factory=lambda: _env_bool("ENABLE_ONTOLOGY_LINKING", False))
    enable_midas_ontology: bool = field(default_factory=lambda: _env_bool("ENABLE_MIDAS_ONTOLOGY", False))
    run_sanity_check: bool = field(default_factory=lambda: _env_bool("RUN_SANITY_CHECK", True))
    test_mode: bool = field(default_factory=lambda: _env_bool("TEST_MODE", False))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))

    # Prompt options
    prompt_include_format_examples: bool = field(default_factory=lambda: _env_bool("INCLUDE_FORMAT_EXAMPLES", False))
    prompt_include_reminders: bool = field(default_factory=lambda: _env_bool("INCLUDE_REMINDERS", True))
    prompt_include_few_shot: bool = field(default_factory=lambda: _env_bool("INCLUDE_FEW_SHOT", True))
    prompt_include_fields: bool = field(default_factory=lambda: _env_bool("INCLUDE_FIELDS", True))
    prompt_include_ontologies: bool = field(default_factory=lambda: _env_bool("INCLUDE_ONTOLOGIES", False))
    prompt_simple_prompt: bool = field(default_factory=lambda: _env_bool("SIMPLE_PROMPT", False))
    num_examples: int = field(default_factory=lambda: _env_int("NUM_EXAMPLES", 2))

    # LLM settings - Ollama
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral:7b"))
    ollama_models: list[str] = field(default_factory=lambda: _parse_model_list(os.getenv("OLLAMA_MODELS", "")))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # LLM settings - NIM / OpenAI-compatible
    nim_models: list[str] = field(default_factory=lambda: _parse_model_list(
        #os.getenv("NIM_MODELS", "nvidia/llama-3.3-nemotron-super-49b-v1")
        #os.getenv("NIM_MODELS", "meta/llama-3.1-8b-instruct")
        os.getenv("NIM_MODELS", "deepseek-ai/deepseek-r1-distill-llama-70b")
    ))
    nim_host: str = field(default_factory=lambda: os.getenv("NIM_HOST", "http://localhost:8000"))

    # LLM settings - shared
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 300000))
    llm_api_type: str = field(default_factory=lambda: os.getenv("LLM_API_TYPE", "openai_compatible"))  # "ollama" or "openai_compatible"

    # Reporting/output
    show_config: bool = field(default_factory=lambda: _env_bool("SHOW_CONFIG", True))
    generate_html_report: bool = field(default_factory=lambda: _env_bool("GENERATE_HTML_REPORT", False))
    generate_json_output: bool = field(default_factory=lambda: _env_bool("GENERATE_JSON_OUTPUT", False))
    generate_evaluation_report: bool = field(default_factory=lambda: _env_bool("GENERATE_EVALUATION_REPORT", True))

    # Embeddings / Sentence Transformers
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL","FremyCompany/BioLORD-2023-C"
    ))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ontology_path.mkdir(parents=True, exist_ok=True)
        # normalize hosts
        if not self.ollama_host.startswith(("http://", "https://")):
            self.ollama_host = f"http://{self.ollama_host}"
        if not self.nim_host.startswith(("http://", "https://")):
            self.nim_host = f"http://{self.nim_host}"

    @property
    def active_llm_model(self) -> str:
        """Return the model for the currently configured API type.

        For openai_compatible (NIM), returns the first model from nim_models.
        For ollama, returns ollama_model (or first from ollama_models if set).
        """
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

    def log_config(self, logger: logging.Logger | None = None) -> None:
        """Log all configuration options in a nicely formatted table."""
        logger = logger or LOGGER
        max_name = max(len(f.name) for f in fields(self))
        header = f"{'Option':<{max_name}}  Value"
        sep = "-" * len(header)
        logger.info("Configuration:")
        logger.info(sep)
        logger.info(header)
        logger.info(sep)
        for f in fields(self):
            value = getattr(self, f.name)
            logger.info(f"{f.name:<{max_name}}  {value}")
        # Also log computed properties
        logger.info(sep)
        logger.info("Active LLM settings:")
        logger.info(f"{'active_llm_model':<{max_name}}  {self.active_llm_model}")
        logger.info(f"{'active_llm_models':<{max_name}}  {self.active_llm_models}")
        logger.info(f"{'active_llm_host':<{max_name}}  {self.active_llm_host}")
        logger.info(sep)

    @active_llm_host.setter
    def active_llm_host(self, value: str) -> None:
        """Set the host for the currently configured API type."""
        if self.llm_api_type == "openai_compatible":
            self.nim_host = value
        else:
            self.ollama_host = value