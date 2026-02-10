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
                        # Only set if not already in env (env takes precedence)
                        if key and key not in os.environ:
                            os.environ[key] = value
            break  # stop after first found


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

    # LLM settings
    llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral:7b"))
    llm_models: list[str] = field(default_factory=lambda: _parse_model_list(os.getenv("OLLAMA_MODELS", "")))
    llm_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 300000))

    # Reporting/output
    show_config: bool = field(default_factory=lambda: _env_bool("SHOW_CONFIG", True))
    generate_html_report: bool = field(default_factory=lambda: _env_bool("GENERATE_HTML_REPORT", False))
    generate_json_output: bool = field(default_factory=lambda: _env_bool("GENERATE_JSON_OUTPUT", False))
    generate_evaluation_report: bool = field(default_factory=lambda: _env_bool("GENERATE_EVALUATION_REPORT", True))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ontology_path.mkdir(parents=True, exist_ok=True)
        # normalize host
        if not self.llm_host.startswith(("http://", "https://")):
            self.llm_host = f"http://{self.llm_host}"

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
        logger.info(sep)

