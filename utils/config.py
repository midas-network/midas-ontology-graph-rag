from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class ExtractionConfig:
    """Application configuration with env var overrides."""

    ontology_path: Path = field(default_factory=lambda: Path(os.getenv("ONTOLOGY_PATH", "data/ontologies")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "data/output/runs")))
    papers_path: Path = field(default_factory=lambda: Path(os.getenv("PAPERS_PATH", "data/modeling_papers.json")))
    abstract_path: Path = field(default_factory=lambda: Path(os.getenv("ABSTRACT_PATH", "data/fred-abstract.txt")))

    enable_ontology_linking: bool = field(default_factory=lambda: _env_bool("ENABLE_ONTOLOGY_LINKING", False))
    enable_midas_ontology: bool = field(default_factory=lambda: _env_bool("ENABLE_MIDAS_ONTOLOGY", False))
    include_format_examples: bool = field(default_factory=lambda: _env_bool("INCLUDE_FORMAT_EXAMPLES", False))
    include_reminders: bool = field(default_factory=lambda: _env_bool("INCLUDE_REMINDERS", False))
    include_few_shot: bool = field(default_factory=lambda: _env_bool("INCLUDE_FEW_SHOT", False))
    include_fields: bool = field(default_factory=lambda: _env_bool("INCLUDE_FIELDS", True))
    include_ontologies: bool = field(default_factory=lambda: _env_bool("INCLUDE_ONTOLOGIES", False))
    use_examples: bool = field(default_factory=lambda: _env_bool("USE_EXAMPLES", False))
    run_sanity_check: bool = field(default_factory=lambda: _env_bool("RUN_SANITY_CHECK", True))
    test_mode: bool = field(default_factory=lambda: _env_bool("TEST_MODE", False))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))

    num_examples: int = field(default_factory=lambda: _env_int("NUM_EXAMPLES", 2))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 300))

    llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:72b"))
    llm_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ontology_path.mkdir(parents=True, exist_ok=True)
        # normalize host
        if not self.llm_host.startswith(("http://", "https://")):
            self.llm_host = f"http://{self.llm_host}"
