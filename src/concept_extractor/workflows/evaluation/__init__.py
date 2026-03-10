"""evaluation package — split from run_evaluation.py."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path for concept_extractor imports
_src_path = Path(__file__).resolve().parent.parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from .parsing import (
    load_evaluation_dataset,
    parse_constrained_response,
    validate_constrained_payload,
    get_expected_values,
    JSON_OUTPUT_SUFFIX,
    REPO_ROOT,
    DEFAULT_EVALUATION_DATASET_PATH,
    LEGACY_EVALUATION_DATASET_PATH,
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_GOLD_OUTPUT_DIR,
    DEFAULT_VALIDATION_SCHEMA_PATH,
)
from .matching import (
    evaluate_semantic_match,
    fallback_string_match,
    vector_match_any_model,
)
from .engine import (
    evaluate_extraction,
    run_evaluation,
)
from .reporting import (
    print_summary,
    sanitize_model_directory_name,
    results_model_directory_name,
    parse_embedding_models_arg,
    log_active_run_configuration,
)
