"""Evaluation utilities for midas-llm."""
from .vector_similarity import (
    compute_similarity,
    compute_similarity_batch,
    vector_match,
    vector_match_tiered,
    evaluate_threshold_quality,
    find_optimal_threshold,
    run_domain_validation,
    INFECTIOUS_DISEASE_TEST_PAIRS,
)

__all__ = [
    "compute_similarity",
    "compute_similarity_batch",
    "vector_match",
    "vector_match_tiered",
    "evaluate_threshold_quality",
    "find_optimal_threshold",
    "run_domain_validation",
    "INFECTIOUS_DISEASE_TEST_PAIRS",
]
