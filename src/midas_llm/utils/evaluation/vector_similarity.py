"""Vector similarity evaluation using sentence embeddings.

This module provides fast semantic similarity matching using HuggingFace
sentence-transformers. It serves as a middle tier between fast string matching
and slow LLM-based semantic evaluation.

The module handles the case where sentence-transformers is not installed
gracefully, returning None so callers can fall back to other methods.
"""
from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("midas-llm")

# Lazy-loaded singleton for the embedding model
_embedding_model = None
_embedding_model_name: str | None = None
_sentence_transformers_available: bool | None = None


def _check_sentence_transformers_available() -> bool:
    """Check if sentence-transformers is installed."""
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            import sentence_transformers  # noqa: F401
            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
            LOGGER.warning(
                "sentence-transformers not installed. Vector similarity evaluation disabled. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers_available


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load and cache the sentence embedding model.

    Args:
        model_name: HuggingFace model name for sentence embeddings.
                   Defaults to 'all-MiniLM-L6-v2' (fast, good for short text).

    Returns:
        SentenceTransformer model instance, or None if not available.
    """
    global _embedding_model, _embedding_model_name

    if not _check_sentence_transformers_available():
        return None

    # Load model if not cached or if model name changed
    if _embedding_model is None or _embedding_model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            LOGGER.info("Loading sentence embedding model: %s", model_name)
            _embedding_model = SentenceTransformer(model_name)
            _embedding_model_name = model_name
            LOGGER.info("Sentence embedding model loaded successfully")
        except Exception as e:
            LOGGER.error("Failed to load embedding model '%s': %s", model_name, e)
            return None

    return _embedding_model


def compute_similarity(
    text_a: str,
    text_b: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> float | None:
    """Compute cosine similarity between two strings using embeddings.

    Args:
        text_a: First text string.
        text_b: Second text string.
        model_name: HuggingFace model name for embeddings.

    Returns:
        Cosine similarity score in range [0, 1], or None if embeddings unavailable.
    """
    model = _get_embedding_model(model_name)
    if model is None:
        return None

    try:
        # Encode both texts
        embeddings = model.encode([text_a, text_b], convert_to_tensor=True)

        # Compute cosine similarity
        from sentence_transformers import util
        similarity = util.cos_sim(embeddings[0], embeddings[1])

        # Convert to Python float, ensure [0, 1] range
        score = float(similarity.item())
        # Cosine similarity can be negative for very different texts; clamp to [0, 1]
        return max(0.0, min(1.0, score))
    except Exception as e:
        LOGGER.warning("Error computing similarity: %s", e)
        return None


def compute_similarity_batch(
    text: str,
    candidates: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float] | None:
    """Compute cosine similarity between a text and multiple candidates.

    More efficient than calling compute_similarity repeatedly because
    it encodes all texts in a single batch.

    Args:
        text: The query text.
        candidates: List of candidate texts to compare against.
        model_name: HuggingFace model name for embeddings.

    Returns:
        List of similarity scores, one per candidate, or None if unavailable.
    """
    if not candidates:
        return []

    model = _get_embedding_model(model_name)
    if model is None:
        return None

    try:
        # Encode query and all candidates in one batch
        all_texts = [text] + candidates
        embeddings = model.encode(all_texts, convert_to_tensor=True)

        # Compute similarities between query (index 0) and each candidate
        from sentence_transformers import util
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = util.cos_sim(query_embedding, candidate_embeddings)

        # Convert to Python floats, clamped to [0, 1]
        scores = [max(0.0, min(1.0, float(s))) for s in similarities[0]]
        return scores
    except Exception as e:
        LOGGER.warning("Error computing batch similarity: %s", e)
        return None


def vector_match(
    extracted_value: str,
    expected_values: list[str],
    threshold: float = 0.75,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[bool, str | None, float]:
    """Compare extracted value against expected values using embeddings.

    Computes similarity of the extracted value against every expected value
    and returns the best match if it exceeds the threshold.

    Args:
        extracted_value: The value extracted by the LLM.
        expected_values: List of expected values from gold standard.
        threshold: Minimum similarity score to count as a match.
        model_name: HuggingFace model name for embeddings.

    Returns:
        Tuple of (is_match, best_matching_expected_value_or_None, similarity_score).
        If embeddings are unavailable, returns (False, None, 0.0).
    """
    if not extracted_value or not expected_values:
        return (False, None, 0.0)

    # Compute similarities in batch for efficiency
    similarities = compute_similarity_batch(extracted_value, expected_values, model_name)

    if similarities is None:
        return (False, None, 0.0)

    # Find best match
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    best_score = similarities[best_idx]
    best_match = expected_values[best_idx]

    is_match = best_score >= threshold

    return (is_match, best_match if is_match else None, best_score)


def vector_match_tiered(
    extracted_value: str,
    expected_values: list[str],
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[str, str | None, float]:
    """Compare extracted value with tiered thresholds for three-tier evaluation.

    This function implements the middle tier of a three-tier evaluation strategy:
    1. If similarity >= high_threshold: definite MATCH (skip LLM eval)
    2. If similarity <= low_threshold: definite NO_MATCH (skip LLM eval)
    3. Otherwise: AMBIGUOUS (should fall through to LLM eval)

    Args:
        extracted_value: The value extracted by the LLM.
        expected_values: List of expected values from gold standard.
        high_threshold: Auto-match above this threshold.
        low_threshold: Auto-reject below this threshold.
        model_name: HuggingFace model name for embeddings.

    Returns:
        Tuple of (decision, best_match_or_None, similarity_score).
        decision is one of: "MATCH", "NO_MATCH", "AMBIGUOUS", "UNAVAILABLE".
    """
    if not extracted_value or not expected_values:
        return ("NO_MATCH", None, 0.0)

    # Compute similarities in batch
    similarities = compute_similarity_batch(extracted_value, expected_values, model_name)

    if similarities is None:
        return ("UNAVAILABLE", None, 0.0)

    # Find best match
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    best_score = similarities[best_idx]
    best_match = expected_values[best_idx]

    if best_score >= high_threshold:
        return ("MATCH", best_match, best_score)
    elif best_score <= low_threshold:
        return ("NO_MATCH", None, best_score)
    else:
        return ("AMBIGUOUS", best_match, best_score)


def evaluate_threshold_quality(
    test_pairs: list[tuple[str, str, bool]],
    threshold: float,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Evaluate precision/recall/F1 at a given threshold using labeled pairs.

    This is a utility function for tuning threshold values. Given labeled pairs
    of texts and their expected match status, it computes how well the vector
    similarity performs at a specific threshold.

    Args:
        test_pairs: List of (text_a, text_b, expected_match) tuples.
                   expected_match is True if the texts should match.
        threshold: Similarity threshold to evaluate.
        model_name: HuggingFace model name for embeddings.

    Returns:
        Dict with keys: precision, recall, f1, accuracy, total_pairs,
                       true_positives, false_positives, true_negatives, false_negatives.
    """
    if not test_pairs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "total_pairs": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for text_a, text_b, expected_match in test_pairs:
        similarity = compute_similarity(text_a, text_b, model_name)

        if similarity is None:
            # Skip pairs where similarity couldn't be computed
            continue

        predicted_match = similarity >= threshold

        if expected_match and predicted_match:
            true_positives += 1
        elif expected_match and not predicted_match:
            false_negatives += 1
        elif not expected_match and predicted_match:
            false_positives += 1
        else:
            true_negatives += 1

    total = true_positives + false_positives + true_negatives + false_negatives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total_pairs": total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "threshold": threshold,
    }


def find_optimal_threshold(
    test_pairs: list[tuple[str, str, bool]],
    thresholds: list[float] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    optimize_for: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Find the optimal threshold for a set of labeled pairs.

    Tests multiple thresholds and returns the one that maximizes the
    specified metric.

    Args:
        test_pairs: List of (text_a, text_b, expected_match) tuples.
        thresholds: List of thresholds to test. Defaults to [0.5, 0.55, ..., 0.95].
        model_name: HuggingFace model name for embeddings.
        optimize_for: Metric to optimize ("f1", "precision", "recall", "accuracy").

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal).
    """
    if thresholds is None:
        thresholds = [0.50 + i * 0.05 for i in range(10)]  # 0.50 to 0.95

    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] = {}
    best_score = -1.0

    for threshold in thresholds:
        metrics = evaluate_threshold_quality(test_pairs, threshold, model_name)
        score = metrics.get(optimize_for, 0.0)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


# Domain-specific test pairs for infectious disease terminology
INFECTIOUS_DISEASE_TEST_PAIRS: list[tuple[str, str, bool]] = [
    # Should match (synonyms, abbreviations, related terms)
    ("Influenza", "Flu", True),
    ("H1N1", "Influenza A", True),
    ("SARS-CoV-2", "COVID-19 virus", True),
    ("agent-based model", "ABM", True),  # Note: embeddings may struggle with abbreviations
    ("children", "pediatric population", True),
    ("USA", "United States", True),
    ("human", "Homo sapiens", True),
    ("mosquito", "Aedes aegypti", True),
    ("epidemic", "outbreak", True),
    ("transmission", "spread", True),
    ("mortality", "death rate", True),
    ("incidence", "case rate", True),
    ("prevalence", "infection rate", True),
    ("vaccine", "vaccination", True),
    ("hospital", "healthcare facility", True),

    # Should NOT match (different concepts)
    ("malaria", "dengue", False),
    ("stochastic", "deterministic", False),
    ("bacteria", "virus", False),
    ("children", "elderly", False),
    ("rural", "urban", False),
    ("prevention", "treatment", False),
    ("incidence", "prevalence", False),
    ("acute", "chronic", False),
    ("symptomatic", "asymptomatic", False),
    ("endemic", "pandemic", False),
]


def run_domain_validation(
    model_name: str = "all-MiniLM-L6-v2",
    logger: logging.Logger = LOGGER,
) -> dict[str, Any]:
    """Run validation using domain-specific test pairs.

    This function tests the embedding model against known infectious disease
    terminology pairs to help tune thresholds.

    Args:
        model_name: HuggingFace model name for embeddings.
        logger: Logger instance.

    Returns:
        Dict with validation results including optimal thresholds.
    """
    if not _check_sentence_transformers_available():
        return {"error": "sentence-transformers not available"}

    logger.info("Running domain validation for vector similarity...")

    # Find optimal threshold
    optimal_threshold, optimal_metrics = find_optimal_threshold(
        INFECTIOUS_DISEASE_TEST_PAIRS,
        model_name=model_name,
    )

    logger.info("Optimal threshold: %.2f (F1=%.2f, P=%.2f, R=%.2f)",
                optimal_threshold,
                optimal_metrics["f1"],
                optimal_metrics["precision"],
                optimal_metrics["recall"])

    # Test at multiple thresholds
    threshold_results = {}
    for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        metrics = evaluate_threshold_quality(
            INFECTIOUS_DISEASE_TEST_PAIRS,
            threshold,
            model_name,
        )
        threshold_results[f"threshold_{threshold:.2f}"] = metrics

    # Show individual pair results at optimal threshold
    pair_results = []
    for text_a, text_b, expected in INFECTIOUS_DISEASE_TEST_PAIRS:
        similarity = compute_similarity(text_a, text_b, model_name)
        predicted = similarity >= optimal_threshold if similarity else False
        correct = predicted == expected
        pair_results.append({
            "text_a": text_a,
            "text_b": text_b,
            "expected_match": expected,
            "similarity": similarity,
            "predicted_match": predicted,
            "correct": correct,
        })

    return {
        "model": model_name,
        "optimal_threshold": optimal_threshold,
        "optimal_metrics": optimal_metrics,
        "threshold_results": threshold_results,
        "pair_results": pair_results,
    }
