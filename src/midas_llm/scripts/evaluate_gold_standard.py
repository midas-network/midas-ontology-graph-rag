#!/usr/bin/env python
"""Gold standard evaluation script for LLM extraction accuracy.

Compares extracted attributes against expected values using LLM-based semantic matching.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.config import ExtractionConfig
from midas_llm.utils.logging.logger import configure_logging
from midas_llm.utils.llm.llm_client import send_to_llm
from midas_llm.utils.llm.llm_utils import autodetect_llm_host
from midas_llm.utils.prompt.builders import prepare_and_display_prompt, build_query
from midas_llm.utils.parsers.extraction_parser import parse_and_display_extracted_data
from midas_llm.utils.reporting.evaluation_reports import generate_evaluation_text_report, \
    generate_evaluation_html_report
from midas_llm.utils.evaluation.vector_similarity import vector_match_tiered

LOGGER = logging.getLogger("midas-llm")

# Repo root for resolving paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_gold_standard(path: str | None = None) -> list[dict]:
    """Load gold standard test abstracts."""
    if path is None:
        path = str(REPO_ROOT / "resources" / "test_abstracts.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("abstracts", [])


def build_semantic_eval_prompt(
        attribute: str,
        extracted_value: str,
        expected_values: list[str],
) -> str:
    """Build prompt for LLM to evaluate semantic match."""
    expected_str = ", ".join(f'"{v}"' for v in expected_values)

    return f"""You are evaluating whether an extracted value semantically matches any of the expected values for a scientific metadata field.

Field: {attribute}
Extracted value: "{extracted_value}"
Expected values (any match is acceptable): [{expected_str}]

Does the extracted value mean the same thing as any of the expected values? Consider synonyms, abbreviations, and related terms.

Examples:
- "influenza" matches "flu" or "Influenza A" (same disease)
- "agent-based" matches "ABM" or "individual-based model" (same model type)  
- "USA" matches "United States" or "US" (same country)
- "SARS-CoV-2" matches "COVID-19 virus" (same pathogen)
- "children" matches "pediatric population" or "school-aged children" (similar population)
- "May 2022" matches "2022-05" (same date, different format)
- "December 2023" matches "2023-12" (same date, different format)

Respond with ONLY one word: MATCH or NO_MATCH"""


def evaluate_semantic_match(
        attribute: str,
        extracted_value: str,
        expected_values: list[str],
        llm_model: str,
        llm_host: str,
        logger: logging.Logger,
        timeout_seconds: int = 30,
            api_type: str = "ollama",
) -> tuple[bool, str | None]:
    """Use LLM to evaluate if extracted value semantically matches expected."""

    prompt = build_semantic_eval_prompt(attribute, extracted_value, expected_values)

    try:
        response = send_to_llm(
            prompt=prompt,
            llm_model=llm_model,
            llm_host=llm_host,
            timeout_seconds=timeout_seconds,
            logger=logger,
            api_type=api_type
        )

        result = response.content.strip().upper()
        is_match = "MATCH" in result and "NO_MATCH" not in result

        if is_match:
            # Find which expected value it likely matched
            for exp in expected_values:
                if exp.lower() in extracted_value.lower() or extracted_value.lower() in exp.lower():
                    return True, exp
            return True, expected_values[0]  # Default to first expected

        return False, None

    except Exception as e:
        logger.warning("Semantic eval failed, falling back to string match: %s", e)
        # Fallback to simple string matching
        return fallback_string_match(extracted_value, expected_values)


def fallback_string_match(extracted_value: str, expected_values: list[str]) -> tuple[bool, str | None]:
    """Fallback string matching when LLM eval fails."""
    extracted_norm = extracted_value.lower().strip().replace("-", " ").replace("_", " ")

    for expected in expected_values:
        expected_norm = expected.lower().strip().replace("-", " ").replace("_", " ")
        if extracted_norm == expected_norm:
            return True, expected
        if expected_norm in extracted_norm or extracted_norm in expected_norm:
            return True, expected
        # Check for date matches (e.g., "May 2022" vs "2022-05")
        if dates_match(extracted_value, expected):
            return True, expected

    return False, None


def dates_match(extracted: str, expected: str) -> bool:
    """Check if two date strings represent the same date."""
    month_names = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12',
    }

    def normalize_date(date_str: str) -> str:
        """Normalize date to YYYY-MM format."""
        date_str = date_str.lower().strip().replace('**', '').strip()

        # Try "Month YYYY" format (e.g., "May 2022")
        for month_name, month_num in month_names.items():
            if month_name in date_str:
                # Find year (4 digits)
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return f"{year_match.group(1)}-{month_num}"

        # Try YYYY-MM format already
        ym_match = re.match(r'^(\d{4})-(\d{1,2})(?:-\d{1,2})?$', date_str)
        if ym_match:
            return f"{ym_match.group(1)}-{ym_match.group(2).zfill(2)}"

        return date_str

    return normalize_date(extracted) == normalize_date(expected)


def evaluate_extraction(
        extracted: dict[str, Any],
        expected: dict[str, list[str]],
        llm_model: str,
        llm_host: str,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Evaluate extraction against expected values using three-tier matching.

    Evaluation tiers (in order):
    1. Vector similarity (fast, no LLM call)
       - similarity >= high_threshold: auto-MATCH
       - similarity <= low_threshold: auto-NO_MATCH
       - otherwise: fall through to tier 2
    2. LLM semantic evaluation (if enabled and vector was ambiguous)
    3. Fallback string matching (if LLM unavailable or disabled)

    Returns evaluation results with hits, misses, and false positives.
    """
    results = {
        "hits": [],
        "misses": [],
        "false_positives": [],
        "not_expected": [],
        "scores": {},
        "vector_stats": {
            "auto_matches": 0,
            "auto_rejects": 0,
            "ambiguous_to_llm": 0,
            "vector_unavailable": 0,
        },
    }

    # Track which expected values were matched
    matched_expected = {attr: set() for attr in expected}

    # Check each extracted attribute
    for attr, data in extracted.items():
        value = data.get("value", "") if isinstance(data, dict) else str(data)

        if attr not in expected:
            results["not_expected"].append({
                "attribute": attr,
                "extracted_value": value,
            })
            continue

        expected_values = expected[attr]
        if not expected_values:  # Skip empty expected lists
            continue

        # Handle comma-separated values
        extracted_values = [v.strip() for v in value.split(",")]

        for ext_val in extracted_values:
            if not ext_val:
                continue

            matched = False
            matched_val = None
            similarity_score = None
            match_method = None

            # Special handling for date fields - skip vector eval
            is_date_field = attr in ("study_dates_start", "study_dates_end")

            # Tier 1: Vector similarity (if enabled and not a date field)
            if use_vector_eval and not is_date_field:
                decision, best_match, similarity_score = vector_match_tiered(
                    ext_val,
                    expected_values,
                    high_threshold=vector_high_threshold,
                    low_threshold=vector_low_threshold,
                    model_name=embedding_model,
                )

                if decision == "MATCH":
                    matched = True
                    matched_val = best_match
                    match_method = "vector"
                    results["vector_stats"]["auto_matches"] += 1
                    logger.debug("Vector auto-match: '%s' ↔ '%s' (score=%.3f)",
                                 ext_val, matched_val, similarity_score)
                elif decision == "NO_MATCH":
                    # Vector says definite no-match, but still try dates_match for date-like values
                    if dates_match(ext_val, expected_values[0] if expected_values else ""):
                        matched = True
                        matched_val = expected_values[0]
                        match_method = "date_format"
                    else:
                        results["vector_stats"]["auto_rejects"] += 1
                        logger.debug("Vector auto-reject: '%s' (score=%.3f)", ext_val, similarity_score)
                elif decision == "AMBIGUOUS":
                    results["vector_stats"]["ambiguous_to_llm"] += 1
                    logger.debug("Vector ambiguous: '%s' (score=%.3f), falling through to LLM",
                                 ext_val, similarity_score)
                else:  # UNAVAILABLE
                    results["vector_stats"]["vector_unavailable"] += 1

            # Tier 2: LLM semantic evaluation (for ambiguous vector scores or if vector disabled)
            if not matched and matched_val is None and use_llm_eval:
                # Only call LLM if vector was ambiguous or unavailable
                if not use_vector_eval or similarity_score is None or (
                        vector_low_threshold < (similarity_score or 0) < vector_high_threshold
                ):
                    try:
                        matched, matched_val = evaluate_semantic_match(
                            attr, ext_val, expected_values, llm_model, llm_host,
                            api_type=config.llm_api_type,
                            timeout_seconds=timeout_seconds,
                        )
                        if matched:
                            match_method = "llm"
                    except Exception as e:
                        logger.warning("LLM eval failed for '%s': %s", ext_val, e)

            # Tier 3: Fallback string matching
            if not matched and matched_val is None:
                matched, matched_val = fallback_string_match(ext_val, expected_values)
                if matched:
                    match_method = "string"

            if matched:
                hit_record = {
                    "attribute": attr,
                    "extracted_value": ext_val,
                    "matched_expected": matched_val,
                    "match_method": match_method,
                }
                if similarity_score is not None:
                    hit_record["similarity_score"] = round(similarity_score, 4)
                results["hits"].append(hit_record)
                if matched_val:
                    matched_expected[attr].add(matched_val)
            else:
                fp_record = {
                    "attribute": attr,
                    "extracted_value": ext_val,
                    "expected_values": expected_values,
                }
                if similarity_score is not None:
                    fp_record["similarity_score"] = round(similarity_score, 4)
                results["false_positives"].append(fp_record)

    # Check for misses (expected values not matched)
    for attr, expected_values in expected.items():
        for exp_val in expected_values:
            if exp_val not in matched_expected.get(attr, set()):
                results["misses"].append({
                    "attribute": attr,
                    "expected_value": exp_val,
                })

    # Calculate scores
    total_expected = sum(len(v) for v in expected.values())
    total_hits = len(results["hits"])
    total_misses = len(results["misses"])
    total_false_positives = len(results["false_positives"])

    results["scores"] = {
        "total_expected": total_expected,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_false_positives": total_false_positives,
        "recall": total_hits / total_expected if total_expected > 0 else 0,
        "precision": total_hits / (total_hits + total_false_positives) if (
                                                                                      total_hits + total_false_positives) > 0 else 0,
    }

    if results["scores"]["precision"] + results["scores"]["recall"] > 0:
        results["scores"]["f1"] = 2 * (results["scores"]["precision"] * results["scores"]["recall"]) / (
                    results["scores"]["precision"] + results["scores"]["recall"])
    else:
        results["scores"]["f1"] = 0

    return results


def run_evaluation(
        abstracts: list[dict],
        config: ExtractionConfig,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_model: str = "all-MiniLM-L6-v2",
        run_folder: str | None = None,
) -> dict[str, Any]:
    """Run evaluation on gold standard abstracts.

    Uses a three-tier evaluation strategy:
    1. Vector similarity (fast) - auto-match/reject at thresholds
    2. LLM semantic evaluation - for ambiguous vector scores
    3. String matching - fallback
    """

    all_results = []

    # Determine models to run
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_models]

    for abstract_data in abstracts:
        abstract_id = abstract_data["id"]
        title = abstract_data["title"]
        abstract = abstract_data["abstract"]
        expected = abstract_data["expected"]

        logger.info("=" * 60)
        logger.info("Evaluating: %s", title[:60])
        logger.info("=" * 60)

        # Build prompt using same approach as extract_concepts.py
        query = build_query(
            abstract,
            include_examples=config.prompt_include_format_examples,
            include_reminders=config.prompt_include_reminders,
            include_few_shot=config.prompt_include_few_shot,
            include_fields=config.prompt_include_fields,
            include_ontologies=config.prompt_include_ontologies,
            simple_prompt=config.prompt_simple_prompt,
        )
        prompt = prepare_and_display_prompt(query, "", logger=logger)

        # Save prompt to file in run folder if provided
        if run_folder:
            prompt_file = Path(run_folder) / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            logger.info("Saved prompt to: %s", prompt_file)

        abstract_results = {
            "id": abstract_id,
            "title": title,
            "models": {},
        }

        for model in models:
            logger.info("Running model: %s", model)

            try:
                response = send_to_llm(
                    prompt=prompt,
                    llm_model=model,
                    llm_host=config.active_llm_host,
                    timeout_seconds=config.llm_timeout,
                    logger=logger,
                    api_type=config.llm_api_type,
                )

                extracted = parse_and_display_extracted_data(response.content, logger=logger)

                logger.info("Evaluating extraction with semantic matching...")
                evaluation = evaluate_extraction(
                    extracted,
                    expected,
                    model,
                    config.active_llm_host,
                    logger,
                    use_llm_eval=use_llm_eval,
                    use_vector_eval=use_vector_eval,
                    vector_high_threshold=vector_high_threshold,
                    vector_low_threshold=vector_low_threshold,
                    embedding_model=embedding_model,
                )

                abstract_results["models"][model] = {
                    "extracted": extracted,
                    "evaluation": evaluation,
                    "raw_response": response.content[:2000],
                }

                # Log legend once
                logger.info("")
                logger.info("EVALUATION LEGEND:")
                logger.info("  HIT = LLM extracted a value that matches something in the gold standard")
                logger.info("  MISS = Gold standard expects a value, but LLM didn't extract it")
                logger.info("  FALSE POSITIVE = LLM extracted a value not in the gold standard")
                logger.info("")

                # Log vector evaluation stats if enabled
                vector_stats = evaluation.get("vector_stats", {})
                if use_vector_eval and any(vector_stats.values()):
                    logger.info("VECTOR EVALUATION STATS:")
                    logger.info("  Auto-matches (score >= %.2f): %d", vector_high_threshold,
                                vector_stats.get("auto_matches", 0))
                    logger.info("  Auto-rejects (score <= %.2f): %d", vector_low_threshold,
                                vector_stats.get("auto_rejects", 0))
                    logger.info("  Ambiguous → LLM: %d", vector_stats.get("ambiguous_to_llm", 0))
                    if vector_stats.get("vector_unavailable", 0) > 0:
                        logger.info("  Vector unavailable: %d", vector_stats.get("vector_unavailable", 0))
                    logger.info("")

                # Log summary
                scores = evaluation["scores"]
                logger.info(
                    "Model %s: Recall=%.2f, Precision=%.2f, F1=%.2f (hits=%d, misses=%d, fp=%d)",
                    model,
                    scores["recall"],
                    scores["precision"],
                    scores["f1"],
                    scores["total_hits"],
                    scores["total_misses"],
                    scores["total_false_positives"],
                )

                # Log detailed hits and misses with explanations
                if evaluation["hits"]:
                    logger.info("  HITS (LLM extracted value matches gold standard):")
                    for hit in evaluation["hits"]:
                        score_str = ""
                        if "similarity_score" in hit:
                            score_str = f" [sim={hit['similarity_score']:.3f}]"
                        method_str = ""
                        if "match_method" in hit:
                            method_str = f" ({hit['match_method']})"
                        logger.info("    ✓ %s: LLM='%s' → matched gold='%s'%s%s",
                                    hit["attribute"], hit["extracted_value"],
                                    hit["matched_expected"], method_str, score_str)

                if evaluation["misses"]:
                    logger.info("  MISSES (gold standard value NOT extracted by LLM):")
                    for miss in evaluation["misses"]:
                        logger.info("    ✗ %s: gold='%s' was not extracted", miss["attribute"], miss["expected_value"])

                if evaluation["false_positives"]:
                    logger.info("  FALSE POSITIVES (LLM extracted value NOT in gold standard):")
                    for fp in evaluation["false_positives"]:
                        score_str = ""
                        if "similarity_score" in fp:
                            score_str = f" [best_sim={fp['similarity_score']:.3f}]"
                        logger.info("    ? %s: LLM='%s' not in gold=%s%s",
                                    fp["attribute"], fp["extracted_value"], fp["expected_values"], score_str)

            except Exception as e:
                logger.error("Model %s failed: %s", model, e)
                abstract_results["models"][model] = {
                    "error": str(e),
                }

        all_results.append(abstract_results)

    return {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "abstracts": all_results,
    }


def print_summary(results: dict, logger: logging.Logger) -> None:
    """Print summary of evaluation results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    # Aggregate by model
    model_scores = {}

    for abstract_result in results["abstracts"]:
        for model, model_result in abstract_result["models"].items():
            if "error" in model_result:
                continue

            if model not in model_scores:
                model_scores[model] = {
                    "total_hits": 0,
                    "total_misses": 0,
                    "total_false_positives": 0,
                    "total_expected": 0,
                    "abstracts_evaluated": 0,
                }

            scores = model_result["evaluation"]["scores"]
            model_scores[model]["total_hits"] += scores["total_hits"]
            model_scores[model]["total_misses"] += scores["total_misses"]
            model_scores[model]["total_false_positives"] += scores["total_false_positives"]
            model_scores[model]["total_expected"] += scores["total_expected"]
            model_scores[model]["abstracts_evaluated"] += 1

    for model, scores in model_scores.items():
        recall = scores["total_hits"] / scores["total_expected"] if scores["total_expected"] > 0 else 0
        precision = scores["total_hits"] / (scores["total_hits"] + scores["total_false_positives"]) if (scores[
                                                                                                            "total_hits"] +
                                                                                                        scores[
                                                                                                            "total_false_positives"]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logger.info("")
        logger.info("Model: %s", model)
        logger.info("  Abstracts evaluated: %d", scores["abstracts_evaluated"])
        logger.info("  Total expected values: %d", scores["total_expected"])
        logger.info("  Hits: %d", scores["total_hits"])
        logger.info("  Misses: %d", scores["total_misses"])
        logger.info("  False positives: %d", scores["total_false_positives"])
        logger.info("  Recall: %.2f%%", recall * 100)
        logger.info("  Precision: %.2f%%", precision * 100)
        logger.info("  F1 Score: %.2f%%", f1 * 100)


def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(description="Evaluate LLM extraction against gold standard")
    parser.add_argument(
        "-n", "--num-papers",
        type=int,
        default=1,
        help="Number of papers to evaluate (default: 1, use -1 for all)"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help="Evaluate a specific paper by ID (e.g., 'malaria_zanzibar')"
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Use string matching instead of LLM semantic evaluation"
    )
    parser.add_argument(
        "--no-vector-eval",
        action="store_true",
        help="Disable vector similarity evaluation (skip to LLM/string matching)"
    )
    parser.add_argument(
        "--vector-high-threshold",
        type=float,
        default=0.85,
        help="Vector similarity threshold for auto-match (default: 0.85)"
    )
    parser.add_argument(
        "--vector-low-threshold",
        type=float,
        default=0.50,
        help="Vector similarity threshold for auto-reject (default: 0.50)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace embedding model for vector similarity (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List available paper IDs and exit"
    )
    parser.add_argument(
        "--validate-thresholds",
        action="store_true",
        help="Run domain validation to find optimal thresholds and exit"
    )
    args = parser.parse_args()

    config = ExtractionConfig()
    logger = configure_logging(debug=config.debug)

    # Handle --validate-thresholds before loading abstracts
    if args.validate_thresholds:
        from midas_llm.utils.evaluation.vector_similarity import run_domain_validation
        logger.info("Running threshold validation with domain-specific test pairs...")
        validation_results = run_domain_validation(model_name=args.embedding_model, logger=logger)

        if "error" in validation_results:
            logger.error("Validation failed: %s", validation_results["error"])
            return

        print("\n" + "=" * 60)
        print("THRESHOLD VALIDATION RESULTS")
        print("=" * 60)
        print(f"\nEmbedding Model: {validation_results['model']}")
        print(f"Optimal Threshold: {validation_results['optimal_threshold']:.2f}")
        print(f"\nAt optimal threshold:")
        opt = validation_results["optimal_metrics"]
        print(f"  Precision: {opt['precision']:.2%}")
        print(f"  Recall:    {opt['recall']:.2%}")
        print(f"  F1 Score:  {opt['f1']:.2%}")
        print(f"  Accuracy:  {opt['accuracy']:.2%}")

        print("\n" + "-" * 60)
        print("Threshold sweep results:")
        for key, metrics in sorted(validation_results.get("threshold_results", {}).items()):
            t = metrics.get("threshold", 0)
            print(f"  {t:.2f}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

        print("\n" + "-" * 60)
        print("Individual pair results at optimal threshold:")
        for pair in validation_results.get("pair_results", []):
            status = "✓" if pair["correct"] else "✗"
            match_str = "should match" if pair["expected_match"] else "should NOT match"
            print(f"  {status} '{pair['text_a']}' ↔ '{pair['text_b']}' ({match_str})")
            print(
                f"      similarity={pair['similarity']:.3f}, predicted={'MATCH' if pair['predicted_match'] else 'NO_MATCH'}")
        return

    # Load gold standard first to support --list-papers
    all_abstracts = load_gold_standard()

    if args.list_papers:
        print("Available paper IDs:")
        for abstract in all_abstracts:
            print(f"  {abstract['id']}: {abstract['title'][:60]}...")
        return

    logger.info("Starting gold standard evaluation...")

    # Autodetect LLM host
    detected_host = autodetect_llm_host(config.active_llm_host, logger=logger)
    if detected_host:
        config.llm_host = detected_host

    if config.show_config:
        config.log_config(logger)

    # Filter abstracts based on args
    if args.paper_id:
        abstracts = [a for a in all_abstracts if a["id"] == args.paper_id]
        if not abstracts:
            logger.error("Paper ID '%s' not found. Use --list-papers to see available IDs.", args.paper_id)
            return
    elif args.num_papers == -1:
        abstracts = all_abstracts
    else:
        abstracts = all_abstracts[:args.num_papers]

    logger.info("Evaluating %d of %d gold standard abstracts", len(abstracts), len(all_abstracts))

    # Evaluation strategy settings
    use_llm_eval = not args.no_llm_eval
    use_vector_eval = not args.no_vector_eval

    # Log evaluation strategy
    if use_vector_eval:
        logger.info("Vector similarity evaluation: ENABLED (model=%s)", args.embedding_model)
        logger.info("  Auto-match threshold: >= %.2f", args.vector_high_threshold)
        logger.info("  Auto-reject threshold: <= %.2f", args.vector_low_threshold)
    else:
        logger.info("Vector similarity evaluation: DISABLED (--no-vector-eval)")

    if use_llm_eval:
        logger.info("LLM semantic matching: ENABLED")
    else:
        logger.info("LLM semantic matching: DISABLED (--no-llm-eval)")

    if not use_vector_eval and not use_llm_eval:
        logger.info("Using string matching only")

    # Create run folder for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = str(REPO_ROOT / "output" / "gold_standard" / "results" / timestamp)
    os.makedirs(run_folder, exist_ok=True)
    logger.info("Run folder: %s", run_folder)

    # Run evaluation
    results = run_evaluation(
        abstracts,
        config,
        logger,
        use_llm_eval=use_llm_eval,
        use_vector_eval=use_vector_eval,
        vector_high_threshold=args.vector_high_threshold,
        vector_low_threshold=args.vector_low_threshold,
        embedding_model=args.embedding_model,
        run_folder=run_folder,
    )

    # Add evaluation config to results for JSON output
    results["evaluation_config"] = {
        "use_vector_eval": use_vector_eval,
        "use_llm_eval": use_llm_eval,
        "vector_high_threshold": args.vector_high_threshold,
        "vector_low_threshold": args.vector_low_threshold,
        "embedding_model": args.embedding_model,
    }

    # Save results to run folder
    output_file = os.path.join(run_folder, "evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    # Generate human-readable evaluation report
    if config.generate_evaluation_report:
        report_file = generate_evaluation_text_report(results, run_folder=run_folder, logger=logger)
        html_report_file = generate_evaluation_html_report(results, run_folder=run_folder, logger=logger)
        logger.info("Human-readable evaluation report: %s", report_file)
        logger.info("HTML evaluation report: %s", html_report_file)

    # Print summary
    print_summary(results, logger)


if __name__ == "__main__":
    main()
