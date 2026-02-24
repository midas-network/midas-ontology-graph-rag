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

from src.midas_llm.data.IdSynonyms import DISEASE_SYNONYMS
from src.midas_llm.utils.parsers.extraction_parser import normalize_absent_value

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from midas_llm.utils.config import ExtractionConfig
from midas_llm.utils.logging.logger import configure_logging
from midas_llm.utils.llm.llm_client import send_to_llm
from midas_llm.utils.llm.llm_utils import autodetect_llm_host
from midas_llm.utils.prompt.builders import prepare_and_display_prompt, build_query
from midas_llm.utils.parsers.extraction_parser import (
    parse_and_display_extracted_data,
    save_response_json,
    normalize_llm_format,
)
from midas_llm.utils.reporting.evaluation_reports import generate_evaluation_text_report, \
    generate_evaluation_html_report, generate_abstract_evaluation_report
from midas_llm.utils.evaluation.vector_similarity import vector_match_tiered

LOGGER = logging.getLogger("midas-llm")

# Repo root for resolving paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Compiled once, reused
_ANNOTATION_PATTERN = re.compile(r'\s*\((inferred|not mentioned)\)')


# ------------------------------------------------------------------
# Gold standard format helpers
# ------------------------------------------------------------------

def get_expected_values(field_data) -> list[str]:
    """Extract plain string values from gold standard v3 format.

    Handles:
      str:            "not mentioned"                        -> ["not mentioned"]
      list of dicts:  [{"value": "X", "tier": "explicit"}]  -> ["X"]
      list of str:    ["X", "Y"]                             -> ["X", "Y"]
      empty list:     []                                     -> []
    """
    if isinstance(field_data, str):
        return [field_data]
    if isinstance(field_data, list):
        return [
            item["value"] if isinstance(item, dict) else str(item)
            for item in field_data
        ]
    return []


def strip_annotations(values: list[str], logger: logging.Logger = LOGGER) -> list[str]:
    """Remove parenthetical annotations like (inferred), (not mentioned) from a list of string values.

    Assumes values have already been converted to strings via get_expected_values().
    """
    result = []
    for v in values:
        if not isinstance(v, str):
            logger.warning(
                "strip_annotations: non-string value, got type %s: %s",
                type(v).__name__, v,
            )
            continue
        result.append(_ANNOTATION_PATTERN.sub('', v).strip())
    return result


def normalize_term(value: str, logger: logging.Logger = LOGGER) -> str:
    """Normalize known synonyms before similarity comparison."""
    if not isinstance(value, str):
        logger.warning("normalize_term: non-string value, got type %s: %s", type(value).__name__, value)
        return str(value)
    return DISEASE_SYNONYMS.get(value.strip().lower(), value)


# ------------------------------------------------------------------
# Gold standard loading
# ------------------------------------------------------------------

def load_gold_standard(path: str | None = None) -> list[dict]:
    """Load gold standard test abstracts."""
    if path is None:
        path = str(REPO_ROOT / "resources" / "test_abstracts.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("abstracts", [])


# ------------------------------------------------------------------
# Matching functions
# ------------------------------------------------------------------

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
        return fallback_string_match(extracted_value, expected_values)


def fallback_string_match(extracted_value: str, expected_values: list[str]) -> tuple[bool, str | None]:
    """Fallback string matching when LLM eval fails."""
    extracted_norm = extracted_value.lower().strip().replace("-", " ").replace("_", " ")

    for expected in expected_values:
        expected_norm = expected.lower().strip().replace("-", " ").replace("_", " ")
        # In fallback_string_match, add a minimum length guard:
        if expected_norm in extracted_norm or extracted_norm in expected_norm:
            if len(extracted_norm) >= 3 and len(expected_norm) >= 3:
                return True, expected
        if extracted_norm == expected_norm:
            return True, expected
        if expected_norm in extracted_norm or extracted_norm in expected_norm:
            return True, expected
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
        date_str = date_str.lower().strip().replace('**', '').strip()
        for month_name, month_num in month_names.items():
            if month_name in date_str:
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return f"{year_match.group(1)}-{month_num}"
        ym_match = re.match(r'^(\d{4})-(\d{1,2})(?:-\d{1,2})?$', date_str)
        if ym_match:
            return f"{ym_match.group(1)}-{ym_match.group(2).zfill(2)}"
        return date_str

    return normalize_date(extracted) == normalize_date(expected)


# ------------------------------------------------------------------
# Core evaluation engine
# ------------------------------------------------------------------

def evaluate_extraction(
        extracted: dict[str, Any],
        expected: dict[str, Any],
        llm_model: str,
        llm_host: str,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_model: str | None = None,
        config: Any = None,
        timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Evaluate extracted attributes against gold standard expected values.

    Uses a three-tier evaluation strategy:
    1. Vector similarity (fast) - auto-match/reject at thresholds
    2. LLM semantic evaluation - for ambiguous vector scores
    3. Fallback string matching
    """
    if embedding_model is None:
        embedding_model = ExtractionConfig().embedding_model

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

    # Track which expected values were matched (keyed by cleaned string)
    matched_expected: dict[str, set[str]] = {attr: set() for attr in expected}

    # Check each extracted attribute
    for attr, data in extracted.items():
        logger.debug("Processing attribute: %s, data type: %s, data: %s",
                      attr, type(data).__name__, data)

        value = data.get("value", "") if isinstance(data, dict) else str(data)

        logger.debug("Attribute: %s, extracted value type: %s, value: %s",
                      attr, type(value).__name__, value)

        if attr not in expected:
            results["not_expected"].append({
                "attribute": attr,
                "extracted_value": value,
            })
            continue

        # --- FIX: convert v3 format (dicts/strings) to plain string list ---
        expected_values = get_expected_values(expected[attr])
        if not expected_values:
            continue

        # Strip (inferred) etc. from expected values
        expected_values = strip_annotations(expected_values, logger)
        expected_values = [normalize_term(v, logger) for v in expected_values]
        logger.debug("Attribute: %s, expected_values (normalized): %s", attr, expected_values)

        # Handle comma-separated extracted values
        if not isinstance(value, str):
            logger.error("Attribute: %s - value is not a string, got type: %s, value: %s",
                          attr, type(value).__name__, value)
            continue

        extracted_values = [v.strip() for v in value.split(",")]

        for ext_val in extracted_values:
            if not ext_val:
                continue

            matched = False
            matched_val = None
            similarity_score = None
            match_method = None

            is_date_field = attr in ("study_dates_start", "study_dates_end")

            # Normalize extracted value using synonym map
            extracted_normalized = normalize_term(ext_val, logger)
            extracted_normalized = normalize_absent_value(extracted_normalized)

            # Tier 1: Vector similarity (if enabled and not a date field)
            decision = None
            if use_vector_eval and not is_date_field:
                try:
                    decision, best_match, similarity_score = vector_match_tiered(
                        extracted_normalized,
                        expected_values,
                        high_threshold=vector_high_threshold,
                        low_threshold=vector_low_threshold,
                        model_name=embedding_model,
                    )
                except Exception as e:
                    logger.error(
                        "Error in vector_match_tiered for attr=%s, extracted=%s, expected=%s: %s",
                        attr, extracted_normalized, expected_values, e,
                    )
                    decision = "UNAVAILABLE"

                if decision == "MATCH":
                    matched = True
                    matched_val = best_match
                    match_method = "vector"
                    results["vector_stats"]["auto_matches"] += 1
                    logger.debug("Vector auto-match: '%s' <-> '%s' (score=%.3f)",
                                  extracted_normalized, matched_val, similarity_score)
                elif decision == "NO_MATCH":
                    if dates_match(extracted_normalized, expected_values[0] if expected_values else ""):
                        matched = True
                        matched_val = expected_values[0]
                        match_method = "date_format"
                    else:
                        results["vector_stats"]["auto_rejects"] += 1
                        logger.debug("Vector auto-reject: '%s' (score=%.3f)",
                                      extracted_normalized, similarity_score)
                elif decision == "AMBIGUOUS":
                    results["vector_stats"]["ambiguous_to_llm"] += 1
                    logger.debug("Vector ambiguous: '%s' (score=%.3f), falling through to LLM",
                                  extracted_normalized, similarity_score)
                else:  # UNAVAILABLE
                    results["vector_stats"]["vector_unavailable"] += 1

            # Tier 2: LLM semantic evaluation
            if not matched and matched_val is None and use_llm_eval:
                if not use_vector_eval or similarity_score is None or (
                        vector_low_threshold < (similarity_score or 0) < vector_high_threshold
                ):
                    try:
                        api_type = config.llm_api_type if config else "ollama"
                        matched, matched_val = evaluate_semantic_match(
                            attr, extracted_normalized, expected_values,
                            llm_model, llm_host,
                            logger=logger,
                            api_type=api_type,
                            timeout_seconds=timeout_seconds,
                        )
                        if matched:
                            match_method = "llm"
                    except Exception as e:
                        logger.warning("LLM eval failed for '%s': %s", extracted_normalized, e)

            # Tier 3: Fallback string matching
            if not matched and matched_val is None:
                matched, matched_val = fallback_string_match(extracted_normalized, expected_values)
                if matched:
                    match_method = "string"

            if matched:
                hit_record = {
                    "attribute": attr,
                    "extracted_value": extracted_normalized,
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
                    "extracted_value": extracted_normalized,
                    "expected_values": expected_values,
                }
                if similarity_score is not None:
                    fp_record["similarity_score"] = round(similarity_score, 4)
                results["false_positives"].append(fp_record)

    # --- FIX: Check for misses using get_expected_values + strip_annotations ---
    for attr, raw_expected in expected.items():
        for exp_val in strip_annotations(get_expected_values(raw_expected), logger):
            normalized_exp = normalize_term(exp_val, logger)
            if normalized_exp not in matched_expected.get(attr, set()):
                results["misses"].append({
                    "attribute": attr,
                    "expected_value": exp_val,
                })

    # --- FIX: Calculate total_expected using get_expected_values ---
    total_expected = sum(len(get_expected_values(v)) for v in expected.values())
    total_hits = len(results["hits"])
    total_misses = len(results["misses"])
    total_false_positives = len(results["false_positives"])

    results["scores"] = {
        "total_expected": total_expected,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_false_positives": total_false_positives,
        "recall": total_hits / total_expected if total_expected > 0 else 0,
        "precision": total_hits / (total_hits + total_false_positives)
        if (total_hits + total_false_positives) > 0 else 0,
    }

    p, r = results["scores"]["precision"], results["scores"]["recall"]
    results["scores"]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return results


# ------------------------------------------------------------------
# Evaluation runner
# ------------------------------------------------------------------

def run_evaluation(
        abstracts: list[dict],
        config: ExtractionConfig,
        logger: logging.Logger,
        use_llm_eval: bool = True,
        use_vector_eval: bool = True,
        vector_high_threshold: float = 0.85,
        vector_low_threshold: float = 0.50,
        embedding_model: str | None = None,
        run_folder: str | None = None,
) -> dict[str, Any]:
    """Run evaluation on gold standard abstracts.

    Uses a three-tier evaluation strategy:
    1. Vector similarity (fast) - auto-match/reject at thresholds
    2. LLM semantic evaluation - for ambiguous vector scores
    3. String matching - fallback
    """
    if embedding_model is None:
        embedding_model = config.embedding_model

    all_results = []
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_models]

    for abstract_data in abstracts:
        abstract_id = abstract_data["id"]
        title = abstract_data["title"]
        abstract = abstract_data["abstract"]
        expected = abstract_data["expected"]

        logger.info("=" * 60)
        logger.info("Evaluating: %s", title[:60])
        logger.info("=" * 60)

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

        if run_folder:
            prompt_file = Path(run_folder) / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            logger.info("Saved prompt to: %s", prompt_file)

        abstract_results = {
            "id": abstract_id,
            "title": title,
            "models": {},
        }

        if run_folder:
            abstract_dir = Path(run_folder) / abstract_id
            os.makedirs(abstract_dir, exist_ok=True)
            logger.info("Created abstract directory: %s", abstract_dir)

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
                    config=config,
                    timeout_seconds=config.llm_timeout,
                )

                # Save artifacts
                if run_folder:
                    safe_model_name = model.replace("/", "-").replace("\\", "-")

                    response_file = abstract_dir / f"{safe_model_name}_response.txt"
                    response_file.write_text(response.content, encoding="utf-8")
                    logger.info("Saved raw response to: %s", response_file)

                    normalized = normalize_llm_format(response.content)
                    normalized_file = abstract_dir / f"{safe_model_name}_response_normalized.txt"
                    normalized_file.write_text(normalized, encoding="utf-8")
                    logger.info("Saved normalized response to: %s", normalized_file)

                    save_response_json(
                        response_content=response.content,
                        model=model,
                        output_path=abstract_dir,
                        abstract_id=abstract_id,
                        evaluation=evaluation,
                        logger=logger,
                    )

                abstract_results["models"][model] = {
                    "extracted": extracted,
                    "evaluation": evaluation,
                    "raw_response": response.content[:2000],
                }

                # Log legend
                logger.info("")
                logger.info("EVALUATION LEGEND:")
                logger.info("  HIT = LLM extracted a value that matches something in the gold standard")
                logger.info("  MISS = Gold standard expects a value, but LLM didn't extract it")
                logger.info("  FALSE POSITIVE = LLM extracted a value not in the gold standard")
                logger.info("")

                # Log vector stats
                vector_stats = evaluation.get("vector_stats", {})
                if use_vector_eval and any(vector_stats.values()):
                    logger.info("VECTOR EVALUATION STATS:")
                    logger.info("  Auto-matches (score >= %.2f): %d",
                                vector_high_threshold, vector_stats.get("auto_matches", 0))
                    logger.info("  Auto-rejects (score <= %.2f): %d",
                                vector_low_threshold, vector_stats.get("auto_rejects", 0))
                    logger.info("  Ambiguous -> LLM: %d", vector_stats.get("ambiguous_to_llm", 0))
                    if vector_stats.get("vector_unavailable", 0) > 0:
                        logger.info("  Vector unavailable: %d", vector_stats.get("vector_unavailable", 0))
                    logger.info("")

                # Log summary
                scores = evaluation["scores"]
                logger.info(
                    "Model %s: Recall=%.2f, Precision=%.2f, F1=%.2f (hits=%d, misses=%d, fp=%d)",
                    model, scores["recall"], scores["precision"], scores["f1"],
                    scores["total_hits"], scores["total_misses"], scores["total_false_positives"],
                )

                # Log detailed results
                if evaluation["hits"]:
                    logger.info("  HITS (LLM extracted value matches gold standard):")
                    for hit in evaluation["hits"]:
                        score_str = f" [sim={hit['similarity_score']:.3f}]" if "similarity_score" in hit else ""
                        method_str = f" ({hit['match_method']})" if "match_method" in hit else ""
                        logger.info("    + %s: LLM='%s' -> matched gold='%s'%s%s",
                                    hit["attribute"], hit["extracted_value"],
                                    hit["matched_expected"], method_str, score_str)

                if evaluation["misses"]:
                    logger.info("  MISSES (gold standard value NOT extracted by LLM):")
                    for miss in evaluation["misses"]:
                        logger.info("    x %s: gold='%s' was not extracted",
                                    miss["attribute"], miss["expected_value"])

                if evaluation["false_positives"]:
                    logger.info("  FALSE POSITIVES (LLM extracted value NOT in gold standard):")
                    for fp in evaluation["false_positives"]:
                        score_str = f" [best_sim={fp['similarity_score']:.3f}]" if "similarity_score" in fp else ""
                        logger.info("    ? %s: LLM='%s' not in gold=%s%s",
                                    fp["attribute"], fp["extracted_value"],
                                    fp["expected_values"], score_str)

            except Exception as e:
                logger.error("Model %s failed: %s", model, e)
                abstract_results["models"][model] = {"error": str(e)}

            # Save per-abstract evaluation report
            if run_folder and not evaluation.get("error"):
                generate_abstract_evaluation_report(
                    abstract_id=abstract_id,
                    title=title,
                    model=model,
                    evaluation=evaluation,
                    output_dir=str(abstract_dir),
                    logger=logger,
                )

        all_results.append(abstract_results)

    return {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "abstracts": all_results,
    }


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def print_summary(results: dict, logger: logging.Logger) -> None:
    """Print summary of evaluation results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

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
        total = scores["total_hits"] + scores["total_false_positives"]
        recall = scores["total_hits"] / scores["total_expected"] if scores["total_expected"] > 0 else 0
        precision = scores["total_hits"] / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(description="Evaluate LLM extraction against gold standard")
    parser.add_argument("-n", "--num-papers", type=int, default=10,
                        help="Number of papers to evaluate (default: 1, use -1 for all)")
    parser.add_argument("--paper-id", type=str, default=None,
                        help="Evaluate a specific paper by ID")
    parser.add_argument("--no-llm-eval", action="store_true",
                        help="Use string matching instead of LLM semantic evaluation")
    parser.add_argument("--no-vector-eval", action="store_true",
                        help="Disable vector similarity evaluation")
    parser.add_argument("--vector-high-threshold", type=float, default=0.85,
                        help="Vector similarity threshold for auto-match (default: 0.85)")
    parser.add_argument("--vector-low-threshold", type=float, default=0.50,
                        help="Vector similarity threshold for auto-reject (default: 0.50)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="HuggingFace embedding model for vector similarity")
    parser.add_argument("--list-papers", action="store_true",
                        help="List available paper IDs and exit")
    parser.add_argument("--validate-thresholds", action="store_true",
                        help="Run domain validation to find optimal thresholds and exit")
    args = parser.parse_args()

    config = ExtractionConfig()
    logger = configure_logging(debug=config.debug)

    if args.embedding_model is None:
        args.embedding_model = config.embedding_model

    # Handle --validate-thresholds
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
        opt = validation_results["optimal_metrics"]
        print(f"\nAt optimal threshold:")
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
            status = "+" if pair["correct"] else "x"
            match_str = "should match" if pair["expected_match"] else "should NOT match"
            print(f"  {status} '{pair['text_a']}' <-> '{pair['text_b']}' ({match_str})")
            print(f"      similarity={pair['similarity']:.3f}, "
                  f"predicted={'MATCH' if pair['predicted_match'] else 'NO_MATCH'}")
        return

    # Load gold standard
    all_abstracts = load_gold_standard()

    if args.list_papers:
        print("Available paper IDs:")
        for abstract in all_abstracts:
            print(f"  {abstract['id']}: {abstract['title'][:60]}...")
        return

    logger.info("Starting gold standard evaluation...")

    detected_host = autodetect_llm_host(config.active_llm_host, logger=logger)
    if detected_host:
        config.llm_host = detected_host

    if config.show_config:
        config.log_config(logger)

    # Filter abstracts
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

    use_llm_eval = not args.no_llm_eval
    use_vector_eval = not args.no_vector_eval

    if use_vector_eval:
        logger.info("Vector similarity evaluation: ENABLED (model=%s)", args.embedding_model)
        logger.info("  Auto-match threshold: >= %.2f", args.vector_high_threshold)
        logger.info("  Auto-reject threshold: <= %.2f", args.vector_low_threshold)
    else:
        logger.info("Vector similarity evaluation: DISABLED")

    if use_llm_eval:
        logger.info("LLM semantic matching: ENABLED")
    else:
        logger.info("LLM semantic matching: DISABLED")

    if not use_vector_eval and not use_llm_eval:
        logger.info("Using string matching only")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = str(REPO_ROOT / "output" / "gold_standard" / "results" / timestamp)
    os.makedirs(run_folder, exist_ok=True)
    logger.info("Run folder: %s", run_folder)

    results = run_evaluation(
        abstracts, config, logger,
        use_llm_eval=use_llm_eval,
        use_vector_eval=use_vector_eval,
        vector_high_threshold=args.vector_high_threshold,
        vector_low_threshold=args.vector_low_threshold,
        embedding_model=args.embedding_model,
        run_folder=run_folder,
    )

    results["evaluation_config"] = {
        "use_vector_eval": use_vector_eval,
        "use_llm_eval": use_llm_eval,
        "vector_high_threshold": args.vector_high_threshold,
        "vector_low_threshold": args.vector_low_threshold,
        "embedding_model": args.embedding_model,
    }

    output_file = os.path.join(run_folder, "evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    if config.generate_evaluation_report:
        report_file = generate_evaluation_text_report(results, run_folder=run_folder, logger=logger)
        html_report_file = generate_evaluation_html_report(results, run_folder=run_folder, logger=logger)
        logger.info("Human-readable evaluation report: %s", report_file)
        logger.info("HTML evaluation report: %s", html_report_file)

    print_summary(results, logger)


if __name__ == "__main__":
    main()