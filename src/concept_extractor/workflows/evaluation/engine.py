"""Core evaluation engine and orchestration."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from concept_extractor.utils.config import ExtractionConfig
from concept_extractor.utils.llm.llm_client import send_to_llm
from concept_extractor.utils.prompt.builders import prepare_and_display_prompt, build_query
from concept_extractor.utils.reporting.evaluation_reports import generate_abstract_evaluation_report
from concept_extractor.utils.parsers.extraction_parser import normalize_absent_value

from .parsing import (
    get_expected_values,
    strip_annotations,
    normalize_term,
    parse_constrained_response,
    JSON_OUTPUT_SUFFIX,
)
from .matching import (
    vector_match_any_model,
    evaluate_semantic_match,
    fallback_string_match,
    dates_match,
    _format_vector_model_scores,
    _vector_model_score_sort_key,
)

LOGGER = logging.getLogger("midas-llm")


# ------------------------------------------------------------------
# Core evaluation engine (unchanged)
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
        embedding_models: list[str] | None = None,
        config: Any = None,
        timeout_seconds: int = 30,
) -> dict[str, Any]:
    resolved_config = config if isinstance(config, ExtractionConfig) else ExtractionConfig.from_yaml()

    if embedding_models is None:
        embedding_models = resolved_config.embedding_models

    if not embedding_models:
        embedding_models = [resolved_config.embedding_model]

    if use_vector_eval:
        logger.info(
            "Vector similarity embedding models (%d): %s",
            len(embedding_models),
            ", ".join(embedding_models),
        )

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

    matched_expected: dict[str, set[str]] = {attr: set() for attr in expected}

    for attr, data in extracted.items():
        logger.debug("Processing attribute: %s, data type: %s", attr, type(data).__name__)

        value = data.get("value", "") if isinstance(data, dict) else str(data)

        if attr not in expected:
            results["not_expected"].append({
                "attribute": attr,
                "extracted_value": value,
            })
            continue

        expected_values = get_expected_values(expected[attr])
        if not expected_values:
            continue

        expected_values = strip_annotations(expected_values, logger)
        expected_values = [normalize_term(v, logger) for v in expected_values]

        if not isinstance(value, str):
            logger.error("Attribute: %s - value is not a string: %s", attr, type(value).__name__)
            continue

        extracted_values = [v.strip() for v in value.split(",")]

        for ext_val in extracted_values:
            if not ext_val:
                continue

            matched = False
            matched_val = None
            similarity_score = None
            vector_model_scores: list[dict[str, Any]] = []
            vector_selected_model: str | None = None
            match_method = None

            is_date_field = attr in ("study_dates_start", "study_dates_end")

            extracted_normalized = normalize_term(ext_val, logger)
            extracted_normalized = normalize_absent_value(extracted_normalized)

            # Tier 1: Vector similarity
            decision = None
            if use_vector_eval and not is_date_field:
                try:
                    decision, best_match, similarity_score, vector_model_scores, vector_selected_model = vector_match_any_model(
                        extracted_normalized,
                        expected_values,
                        embedding_models,
                        vector_high_threshold=vector_high_threshold,
                        vector_low_threshold=vector_low_threshold,
                    )
                except Exception as e:
                    logger.error("Vector match error for %s: %s", attr, e)
                    decision = "UNAVAILABLE"

                if decision == "MATCH":
                    matched, matched_val, match_method = True, best_match, "vector"
                    results["vector_stats"]["auto_matches"] += 1
                elif decision == "NO_MATCH":
                    if dates_match(extracted_normalized, expected_values[0] if expected_values else ""):
                        matched, matched_val, match_method = True, expected_values[0], "date_format"
                    else:
                        results["vector_stats"]["auto_rejects"] += 1
                elif decision == "AMBIGUOUS":
                    results["vector_stats"]["ambiguous_to_llm"] += 1
                else:
                    results["vector_stats"]["vector_unavailable"] += 1

                if vector_model_scores:
                    ranked_vector_scores = [
                        (idx, entry)
                        for idx, entry in enumerate(vector_model_scores)
                        if isinstance(entry, dict)
                    ]
                    ranked_vector_scores.sort(
                        key=lambda item: _vector_model_score_sort_key(item[1], item[0])
                    )
                    per_model_summary = "; ".join(
                        (
                            f"{entry['model']}:{entry['decision']}:"
                            f"{entry['similarity_score']:.4f}"
                            + (
                                f":{entry['best_match']}"
                                if entry.get("best_match")
                                else ""
                            )
                        )
                        for _, entry in ranked_vector_scores
                    )
                    logger.info(
                        "Vector similarity decision | attr=%s value=%s result=%s best_score=%.4f details=[%s]",
                        attr,
                        extracted_normalized,
                        decision or "UNAVAILABLE",
                        similarity_score or 0.0,
                        per_model_summary,
                    )

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
                            logger=logger, api_type=api_type,
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
                if vector_model_scores:
                    hit_record["vector_model_scores"] = vector_model_scores
                if vector_selected_model:
                    hit_record["vector_selected_model"] = vector_selected_model
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
                if vector_model_scores:
                    fp_record["vector_model_scores"] = vector_model_scores
                if vector_selected_model:
                    fp_record["vector_selected_model"] = vector_selected_model
                results["false_positives"].append(fp_record)

    # Check for misses
    for attr, raw_expected in expected.items():
        for exp_val in strip_annotations(get_expected_values(raw_expected), logger):
            normalized_exp = normalize_term(exp_val, logger)
            if normalized_exp not in matched_expected.get(attr, set()):
                results["misses"].append({
                    "attribute": attr,
                    "expected_value": exp_val,
                })

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
        embedding_models: list[str] | None = None,
        run_folder: str | None = None,
        json_schema: dict[str, Any] | None = None,  # ── NEW
        openai_allow_json_schema: bool = False,
        validate_constrained_json: bool = False,
        validation_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embedding_models is None:
        embedding_models = config.embedding_models

    all_results = []
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]

    # ── NEW: Log constrained decoding status ──
    if json_schema is not None:
        logger.info(
            "Constrained decoding: ENABLED (%d schema properties)",
            len(json_schema.get("properties", {})),
        )
    else:
        logger.info("Constrained decoding: DISABLED")

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

        # ── NEW: Append JSON instructions when using constrained mode ──
        if json_schema is not None:
            prompt = prompt + JSON_OUTPUT_SUFFIX

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
                # ── NEW: Pass json_schema to send_to_llm ──
                response = send_to_llm(
                    prompt=prompt,
                    llm_model=model,
                    llm_host=config.active_llm_host,
                    timeout_seconds=config.llm_timeout,
                    logger=logger,
                    api_type=config.llm_api_type,
                    json_schema=json_schema,  # None → unconstrained
                    allow_json_schema=openai_allow_json_schema,
                )
                timing = {
                    "request_duration_s": response.request_duration_s,
                    "total_duration_s": response.total_duration_s,
                    "load_duration_s": response.load_duration_s,
                    "prompt_eval_duration_s": response.prompt_eval_duration_s,
                    "eval_duration_s": response.eval_duration_s,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                }
                logger.info(
                    "LLM timing | request=%.2fs backend_total=%s prompt_eval=%s eval=%s",
                    timing["request_duration_s"] or 0.0,
                    (
                        f"{timing['total_duration_s']:.2f}s"
                        if isinstance(timing["total_duration_s"], (int, float))
                        else "n/a"
                    ),
                    (
                        f"{timing['prompt_eval_duration_s']:.2f}s"
                        if isinstance(timing["prompt_eval_duration_s"], (int, float))
                        else "n/a"
                    ),
                    (
                        f"{timing['eval_duration_s']:.2f}s"
                        if isinstance(timing["eval_duration_s"], (int, float))
                        else "n/a"
                    ),
                )

                if response.constrained:
                    logger.info("Parsing constrained JSON response")
                else:
                    logger.warning(
                        "Response was not marked constrained; attempting strict JSON parse."
                    )
                extracted = parse_constrained_response(
                    response.content,
                    validation_schema=validation_schema,
                    validate_schema=validate_constrained_json,
                    logger=logger,
                )
                if not extracted:
                    raise ValueError(
                        "Failed to parse constrained JSON response. "
                        "Free-text responses are unsupported."
                    )

                logger.info("Evaluating extraction with semantic matching...")
                evaluation = evaluate_extraction(
                    extracted, expected,
                    model, config.active_llm_host, logger,
                    use_llm_eval=use_llm_eval,
                    use_vector_eval=use_vector_eval,
                    vector_high_threshold=vector_high_threshold,
                    vector_low_threshold=vector_low_threshold,
                    embedding_models=embedding_models,
                    config=config,
                    timeout_seconds=config.llm_timeout,
                )

                # Save artifacts
                if run_folder:
                    safe_model_name = model.replace("/", "-").replace("\\", "-")

                    response_file = abstract_dir / f"{safe_model_name}_response.json"
                    response_file.write_text(response.content, encoding="utf-8")

                    # ── NEW: Save constrained status ──
                    meta_file = abstract_dir / f"{safe_model_name}_meta.json"
                    meta_file.write_text(json.dumps({
                        "model": model,
                        "constrained": response.constrained,
                        "response_length": len(response.content),
                        "timing": timing,
                    }, indent=2), encoding="utf-8")

                abstract_results["models"][model] = {
                    "extracted": extracted,
                    "evaluation": evaluation,
                    "raw_response": response.content[:2000],
                    "constrained": response.constrained,  # ── NEW
                    "timing": timing,
                }

                # Log legend
                logger.info("")
                logger.info("EVALUATION LEGEND:")
                logger.info("  HIT = LLM extracted a value that matches something in the evaluation dataset")
                logger.info("  MISS = Evaluation dataset expects a value, but LLM didn't extract it")
                logger.info("  FALSE POSITIVE = LLM extracted a value not in the evaluation dataset")
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
                    "Model %s: Recall=%.2f, Precision=%.2f, F1=%.2f "
                    "(hits=%d, misses=%d, fp=%d, constrained=%s)",
                    model, scores["recall"], scores["precision"], scores["f1"],
                    scores["total_hits"], scores["total_misses"],
                    scores["total_false_positives"], response.constrained,
                )

                # Log detailed results
                if evaluation["hits"]:
                    logger.info("  HITS (LLM extracted value matches evaluation dataset):")
                    for hit in evaluation["hits"]:
                        score_str = f" [sim={hit['similarity_score']:.3f}]" if "similarity_score" in hit else ""
                        method_str = f" ({hit['match_method']})" if "match_method" in hit else ""
                        vector_model_str = (
                            f" [vector_model={hit['vector_selected_model']}]"
                            if hit.get("vector_selected_model")
                            else ""
                        )
                        vector_details = _format_vector_model_scores(hit.get("vector_model_scores"))
                        vector_details_str = (
                            f" [vector_details={vector_details}]"
                            if vector_details
                            else ""
                        )
                        logger.info("    + %s: LLM='%s' -> matched dataset='%s'%s%s%s%s",
                                    hit["attribute"], hit["extracted_value"],
                                    hit["matched_expected"], method_str, score_str,
                                    vector_model_str, vector_details_str)

                if evaluation["misses"]:
                    logger.info("  MISSES (dataset value NOT extracted by LLM):")
                    for miss in evaluation["misses"]:
                        logger.info("    x %s: dataset='%s' was not extracted",
                                    miss["attribute"], miss["expected_value"])

                if evaluation["false_positives"]:
                    logger.info("  FALSE POSITIVES (LLM extracted value NOT in evaluation dataset):")
                    for fp in evaluation["false_positives"]:
                        score_str = f" [best_sim={fp['similarity_score']:.3f}]" if "similarity_score" in fp else ""
                        vector_model_str = (
                            f" [vector_model={fp['vector_selected_model']}]"
                            if fp.get("vector_selected_model")
                            else ""
                        )
                        vector_details = _format_vector_model_scores(fp.get("vector_model_scores"))
                        vector_details_str = (
                            f" [vector_details={vector_details}]"
                            if vector_details
                            else ""
                        )
                        logger.info("    ? %s: LLM='%s' not in gold=%s%s%s%s",
                                    fp["attribute"], fp["extracted_value"],
                                    fp["expected_values"], score_str,
                                    vector_model_str, vector_details_str)

            except Exception as e:
                logger.error("Model %s failed: %s", model, e)
                abstract_results["models"][model] = {"error": str(e)}

            # Save per-abstract evaluation report
            if run_folder and "error" not in abstract_results["models"].get(model, {}):
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
