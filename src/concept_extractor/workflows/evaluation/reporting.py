"""Summary reporting and run-configuration/path helpers."""
from __future__ import annotations

import argparse
import logging
import re
from typing import Any

from concept_extractor.utils.config import ExtractionConfig

LOGGER = logging.getLogger("midas-llm")


# ------------------------------------------------------------------
# Output path helpers
# ------------------------------------------------------------------

def sanitize_model_directory_name(model: str) -> str:
    """Build a filesystem-safe directory name from a model identifier."""
    model_leaf = model.strip().split("/")[-1]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_leaf).strip("._-")
    return sanitized or "unknown-model"


def results_model_directory_name(config: ExtractionConfig) -> str:
    """Choose a model directory name for grouping evaluation runs."""
    models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]
    if not models:
        return "unknown-model"
    if len(models) == 1:
        return sanitize_model_directory_name(models[0])
    return "multi-model"


def parse_embedding_models_arg(value: str) -> list[str]:
    """Parse comma-separated embedding models from CLI."""
    return [model.strip() for model in value.split(",") if model.strip()]


def log_active_run_configuration(
    *,
    logger: logging.Logger,
    config: ExtractionConfig,
    args: argparse.Namespace,
    use_vector_eval: bool,
    use_llm_eval: bool,
) -> None:
    """Log only configuration entries that are actively used for this run."""
    active_models = config.active_llm_models if config.active_llm_models else [config.active_llm_model]

    logger.info("Active run configuration:")
    logger.info("  llm_api_type: %s", config.llm_api_type)
    logger.info("  active_llm_host: %s", config.active_llm_host)
    logger.info("  active_llm_models: %s", active_models)
    logger.info("  llm_timeout: %s", config.llm_timeout)
    logger.info("  constrained_decoding: %s", args.constrained)
    if args.constrained:
        logger.info("  ontology_path: %s", args.ontology_path)
    if config.llm_api_type == "openai_compatible":
        logger.info(
            "  openai_response_format: %s",
            "json_schema" if args.openai_json_schema else "json_object",
        )
    logger.info("  validate_constrained_json: %s", args.validate_constrained_json)
    if args.validate_constrained_json and args.validation_schema_path:
        logger.info("  validation_schema_path: %s", args.validation_schema_path)
    logger.info("  llm_semantic_matching: %s", use_llm_eval)
    logger.info("  vector_similarity: %s", use_vector_eval)
    if use_vector_eval:
        logger.info("  embedding_models: %s", args.embedding_models)
        logger.info("  vector_high_threshold: %.2f", args.vector_high_threshold)
        logger.info("  vector_low_threshold: %.2f", args.vector_low_threshold)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def add_optional_float(accumulator: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, (int, float)):
        accumulator[key] += float(value)
        accumulator[f"{key}_count"] += 1


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def print_summary(results: dict, logger: logging.Logger) -> None:
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
                    "request_duration_s": 0.0,
                    "request_duration_s_count": 0,
                    "total_duration_s": 0.0,
                    "total_duration_s_count": 0,
                    "load_duration_s": 0.0,
                    "load_duration_s_count": 0,
                    "prompt_eval_duration_s": 0.0,
                    "prompt_eval_duration_s_count": 0,
                    "eval_duration_s": 0.0,
                    "eval_duration_s_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                }

            scores = model_result["evaluation"]["scores"]
            model_scores[model]["total_hits"] += scores["total_hits"]
            model_scores[model]["total_misses"] += scores["total_misses"]
            model_scores[model]["total_false_positives"] += scores["total_false_positives"]
            model_scores[model]["total_expected"] += scores["total_expected"]
            model_scores[model]["abstracts_evaluated"] += 1
            timing = model_result.get("timing", {})
            if isinstance(timing, dict):
                add_optional_float(
                    model_scores[model], "request_duration_s", timing.get("request_duration_s")
                )
                add_optional_float(
                    model_scores[model], "total_duration_s", timing.get("total_duration_s")
                )
                add_optional_float(
                    model_scores[model], "load_duration_s", timing.get("load_duration_s")
                )
                add_optional_float(
                    model_scores[model], "prompt_eval_duration_s",
                    timing.get("prompt_eval_duration_s"),
                )
                add_optional_float(
                    model_scores[model], "eval_duration_s", timing.get("eval_duration_s")
                )
                if isinstance(timing.get("prompt_tokens"), int):
                    model_scores[model]["prompt_tokens"] += timing["prompt_tokens"]
                if isinstance(timing.get("completion_tokens"), int):
                    model_scores[model]["completion_tokens"] += timing["completion_tokens"]
                if isinstance(timing.get("reasoning_tokens"), int):
                    model_scores[model]["reasoning_tokens"] += timing["reasoning_tokens"]

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

        avg_request = (
            scores["request_duration_s"] / scores["request_duration_s_count"]
            if scores["request_duration_s_count"] > 0 else None
        )
        avg_total = (
            scores["total_duration_s"] / scores["total_duration_s_count"]
            if scores["total_duration_s_count"] > 0 else None
        )
        avg_prompt_eval = (
            scores["prompt_eval_duration_s"] / scores["prompt_eval_duration_s_count"]
            if scores["prompt_eval_duration_s_count"] > 0 else None
        )
        avg_eval = (
            scores["eval_duration_s"] / scores["eval_duration_s_count"]
            if scores["eval_duration_s_count"] > 0 else None
        )
        avg_load = (
            scores["load_duration_s"] / scores["load_duration_s_count"]
            if scores["load_duration_s_count"] > 0 else None
        )
        total_request = (
            scores["request_duration_s"]
            if scores["request_duration_s_count"] > 0 else None
        )
        total_backend = (
            scores["total_duration_s"]
            if scores["total_duration_s_count"] > 0 else None
        )
        total_prompt_eval = (
            scores["prompt_eval_duration_s"]
            if scores["prompt_eval_duration_s_count"] > 0 else None
        )
        total_eval = (
            scores["eval_duration_s"]
            if scores["eval_duration_s_count"] > 0 else None
        )
        total_load = (
            scores["load_duration_s"]
            if scores["load_duration_s_count"] > 0 else None
        )

        logger.info("  Timing:")
        logger.info(
            "    Request round-trip: total=%s avg=%s over %d call(s)",
            fmt_seconds(total_request),
            fmt_seconds(avg_request),
            scores["request_duration_s_count"],
        )
        logger.info(
            "    Backend total: total=%s avg=%s over %d call(s)",
            fmt_seconds(total_backend),
            fmt_seconds(avg_total),
            scores["total_duration_s_count"],
        )
        logger.info(
            "    Prompt eval (input processing): total=%s avg=%s over %d call(s)",
            fmt_seconds(total_prompt_eval),
            fmt_seconds(avg_prompt_eval),
            scores["prompt_eval_duration_s_count"],
        )
        logger.info(
            "    Eval/generation (\"thinking\"): total=%s avg=%s over %d call(s)",
            fmt_seconds(total_eval),
            fmt_seconds(avg_eval),
            scores["eval_duration_s_count"],
        )
        logger.info(
            "    Model load: total=%s avg=%s over %d call(s)",
            fmt_seconds(total_load),
            fmt_seconds(avg_load),
            scores["load_duration_s_count"],
        )
        logger.info(
            "    Tokens: prompt=%d completion=%d reasoning=%d",
            scores["prompt_tokens"],
            scores["completion_tokens"],
            scores["reasoning_tokens"],
        )
