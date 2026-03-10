#!/usr/bin/env python
"""Evaluation script for LLM extraction accuracy.

Compares extracted attributes against expected values using LLM-based semantic matching.
Supports constrained decoding via MIDAS structured_vocab JSON schema (--constrained flag).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path when running from repo root
_src_path = Path(__file__).resolve().parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from concept_extractor.utils.config import ExtractionConfig
from concept_extractor.utils.logging.logger import configure_logging
from concept_extractor.utils.llm.llm_utils import autodetect_llm_host
from concept_extractor.utils.reporting.evaluation_reports import generate_evaluation_text_report

try:
    from evaluation import (
        REPO_ROOT,
        DEFAULT_EVALUATION_DATASET_PATH,
        LEGACY_EVALUATION_DATASET_PATH,
        DEFAULT_ONTOLOGY_PATH,
        DEFAULT_GOLD_OUTPUT_DIR,
        DEFAULT_VALIDATION_SCHEMA_PATH,
        load_evaluation_dataset,
        run_evaluation,
        print_summary,
        parse_embedding_models_arg,
        results_model_directory_name,
        log_active_run_configuration,
    )
except ModuleNotFoundError:  # pragma: no cover - supports python -m invocation
    from concept_extractor.workflows.evaluation import (
        REPO_ROOT,
        DEFAULT_EVALUATION_DATASET_PATH,
        LEGACY_EVALUATION_DATASET_PATH,
        DEFAULT_ONTOLOGY_PATH,
        DEFAULT_GOLD_OUTPUT_DIR,
        DEFAULT_VALIDATION_SCHEMA_PATH,
        load_evaluation_dataset,
        run_evaluation,
        print_summary,
        parse_embedding_models_arg,
        results_model_directory_name,
        log_active_run_configuration,
    )

LOGGER = logging.getLogger("midas-llm")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM extraction against an evaluation dataset"
    )
    parser.add_argument("-n", "--num-papers", type=int, default=20,
                        help="Number of papers to evaluate (default: 1, use -1 for all)")
    parser.add_argument("--paper-id", type=str, default=None,
                        help="Evaluate a specific paper by ID")
    parser.add_argument(
        "--evaluation-dataset-path",
        "--gold-standard-path",
        dest="evaluation_dataset_path",
        type=str,
        default=str(DEFAULT_EVALUATION_DATASET_PATH),
        help=(
            "Path to evaluation dataset JSON "
            f"(default: {DEFAULT_EVALUATION_DATASET_PATH})"
        ),
    )
    parser.add_argument("--no-llm-eval", action="store_true",
                        help="Use string matching instead of LLM semantic evaluation")
    parser.add_argument("--no-vector-eval", action="store_true",
                        help="Disable vector similarity evaluation")
    parser.add_argument("--vector-high-threshold", type=float, default=0.85,
                        help="Vector similarity threshold for auto-match (default: 0.85)")
    parser.add_argument("--vector-low-threshold", type=float, default=0.50,
                        help="Vector similarity threshold for auto-reject (default: 0.50)")
    parser.add_argument(
        "--embedding-models",
        type=str,
        default=None,
        help=(
            "Comma-separated HuggingFace embedding models for vector similarity "
            "(default: use config embedding_models array)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Deprecated alias for a single embedding model.",
    )
    parser.add_argument("--list-papers", action="store_true",
                        help="List available paper IDs and exit")
    parser.add_argument("--validate-thresholds", action="store_true",
                        help="Run domain validation to find optimal thresholds and exit")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_GOLD_OUTPUT_DIR),
        help=(
            "Output root directory for timestamped evaluation runs "
            f"(default: {DEFAULT_GOLD_OUTPUT_DIR})"
        ),
    )

    # ── NEW: Constrained decoding flags ──
    parser.add_argument(
        "--constrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable constrained decoding with MIDAS structured_vocab JSON schema (default: enabled)",
    )
    parser.add_argument(
        "--structured_vocab-path",
        "--ontology-path",
        dest="ontology_path",
        type=str,
        default=str(DEFAULT_ONTOLOGY_PATH),
        help=f"Path to MIDAS OWL structured_vocab file (default: {DEFAULT_ONTOLOGY_PATH})",
    )
    parser.add_argument(
        "--validate-constrained-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Validate constrained JSON responses against schema "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--validation-schema-path",
        type=str,
        default=None,
        help=(
            "Optional JSON schema file path used for constrained response "
            f"validation (default fallback: {DEFAULT_VALIDATION_SCHEMA_PATH})"
        ),
    )
    parser.add_argument(
        "--openai-json-schema",
        action="store_true",
        default=False,
        help=(
            "For openai_compatible API type, attempt response_format=json_schema. "
            "Default is json_object mode."
        ),
    )

    args = parser.parse_args()

    config = ExtractionConfig.from_yaml()
    logger = configure_logging()

    if args.embedding_models:
        args.embedding_models = parse_embedding_models_arg(args.embedding_models)
    elif args.embedding_model:
        args.embedding_models = parse_embedding_models_arg(args.embedding_model)
    else:
        args.embedding_models = config.embedding_models

    # Handle --validate-thresholds
    if args.validate_thresholds:
        from concept_extractor.utils.evaluation.vector_similarity import run_domain_validation
        validation_model = args.embedding_models[0]
        logger.info("Running threshold validation with domain-specific test pairs...")
        logger.info("Threshold validation model: %s", validation_model)
        validation_results = run_domain_validation(model_name=validation_model, logger=logger)

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

    # Log resolved path-related options used for this run
    evaluation_dataset_path = Path(args.evaluation_dataset_path)
    output_root_path = Path(args.output_dir)
    logger.info("Resolved run paths/options:")
    logger.info("  repo_root: %s", REPO_ROOT)
    logger.info(
        "  evaluation_dataset_path: %s%s",
        evaluation_dataset_path,
        " (default)" if evaluation_dataset_path == DEFAULT_EVALUATION_DATASET_PATH
        else (" (legacy)" if evaluation_dataset_path == LEGACY_EVALUATION_DATASET_PATH else " (override)"),
    )
    logger.info(
        "  output_dir: %s%s",
        output_root_path,
        " (default)" if output_root_path == DEFAULT_GOLD_OUTPUT_DIR else " (override)",
    )
    if args.constrained:
        ontology_path = Path(args.ontology_path)
        logger.info(
            "  ontology_path: %s%s",
            ontology_path,
            " (default)" if ontology_path == DEFAULT_ONTOLOGY_PATH else " (override)",
        )
    if args.validate_constrained_json:
        if args.validation_schema_path:
            logger.info("  validation_schema_path: %s (override)", Path(args.validation_schema_path))
        else:
            logger.info("  validation_schema_path: <runtime structured_vocab schema>")

    # Load evaluation dataset
    try:
        all_abstracts = load_evaluation_dataset(args.evaluation_dataset_path)
    except FileNotFoundError:
        logger.error("Evaluation dataset not found: %s", args.evaluation_dataset_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Invalid evaluation dataset JSON at %s: %s", args.evaluation_dataset_path, e)
        return

    if args.list_papers:
        print("Available paper IDs:")
        for abstract in all_abstracts:
            print(f"  {abstract['id']}: {abstract['title'][:60]}...")
        return

    logger.info("Starting evaluation...")

    detected_host = autodetect_llm_host(
        config.active_llm_host,
        api_type=config.llm_api_type,
        logger=logger,
    )
    if detected_host:
        config.active_llm_host = detected_host

    use_llm_eval = not args.no_llm_eval
    use_vector_eval = not args.no_vector_eval

    if config.show_config:
        log_active_run_configuration(
            logger=logger,
            config=config,
            args=args,
            use_vector_eval=use_vector_eval,
            use_llm_eval=use_llm_eval,
        )

    # ── NEW: Load MIDAS structured_vocab for constrained decoding ──
    json_schema = None
    if args.constrained:
        try:
            from concept_extractor.utils.structured_vocab.midas_vocabulary import MIDASVocabulary

            owl_path = args.ontology_path or str(DEFAULT_ONTOLOGY_PATH)
            logger.info("Loading MIDAS structured_vocab from: %s", owl_path)
            vocab = MIDASVocabulary.from_owl(owl_path)
            json_schema = vocab.build_json_schema()

            logger.info(
                "MIDAS structured_vocab loaded: %d schema properties",
                len(json_schema.get("properties", {})),
            )
            logger.info("Constrained decoding: ENABLED")

            # Also append structured_vocab vocabulary to prompt
            # (helps even if constrained decoding falls back)
            vocab_text = vocab.build_prompt_section()
            logger.info("Ontology vocabulary section: %d chars", len(vocab_text))

        except FileNotFoundError:
            logger.error(
                "Ontology file not found: %s — run with --no-constrained "
                "or download with:\n"
                "  curl -o resources/ontologies/midas_data/midas-data.owl "
                "https://raw.githubusercontent.com/midas-network/midas-data/"
                "refs/heads/main/midas-data.owl",
                owl_path,
            )
            return
        except ImportError:
            logger.error(
                "midas_vocabulary module not found. Ensure "
                "midas_vocabulary.py is in src/concept_extractor/utils/structured_vocab/"
            )
            return
        except Exception as e:
            logger.error("Failed to load MIDAS structured_vocab: %s", e)
            return
    else:
        logger.error(
            "Free-text parsing is disabled. Run with --constrained "
            "(or remove --no-constrained)."
        )
        return

    validation_schema = None
    resolved_validation_schema_path = None
    if args.validate_constrained_json:
        if args.validation_schema_path:
            candidate = Path(args.validation_schema_path)
            if not candidate.exists():
                logger.error("Validation schema file not found: %s", candidate)
                return
            try:
                validation_schema = json.loads(candidate.read_text(encoding="utf-8"))
                resolved_validation_schema_path = str(candidate)
                logger.info(
                    "Constrained response validation enabled using schema file: %s",
                    candidate,
                )
            except json.JSONDecodeError as e:
                logger.error("Invalid schema JSON at %s: %s", candidate, e)
                return
        elif json_schema is not None:
            validation_schema = json_schema
            logger.info(
                "Constrained response validation enabled using runtime structured_vocab schema."
            )
        else:
            if DEFAULT_VALIDATION_SCHEMA_PATH.exists():
                try:
                    validation_schema = json.loads(
                        DEFAULT_VALIDATION_SCHEMA_PATH.read_text(encoding="utf-8")
                    )
                    resolved_validation_schema_path = str(DEFAULT_VALIDATION_SCHEMA_PATH)
                    logger.info(
                        "Constrained response validation enabled using schema file: %s",
                        DEFAULT_VALIDATION_SCHEMA_PATH,
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid schema JSON at %s: %s",
                        DEFAULT_VALIDATION_SCHEMA_PATH,
                        e,
                    )
                    return
            else:
                logger.error(
                    "Constrained response validation requested, but no schema found. "
                    "Set --validation-schema-path or ensure %s exists.",
                    DEFAULT_VALIDATION_SCHEMA_PATH,
                )
                return

    # Filter abstracts
    if args.paper_id:
        abstracts = [a for a in all_abstracts if a["id"] == args.paper_id]
        if not abstracts:
            logger.error("Paper ID '%s' not found. Use --list-papers.", args.paper_id)
            return
    elif args.num_papers == -1:
        abstracts = all_abstracts
    else:
        abstracts = all_abstracts[:args.num_papers]

    logger.info("Evaluating %d of %d dataset abstracts", len(abstracts), len(all_abstracts))

    if use_vector_eval:
        logger.info(
            "Vector similarity evaluation: ENABLED (models=%s)",
            ", ".join(args.embedding_models),
        )
        logger.info("  Auto-match threshold: >= %.2f", args.vector_high_threshold)
        logger.info("  Auto-reject threshold: <= %.2f", args.vector_low_threshold)
    else:
        logger.info("Vector similarity evaluation: DISABLED")

    if use_llm_eval:
        logger.info("LLM semantic matching: ENABLED")
    else:
        logger.info("LLM semantic matching: DISABLED")
    if config.llm_api_type == "openai_compatible":
        logger.info(
            "OpenAI response_format mode: %s",
            "json_schema" if args.openai_json_schema else "json_object",
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = results_model_directory_name(config)
    run_root = Path(args.output_dir) / model_dir
    run_folder = str(run_root / timestamp)
    os.makedirs(run_folder, exist_ok=True)
    logger.info("Run folder: %s", run_folder)

    # ── NEW: Save schema to run folder for reproducibility ──
    if json_schema is not None:
        schema_file = os.path.join(run_folder, "constrained_schema.json")
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(json_schema, f, indent=2)
        logger.info("Saved constrained schema to: %s", schema_file)

    results = run_evaluation(
        abstracts, config, logger,
        use_llm_eval=use_llm_eval,
        use_vector_eval=use_vector_eval,
        vector_high_threshold=args.vector_high_threshold,
        vector_low_threshold=args.vector_low_threshold,
        embedding_models=args.embedding_models,
        run_folder=run_folder,
        json_schema=json_schema,  # ── NEW: passed through
        openai_allow_json_schema=args.openai_json_schema,
        validate_constrained_json=args.validate_constrained_json,
        validation_schema=validation_schema,
    )

    results["evaluation_config"] = {
        "use_vector_eval": use_vector_eval,
        "use_llm_eval": use_llm_eval,
        "vector_high_threshold": args.vector_high_threshold,
        "vector_low_threshold": args.vector_low_threshold,
        "embedding_models": args.embedding_models,
        "embedding_model": args.embedding_models[0] if args.embedding_models else None,  # backward-compatible key
        "constrained": args.constrained,  # ── NEW
        "ontology_path": args.ontology_path,  # ── NEW
        "evaluation_dataset_path": args.evaluation_dataset_path,
        "output_dir": args.output_dir,
        "validate_constrained_json": args.validate_constrained_json,
        "openai_json_schema": args.openai_json_schema,
        "validation_schema_path": (
            args.validation_schema_path
            or resolved_validation_schema_path
            or ("<runtime_ontology_schema>" if validation_schema is json_schema and json_schema is not None else None)
        ),
    }

    output_file = os.path.join(run_folder, "evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    report_file = generate_evaluation_text_report(
        results, run_folder=run_folder, logger=logger,
    )
    logger.info("Human-readable evaluation report: %s", report_file)

    print_summary(results, logger)


if __name__ == "__main__":
    main()
