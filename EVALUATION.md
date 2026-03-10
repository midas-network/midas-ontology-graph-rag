# Evaluation Workflow (High-Level)

This evaluation pipeline measures how well model-generated extraction output matches a reference dataset.

It is designed to be easy to run from one CLI entry point, while keeping parsing, matching, orchestration, and reporting separated into focused modules.

## End-to-End Steps

1. Start from the CLI
   - The run begins in `run_evaluation.py`, inside `main()`.

2. Resolve config, inputs, and run options
   - The CLI resolves model settings, dataset path, output folder, and constrained-output options.
   - It also decides which abstracts to evaluate.

3. Load evaluation data
   - The dataset loader returns the abstracts and expected values used as evaluation targets.

4. Execute model evaluation loop
   - The engine loops through abstracts and models, builds prompts, sends requests, parses responses, and evaluates matches.

5. Parse structured model output
   - Responses are normalized into a consistent internal structure before scoring.

6. Score extracted values against expected values
   - Per-field scoring is handled in one place, using layered matching.

7. Apply layered matching (conceptually)
   - The system tries semantic/vector-style matching first, then uses fallback methods when needed.

8. Aggregate metrics and write outputs
   - The run records hits, misses, false positives, and summary metrics.
   - It writes structured JSON and human-readable reports.

## Package Map

- `evaluation/parsing.py`
  - Dataset loading and structured response parsing/validation.
- `evaluation/matching.py`
  - Matching strategies (vector/semantic/fallback).
- `evaluation/engine.py`
  - Main runtime orchestration and per-abstract scoring.
- `evaluation/reporting.py`
  - Run-level summaries and reporting helpers.
- `evaluation/__init__.py`
  - Public re-exports used by the CLI.

## Methods of Interest

run_evaluation.py:
- `main()`

evaluation/engine.py:
- `run_evaluation()`
- `evaluate_extraction()`

evaluation/parsing.py:
- `load_evaluation_dataset()`
- `parse_constrained_response()`
- `validate_constrained_payload()`
- `get_expected_values()`

evaluation/matching.py:
- `vector_match_any_model()`
- `evaluate_semantic_match()`
- `fallback_string_match()`

evaluation/reporting.py:
- `print_summary()`
- `log_active_run_configuration()`
