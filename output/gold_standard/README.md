# Gold Standard Evaluation

This directory contains gold standard test abstracts and evaluation tools for measuring LLM extraction accuracy.

## Files

- `test_abstracts.json` - Gold standard abstracts with expected extraction values
- `results/` - Evaluation results from test runs

## Running Evaluation

```bash
# From repo root
python scripts/evaluate_gold_standard.py
```

## Configuration

The evaluation uses the same configuration as the main extraction:
- Set `OLLAMA_MODELS` in `ollama.cfg` to test multiple models
- Results are saved to `output/gold_standard/results/`

## Metrics

The evaluation calculates:
- **Recall**: % of expected values that were correctly extracted
- **Precision**: % of extracted values that match expected values
- **F1 Score**: Harmonic mean of precision and recall
- **Hits**: Correctly extracted values
- **Misses**: Expected values not extracted
- **False Positives**: Extracted values not in expected set

## Adding New Test Abstracts

Edit `test_abstracts.json` to add new abstracts:

```json
{
  "id": "unique_id",
  "title": "Paper Title",
  "abstract": "Full abstract text...",
  "expected": {
    "pathogen_name": ["value1", "value2"],
    "model_type": ["agent-based"],
    ...
  }
}
```

Expected values are lists - the extractor can match any value in the list.

## Expected Fields

Common fields to test:
- `model_type`
- `model_determinism`
- `pathogen_name`
- `pathogen_type`
- `disease_name`
- `host_species`
- `primary_population`
- `population_setting_type`
- `geographic_scope`
- `geographic_units`
- `historical_vs_hypothetical`
- `study_goal_category`
- `intervention_present`
- `intervention_types`
- `data_used`
- `data_source`
- `key_outcome_measures`
- `code_available`
- `study_dates_start`
- `study_dates_end`
- `calibration_mentioned`
- `calibration_techniques`
