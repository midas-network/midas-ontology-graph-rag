# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIDAS-LLM extracts structured metadata from infectious disease modeling abstracts using LLMs. It identifies attributes like model type, pathogen/disease info, geographic scope, interventions, and study outcomes, with optional ontology linking to standardized identifiers.

## Common Commands

### Installation

```bash
# Editable install (for development)
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With embeddings support
pip install -e ".[embeddings]"
```

### Running Extraction

```bash
# Main extraction CLI
midas-extract

# Or as module
python -m midas_llm

# Show prompt only (no LLM call)
midas-extract --prompt-only
```

### Downloading Ontologies

```bash
# Download all ontologies
midas-download-ontologies --output-dir ./resources/ontologies

# Skip large ontologies
midas-download-ontologies --skip ncbi_taxonomy mesh

# Download specific ontologies only
midas-download-ontologies --only apollo_sv ido doid
```

### Development

```bash
# Run tests
pytest

# Linting
ruff check src/
ruff format src/

# Type checking
mypy src/

# Run gold standard evaluation
python -m midas_llm.scripts.evaluate_gold_standard
```

## Architecture

### Main Entry Points

- `src/midas_llm/extract_concepts.py` - Main extraction workflow (`main()` function)
- `src/midas_llm/scripts/` - Standalone utility scripts
  - `download_ontologies.py` - Downloads biomedical ontologies
  - `evaluate_gold_standard.py` - Evaluates extraction accuracy against gold standard
  - `send_fulltext_to_llm.py` - Sends full texts to LLM for processing

### Core Modules

- `utils/config.py` - `ExtractionConfig` dataclass with env var overrides (see below)
- `utils/llm/llm_client.py` - LLM API client (Ollama and OpenAI-compatible)
- `utils/parsers/extraction_parser.py` - Parses LLM responses into structured attributes
- `utils/ontology_linker/` - Concept to ontology identifier mapping
- `utils/loaders/ontology_loader.py` - Loads ontologies with parallel processing
- `utils/prompt/builders.py` - Builds extraction prompts
- `utils/reporting/` - Generates HTML/text reports and evaluation summaries
- `model/` - Graph RAG, hybrid retrieval, and ontology models

### Extraction Workflow

1. Load configuration from env vars + `ollama.cfg`
2. Probe/autodetect LLM host, run sanity check if enabled
3. Optionally load MIDAS ontology
4. Build prompt from abstract + examples/field definitions
5. Send to LLM(s) - supports multiple models for comparison
6. Parse LLM response using regex patterns in `extraction_parser.py`
7. If enabled, link extracted concepts to ontologies (parallel background loading)
8. Generate reports (HTML/JSON/evaluation)

### Ontology Linking

Background ontology loading starts before LLM request to improve performance. Supports:

| Attribute Type | Ontology | Identifier Format |
|---------------|----------|------------------|
| `host_species`, `pathogen_name` | NCBI Taxonomy | `NCBI:txid{id}` |
| `geographic_units`, `geographic_scope` | ISO 3166 / GeoNames | `ISO:{code}` |
| `disease_name`, `pathogen_type` | Disease Ontology (DOID) | `DOID:{id}` |
| `model_type`, `intervention_types` | MIDAS/Apollo | `APOLLO_SV:{id}` |

## Configuration

### Primary Config File

`ollama.cfg` (searched in current dir, repo root, or `~/.ollama.cfg`):

```
# Ollama - single model or multiple for comparison
OLLAMA_MODEL=mistral:7b
OLLAMA_MODELS=mistral:7b,qwen2.5:7b,llama3.1:8b

# OpenAI-compatible / NIM - comma-separated list (defaults to 2 models for sensitivity analysis)
NIM_MODELS=meta/llama-3.1-8b-instruct,meta/llama-3.1-70b-instruct

OLLAMA_HOST=http://gpu-n28:53475
```

### Environment Variables

**LLM Settings:**
- `OLLAMA_MODEL` / `OLLAMA_MODELS` - Model(s) to use (Ollama)
- `OLLAMA_HOST` - LLM server URL (Ollama)
- `NIM_MODELS` - Models to use (OpenAI-compatible), defaults to 2 models for sensitivity analysis: `meta/llama-3.1-8b-instruct,meta/llama-3.1-70b-instruct`
- `NIM_HOST` - LLM server URL (OpenAI-compatible)
- `LLM_TIMEOUT` - Request timeout in ms (default: 300000)
- `LLM_API_TYPE` - "ollama" or "openai_compatible" (default: "openai_compatible")

**Paths:**
- `ONTOLOGY_PATH` - Ontology files directory (default: "resources/ontologies")
- `OUTPUT_DIR` - Output directory (default: "output/extract_concepts/runs")
- `PAPERS_PATH` - Papers JSON path
- `ABSTRACT_PATH` - Abstract text path

**Feature Flags:**
- `ENABLE_ONTOLOGY_LINKING` - Enable concept-to-ontology mapping
- `ENABLE_MIDAS_ONTOLOGY` - Load MIDAS ontology
- `RUN_SANITY_CHECK` - Pre-flight LLM connectivity test (default: true)
- `TEST_MODE` - LLM ping only, no extraction
- `DEBUG` - Enable debug logging

**Prompt Options:**
- `INCLUDE_FORMAT_EXAMPLES` - Include format examples in prompt
- `INCLUDE_REMINDERS` - Include reminder instructions (default: true)
- `INCLUDE_FEW_SHOT` - Include few-shot examples (default: true)
- `INCLUDE_FIELDS` - Include field definitions (default: true)
- `INCLUDE_ONTOLOGIES` - Include ontology information
- `SIMPLE_PROMPT` - Use simplified prompt
- `NUM_EXAMPLES` - Number of few-shot examples (default: 2)

**Output Options:**
- `SHOW_CONFIG` - Log configuration on startup (default: true)
- `GENERATE_HTML_REPORT` - Generate HTML report
- `GENERATE_JSON_OUTPUT` - Generate JSON output
- `GENERATE_EVALUATION_REPORT` - Generate evaluation report (default: true)

### Config Access

```python
from midas_llm.utils.config import ExtractionConfig

config = ExtractionConfig()
print(config.active_llm_model)  # Gets first model based on LLM_API_TYPE
print(config.active_llm_models)  # Gets list of all models for sensitivity analysis
print(config.active_llm_host)
```

## Code Style Guidelines

From `INSTRUCTIONS_TO_LLM.TXT` and `.ai/md/PROJECT_INSTRUCTIONS.md`:

1. **No emojis in output** - Use plain text: `✓`, `WARNING:`, `ERROR:`
2. **Python 3.10+ syntax** - Use `X | Y` union syntax, not `Union[X, Y]`
3. **Type hints mandatory** - All public interfaces must be typed
4. **Single responsibility** - Functions should do one thing well
5. **Modern syntax**:
   ```python
   # Correct
   def fetch(url: str) -> dict | None: ...
   def process(items: list[str]) -> list[str]: ...

   # Avoid
   from typing import Union, List, Optional
   def fetch(url: str) -> Optional[dict]: ...
   ```
6. **Docstrings** - Google style for public API
7. **Dataclasses** - Use for data containers, prefer frozen when possible
8. **Protocols** - Prefer structural typing over ABCs when appropriate

## Known Schema Fields

The parser recognizes these fields (`utils/parsers/extraction_parser.py`):

- Model: `model_type`, `model_determinism`
- Pathogen/Disease: `pathogen_name`, `pathogen_type`, `disease_name`
- Population: `host_species`, `primary_population`, `population_setting_type`
- Geography/Time: `geographic_scope`, `geographic_units`, `study_dates_start`, `study_dates_end`, `historical_vs_hypothetical`
- Study: `study_goal_category`, `intervention_present`, `intervention_types`
- Data: `data_used`, `data_source`, `key_outcome_measures`
- Methods: `calibration_mentioned`, `calibration_techniques`
- Other: `code_available`, `extraction_notes`

## Gold Standard Evaluation

Edit `resources/test_abstracts.json` to add test abstracts with expected values:

```json
{
  "id": "unique_id",
  "title": "Paper Title",
  "abstract": "Full abstract text...",
  "expected": {
    "pathogen_name": ["value1", "value2"],
    "model_type": ["agent-based"]
  }
}
```

Results saved to `results/{timestamp}/` with:
- `evaluation_config.json` - Evaluation configuration (saved immediately at start)
- `{abstract_id}/` - Per-abstract directory with:
  - `{model_name}_response.txt` - Raw LLM response text
  - `{model_name}_response.json` - Response validated against `resources/response_schema.json` with:
    - `content` - Raw LLM response text
    - `model` - Model name used
    - `done` - Whether response is complete
    - `timestamp` - ISO 8601 timestamp
    - `abstract_id` - Abstract identifier
    - `extracted` - Extracted and parsed data from LLM response
    - `evaluation` - Full evaluation results (hits, misses, false_positives, scores, vector_stats)
    - `raw_response` - First 2000 characters of raw LLM response
- `evaluation.json` - Full evaluation results (updated incrementally after each abstract)
- `evaluation.txt` - Human-readable evaluation report
- `evaluation.html` - HTML evaluation report
- Recall, Precision, F1 scores
- Hits, misses, false positives
- LLM semantic matching for values

## Important Files

- `resources/prompts/extract_concepts/` - Prompt template components
- `resources/ontologies/` - Downloaded ontology files (gitignored)
- `output/` - Generated outputs (gitignored)
- `.ai/md/` - Development notes and feature documentation
