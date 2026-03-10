# MIDAS-LLM

Automated pipeline that uses LLMs to extract structured epidemiological metadata from research abstracts and evaluates extraction quality against a manually curated gold-standard dataset.

Given an abstract, the pipeline prompts an LLM to produce a structured JSON object with 22 schema fields (pathogen, disease, location, modeling technique, etc.) drawn from the [MIDAS controlled vocabulary](https://github.com/midas-network/midas-data). Extracted values are then scored against gold-standard annotations using a multi-model vector-similarity ensemble with LLM semantic fallback.

## Prerequisites

- macOS
- [Homebrew](https://brew.sh)
- Python 3.10–3.12 (recommended: 3.12)
- [Ollama](https://ollama.com) — install if needed:

```bash
brew install --cask ollama
```

Then open the Ollama app once — it runs as a background menu-bar service and starts the API server at `http://localhost:11434` automatically on login.

## Quick start

```bash
# Clone
git clone https://github.com/midas-network/midas-ontology-graph-rag.git midas-llm
cd midas-llm

# Install (see full Install section below for architecture-specific PATH notes)
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[embeddings]"

# Download the MIDAS ontology
mkdir -p resources/ontologies/midas_data
curl -L -o resources/ontologies/midas_data/midas-data.owl \
  https://raw.githubusercontent.com/midas-network/midas-data/refs/heads/main/midas-data.owl

# Pull the default model and run
ollama pull qwen2:7b-instruct
run_evaluation
```

## Usage

The CLI entrypoint after install is `run_evaluation`. The underlying workflow lives at `src/concept_extractor/workflows/run_evaluation.py`.

```bash
run_evaluation                       # Evaluate default dataset (first 20 abstracts)
run_evaluation --num-papers 1        # Single abstract
run_evaluation --num-papers -1       # All abstracts
run_evaluation --paper-id <id>       # Specific abstract by ID
run_evaluation --list-papers         # List available paper IDs
run_evaluation --no-constrained      # Skip ontology-constrained decoding
run_evaluation --help                # Full option list
```

## Output

Results are written to `output/concept_extractor/results/<model>/<timestamp>/` and include:

- `evaluation.json` — structured per-field scores
- `evaluation_report.txt` — human-readable summary with precision, recall, and F1
- Per-abstract subfolders with the raw LLM response and field-level evaluation artifacts

## Configuration

Config is resolved in priority order: **environment variables → `config.yaml` → code defaults**. The YAML keys map 1:1 to `ExtractionConfig` fields; the env var override is the uppercase form (e.g., `llm_api_type` → `LLM_API_TYPE`).

### LLM backend

| Key | Default | Description |
|-----|---------|-------------|
| `llm_api_type` | `ollama` | `ollama` or `openai_compatible` |
| `ollama_model` | `qwen2:7b-instruct` | Fallback model when `ollama_models` is empty |
| `ollama_models` | `[]` | Model list for multi-model evaluation runs |
| `ollama_host` | `http://localhost:11434` | Ollama server URL |
| `nim_models` | `["nvidia/qwen2:7b-instruct"]` | Model list for OpenAI-compatible/NIM backends |
| `nim_host` | `http://localhost:8000` | OpenAI-compatible server URL |
| `llm_timeout` | `300000` | LLM request timeout in milliseconds (300 s) |

### Prompt assembly

| Key | Default | Description |
|-----|---------|-------------|
| `prompt_include_fields` | `true` | Include `fields.txt` (schema field definitions) |
| `prompt_include_few_shot` | `true` | Include `few-shot.txt` (worked examples) |
| `prompt_include_reminders` | `true` | Include `reminders.txt` (extraction constraints) |
| `prompt_include_format_examples` | `false` | Include `format_examples.txt` |
| `prompt_include_ontologies` | `false` | Include `ontologies.txt` (large; slows inference) |
| `prompt_simple_prompt` | `false` | Use condensed inline prompt instead of file-based sections |

### Evaluation

| Key | Default | Description |
|-----|---------|-------------|
| `embedding_models` | `["BAAI/bge-m3", "FremyCompany/BioLORD-2023-C"]` | Embedding models for vector-similarity scoring |
| `ontology_path` | `resources/ontologies` | Path to MIDAS ontology files |
| `show_config` | `true` | Log active config at run start |

## Project structure

```
src/concept_extractor/
├── utils/
│   ├── config/              # YAML/env config loading
│   ├── evaluation/          # Vector-similarity scoring ensemble
│   ├── llm/                 # LLM client (Ollama + OpenAI-compatible)
│   ├── parsers/             # JSON extraction response parsing
│   ├── prompt/              # Prompt assembly and few-shot generation
│   ├── reporting/           # Evaluation report formatting
│   └── structured_vocab/    # MIDAS vocabulary and synonym handling
└── workflows/
    ├── evaluation/          # Evaluation engine, matching, parsing, reporting
    └── run_evaluation.py    # CLI entrypoint workflow

resources/
├── prompts/extract_concepts/  # Modular prompt sections
├── schemas/                   # MIDAS JSON schema
├── vocab/                     # Controlled vocabulary and synonyms
└── test_abstracts.json        # Gold-standard dataset
```