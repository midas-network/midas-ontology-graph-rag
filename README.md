# MIDAS-LLM

Evaluate LLM extraction quality against a gold-standard dataset of abstracts.

## Prerequisites

- Python 3.10-3.12
- A reachable LLM endpoint (OpenAI-compatible API or Ollama)

## Install

### macOS (Apple Silicon / non-Intel)

```bash
cd <project dir>

brew install python@3.12
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional embedding evaluation dependencies:

```bash
pip install -e ".[embeddings]"
```

Download the MIDAS ontology file (required for default constrained runs):

```bash
mkdir -p resources/ontologies/midas_data
curl -L -o resources/ontologies/midas_data/midas-data.owl \
  https://raw.githubusercontent.com/midas-network/midas-data/refs/heads/main/midas-data.owl
```

### macOS (Intel)

```bash
cd <project dir>

brew install python@3.12
export PATH="/usr/local/opt/python@3.12/bin:$PATH"

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -e .
pip install -e ".[embeddings]"
```

Download the MIDAS ontology file (required for default constrained runs):

```bash
mkdir -p resources/ontologies/midas_data
curl -L -o resources/ontologies/midas_data/midas-data.owl \
  https://raw.githubusercontent.com/midas-network/midas-data/refs/heads/main/midas-data.owl
```

## Configure

Config loading priority is:

1. Environment variables
2. YAML file
3. Code defaults

Default config file: `config.yaml` (repo root).

### Minimal config example

```yaml
llm_api_type: ollama
ollama_model: qwen2:7b-instruct
ollama_models: []
ollama_host: http://localhost:11434
```

### Configuration reference

The YAML keys map 1:1 to `ExtractionConfig` fields (env override is the uppercase form, for example `llm_api_type` -> `LLM_API_TYPE`).

| Key | What it controls |
|---|---|
| `prompt_include_format_examples` | Includes `format_examples.txt` in prompt assembly. |
| `prompt_include_reminders` | Includes `reminders.txt` in prompt assembly. |
| `prompt_include_few_shot` | Includes `few-shot.txt` in prompt assembly. |
| `prompt_include_fields` | Includes `fields.txt` in prompt assembly. |
| `prompt_include_ontologies` | Includes `ontologies.txt` in prompt assembly. |
| `prompt_simple_prompt` | Uses condensed inline prompt instead of file-based sections. |
| `ollama_model` | Single Ollama model fallback when `ollama_models` is empty. |
| `ollama_models` | Ollama model list for multi-model evaluation runs. |
| `ollama_host` | Ollama server base URL. |
| `nim_models` | OpenAI-compatible/NIM model list when `llm_api_type: openai_compatible`. |
| `nim_host` | OpenAI-compatible/NIM server base URL. |
| `llm_timeout` | HTTP timeout passed to LLM requests. |
| `llm_api_type` | Provider switch: `ollama` or `openai_compatible`. |
| `show_config` | Controls active config logging at run start. |
| `embedding_models` | Default embedding model list for vector similarity evaluation. |

Current distributed defaults in `config.yaml`:

```yaml
ontology_path: resources/ontologies

prompt_include_format_examples: false
prompt_include_reminders: true
prompt_include_few_shot: true
prompt_include_fields: true
prompt_include_ontologies: false
prompt_simple_prompt: false

ollama_model: qwen2:7b-instruct
ollama_models: []
ollama_host: http://localhost:11434

nim_models:
  - nvidia/qwen2:7b-instruct
nim_host: http://localhost:8000

llm_timeout: 300000
llm_api_type: ollama

show_config: true

embedding_models:
  - BAAI/bge-m3
  - FremyCompany/BioLORD-2023-C
```

### Start Ollama

```bash
ollama run qwen2:7b-instruct
```

## Run

After install, the CLI entrypoint is:

```bash
run_evaluation --help
```

Source workflow:

- `src/concept_extractor/workflows/run_evaluation.py`

Common commands:

```bash
# Evaluate default dataset (first 20 papers)
run_evaluation

# Evaluate 1 paper
run_evaluation --num-papers 1

# Evaluate all papers
run_evaluation --num-papers -1

# Evaluate one paper by ID
run_evaluation --paper-id <paper_id>

# List available paper IDs
run_evaluation --list-papers
```

## Output

Run artifacts are written to:

- `output/concept_extractor/results/<model>/<timestamp>/`

Typical files include:

- `evaluation.json` (structured results)
- `evaluation_report.txt` (human-readable summary)
- Per-abstract subfolders with model response/evaluation artifacts
