# MIDAS-LLM

Evaluate LLM extraction quality against a gold-standard dataset of abstracts.

## Prerequisites

- Python 3.10+
- A reachable LLM endpoint (OpenAI-compatible API or Ollama)

## Install

```bash
git clone <your-repo-url>
cd midas-llm

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -e .
pip install -e ".[embeddings]"
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

| Key | What it controls | Used by runtime? |
|---|---|---|
| `prompt_include_format_examples` | Includes `format_examples.txt` in prompt assembly. | Yes. |
| `prompt_include_reminders` | Includes `reminders.txt` in prompt assembly. | Yes. |
| `prompt_include_few_shot` | Includes `few-shot.txt` in prompt assembly. | Yes. |
| `prompt_include_fields` | Includes `fields.txt` in prompt assembly. | Yes. |
| `prompt_include_ontologies` | Includes `ontologies.txt` in prompt assembly. | Yes. |
| `prompt_simple_prompt` | Uses condensed inline prompt instead of file-based sections. | Yes. |
| `ollama_model` | Single Ollama model fallback when `ollama_models` is empty. | Yes. |
| `ollama_models` | Ollama model list for multi-model evaluation runs. | Yes. |
| `ollama_host` | Ollama server base URL. | Yes. |
| `nim_models` | OpenAI-compatible/NIM model list when `llm_api_type: openai_compatible`. | Yes. |
| `nim_host` | OpenAI-compatible/NIM server base URL. | Yes. |
| `llm_timeout` | HTTP timeout passed to LLM requests. | Yes. |
| `llm_api_type` | Provider switch: `ollama` or `openai_compatible`. | Yes. |
| `show_config` | Controls active config logging at run start. | Yes. |
| `embedding_models` | Default embedding model list for vector similarity evaluation. | Yes. |

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
