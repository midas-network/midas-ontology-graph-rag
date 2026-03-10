# Quickstart

This guide covers:
1. Installing the project
2. Starting Ollama
3. Downloading the default ontology file
4. Running evaluation with the default `config.yaml`

## Prerequisites

- macOS
- Homebrew
- Python 3.10-3.12 (recommended: 3.12)
- Ollama installed

Install Ollama if needed:

```bash
brew install --cask ollama
```

## 1) Install

From the repository root:

```bash
cd /path/to/midas-llm
```

Apple Silicon (`arm64`):

```bash
brew install python@3.12
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Intel Mac (`x86_64`):

```bash
brew install python@3.12
export PATH="/usr/local/opt/python@3.12/bin:$PATH"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional embedding dependencies (either architecture):

```bash
pip install -e ".[embeddings]"
```

## 2) Start Ollama

The default config uses:
- `llm_api_type: ollama`
- `ollama_host: http://localhost:11434`
- `ollama_model: qwen2:7b-instruct`

Start Ollama service (if not already running):

```bash
ollama serve
```

In a second terminal, pull the default model:

```bash
ollama pull qwen2:7b-instruct
```

## 3) Download Ontology

Download the MIDAS ontology file used by constrained decoding:

```bash
mkdir -p resources/ontologies/midas_data
curl -L -o resources/ontologies/midas_data/midas-data.owl \
  https://raw.githubusercontent.com/midas-network/midas-data/refs/heads/main/midas-data.owl
```

## 4) Run With Default Config

Back in your project terminal (with `.venv` active):

```bash
run_evaluation
```

This uses `config.yaml` in the repo root and evaluates the first 20 papers by default.

Useful variants:

```bash
run_evaluation --num-papers 1
run_evaluation --num-papers -1
run_evaluation --list-papers
```
