# midas-llm

Extract structured metadata from infectious disease modeling abstracts using LLMs.

## Overview

This tool extracts standardized attributes from scientific abstracts about infectious disease modeling studies. It uses large language models (via Ollama) to identify key metadata such as:

- Model type (agent-based, compartmental, etc.)
- Pathogen and disease information
- Geographic scope and population
- Intervention types
- Study outcomes and methods

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running with a compatible model (e.g., `qwen2.5:72b`)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/midas-llm.git
cd midas-llm
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install the package

**For development (editable install):**

```bash
pip install -e .
```

**For production:**

```bash
pip install .
```

**With optional embedding support (for similarity scoring):**

```bash
pip install -e ".[embeddings]"
```

**With development tools:**

```bash
pip install -e ".[dev]"
```

## Configuration

Configuration is handled via environment variables. Create a `.env` file or export them directly:

```bash
# Required: Ollama host and model
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5:72b"

# Optional: Paths (defaults shown)
export ONTOLOGY_PATH="data/ontologies"
export OUTPUT_DIR="data/output/runs"
export ABSTRACT_PATH="data/fred-abstract.txt"

# Optional: Feature flags
export ENABLE_ONTOLOGY_LINKING="false"
export ENABLE_MIDAS_ONTOLOGY="false"
export DEBUG="false"
export TEST_MODE="false"
```

## Usage

### Download Ontologies (optional)

Download epidemiology-relevant ontologies for concept linking:

```bash
midas-download-ontologies --output-dir ./resources/ontologies

# Skip large ontologies
midas-download-ontologies --skip ncbi_taxonomy mesh

# Download specific ontologies only
midas-download-ontologies --only apollo_sv ido doid
```

### Extract Concepts from Abstracts

```bash
# Using the CLI command (after pip install)
midas-extract

# Or run as a module
python -m midas_llm

# Or use the convenience script (for development)
python extract_concepts.py
```

### Output

Results are saved to `output/runs/` with timestamped filenames:
- `YYYYMMDD_HHMMSS.json` - Structured extraction results
- `YYYYMMDD_HHMMSS_extraction_report.html` - Human-readable report
- `YYYYMMDD_HHMMSS.txt` - Console output log

## Project Structure

```
midas-llm/
├── src/
│   └── midas_llm/
│       ├── __init__.py
│       ├── __main__.py
│       ├── extract_concepts.py      # Main extraction logic
│       ├── model/                   # Ontology graph models
│       ├── utils/                   # Utilities
│       │   ├── config.py            # Configuration
│       │   ├── llm/                 # LLM client
│       │   ├── loaders/             # Data loaders
│       │   ├── ontology_linker/     # Concept linking
│       │   ├── parsers/             # Response parsing
│       │   └── prompt/              # Prompt building
│       └── scripts/
│           └── download_ontologies.py
├── data/
│   ├── ontologies/                  # Downloaded ontologies (gitignored)
│   ├── prompt_text/                 # Prompt templates
│   └── output/                      # Run outputs (gitignored)
├── pyproject.toml
└── README.md
```

## Running on a Server/Cluster

### With Ollama on a GPU node

1. Start Ollama on the GPU node:
   ```bash
   ollama serve --host 0.0.0.0:11434
   ```

2. Pull the model:
   ```bash
   ollama pull qwen2.5:72b
   ```

3. Set the host in your environment:
   ```bash
   export OLLAMA_HOST="http://gpu-node:11434"
   ```

4. Run extraction:
   ```bash
   midas-extract
   ```

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=midas-extract
#SBATCH --output=midas-%j.out
#SBATCH --time=01:00:00

module load python/3.11

cd /path/to/midas-llm
source .venv/bin/activate

export OLLAMA_HOST="http://${SLURM_NODELIST}:11434"
export OLLAMA_MODEL="qwen2.5:72b"

midas-extract
```

## Development

### Running tests

```bash
pytest
```

### Linting and formatting

```bash
ruff check src/
ruff format src/
```

### Type checking

```bash
mypy src/
```

## License

MIT
