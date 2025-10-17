## MIDAS Ontology Graph RAG

A Python project that analyzes scientific papers using ontology-based retrieval, reasoning, and scoring. It leverages graph-based approaches and language models to assess paper relevance against MIDAS ontology concepts.

### Features:

* Parses and processes OWL ontology files containing infectious disease concepts
* Retrieves information on scientific papers via the MIDAS API
* Uses ontology-based graph reasoning for conceptual analysis
* Scores papers based on relevance to ontology terms
* Generates detailed assessment reports

## Setup Python

1. pip install -r requirements.txt

## Setup MIDAS API key

1. Login to the MIDAS Members Database.  Your MIDAS API Key is displayed on the main page.
1. Create a file in the root of the project called .env
1. In .env, insert the line: MIDAS_API_KEY="your_api_key_here"

## Run

1. Look in data/paper_ids.txt to find a paper ID that you'd like to run thourgh the Graph RAG.

2. Set the paper_id variable at the top of the main function in main.py

3. python main.py

