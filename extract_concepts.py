"""
Example of using hybrid retrieval with papers and ontology.
"""
import json
from langchain_core.vectorstores import InMemoryVectorStore

from model.graph_rag import initialize_embedding_model, setup_graph_retriever
from model.ontology import build_ontology_graph, create_ontology_documents, load_ontology
from model.hybrid_retrieval import (
    retrieve_hybrid,
    format_context_for_llm,
    prepare_llm_prompt,
    print_retrieval_summary
)
from utils.extract_paper_attributes import create_paper_documents, add_bidirectional_edges


def build_query(abstract_path: str) -> str:
    """
    Build a query that asks for extraction of paper attributes from an abstract.

    Args:
        abstract_path: Path to the text file containing the paper abstract

    Returns:
        Complete query string with instructions and abstract
    """
    query_template = """Given the paper abstract at the end of this prompt, report the status of each of these variables:
    model_type (e.g. "agent-based", "SEIR compartmental", "network", "metapopulation")
model_determinism (e.g. "deterministic", "stochastic", "both", "unspecified")
pathogen_name (e.g. "SARS-CoV-2", "Influenza A(H1N1)", "Plasmodium falciparum")
pathogen_type (e.g. "virus", "bacterium", "parasite", "fungus", "unspecified")
host_species (e.g. "human", "cattle", "wild birds", "mosquito vector", "multi-host")
primary_population (e.g. "general adult population", "school children", "healthcare workers", "men who have sex with men")
population_setting_type (e.g. "community", "hospital", "long-term care facility", "school", "refugee camp")
geographic_scope (e.g. "city-level", "sub-national", "national", "multi-country", "global")
geographic_units (e.g. "New York City, USA", "Sierra Leone", "European Union", "Wuhan, China")
historical_vs_hypothetical (e.g. "historical", "hypothetical future outbreak", "mixed", "unspecified")
study_goal_category (e.g. "forecast/nowcast", "scenario analysis", "intervention evaluation", "parameter estimation", "methodological")
intervention_present (e.g. "yes", "no", "unspecified")
intervention_types (e.g. ["vaccination", "school closure"], ["testing and isolation", "contact tracing"], ["vector control"])
data_used (e.g. ["surveillance case data"], ["hospital admissions", "mobility data"], ["no data (theoretical)"])
key_outcome_measures (e.g. ["incidence", "hospitalizations", "deaths"], ["R0", "attack rate"], ["bed occupancy", "ICU demand"])
code_available (e.g. "yes – GitHub", "no", "unspecified")"""

    with open(abstract_path, "r", encoding="utf-8") as f:
        abstract = f.read()

    return query_template + "\n\n" + abstract


def main():
    # Configuration
    ONTOLOGY_PATH = "midas-data.owl"
    PAPERS_PATH = "./data/modeling_papers.json"

    # 1. Load and prepare ontology
    print("Loading ontology...")
    g, classes, properties = load_ontology(ONTOLOGY_PATH)
    G = build_ontology_graph(g, classes, properties)
    ontology_docs, label_map = create_ontology_documents(G)

    # 2. Load and prepare papers WITH edges to ontology
    print("\nLoading papers and extracting ontology mentions...")
    with open(PAPERS_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
    paper_docs = create_paper_documents(papers, G)

    # 3. Create bidirectional edges between papers and ontology
    print("\nCreating bidirectional edges...")
    ontology_docs = add_bidirectional_edges(ontology_docs, paper_docs)

    print(f"\n✓ Loaded {len(paper_docs)} papers and {len(ontology_docs)} ontology concepts")
    print(f"✓ Papers linked to ontology via 'mentions' edges")
    print(f"✓ Ontology linked to papers via 'mentioned_by' edges")

    # 4. Create vector store
    print("\nCreating vector store and embeddings...")
    embedding_model = initialize_embedding_model()
    all_docs = ontology_docs + paper_docs

    vector_store = InMemoryVectorStore.from_documents(all_docs, embedding_model)

    # 5. Setup graph retriever with edge traversal enabled
    print("\nSetting up GraphRetriever with edge traversal...")
    graph_retriever = setup_graph_retriever(vector_store, include_papers=True)
    print("✓ GraphRetriever configured to traverse:")
    print("  - Paper -> mentions -> Ontology concepts")
    print("  - Ontology -> mentioned_by -> Papers")
    print("  - Ontology -> parents/children -> Related concepts")

    # 5. Query with hybrid retrieval
    query = build_query("data/fred-abstract.txt")
    print(f"\nQuery: {query}\n")


    paper_results, ontology_results, all_results = retrieve_hybrid(
        query=query,
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        k_papers=20,
        k_ontology=20
    )

    # 6. Print summary
    print_retrieval_summary(paper_results, ontology_results)

    # 7. Format context for LLM
    context = format_context_for_llm(all_results)

    # 8. Prepare complete prompt
    prompt = prepare_llm_prompt(query, context)

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="gpt-oss")

    print("\n" + "=" * 80)
    print("SENDING TO OLLAMA")
    print("=" * 80)
    response = llm.invoke(prompt)

    print("\n" + "=" * 80)
    print("LLM RESPONSE")
    print("=" * 80)
    print(response.content)


if __name__ == "__main__":
    main()
