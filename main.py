import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore

from model.graph_rag import identify_matched_concepts, initialize_embedding_model, setup_graph_retriever
from model.text_processing import create_chunk_documents, split_paper_into_chunks
from model.ontology import load_ontology, build_ontology_graph, create_ontology_documents, calculate_ontology_depth
from utils.midas_api import get_paper_data
from model.scoring import calculate_confidence_scores, calculate_relevance_scores, calculate_terminology_scores
from model.reporting import analyze_chunk_similarities, print_top_matches, print_reasoned_paths, \
    print_gaps_and_next_steps, generate_verdict


def setup_environment():
    """Load environment variables and initialize API key."""
    load_dotenv()
    return os.getenv("MIDAS_API_KEY")


def main():
    # Setup environment
    api_key = setup_environment()
    paper_id = 1000  # Default paper ID, find paperIDs in data/paper_ids.tsv
    ontology_path = "midas-data.owl"

    # Get paper data
    paper_data = get_paper_data(paper_id, api_key)

    # Load and process ontology
    g, classes, properties = load_ontology(ontology_path)
    G = build_ontology_graph(g, classes, properties)

    # Create documents for ontology elements
    ontology_docs, label_map = create_ontology_documents(G)

    # Process paper abstract into chunks
    chunks = split_paper_into_chunks(paper_data['paper_abstract'])
    paper_chunk_docs = create_chunk_documents(chunks, G)

    # Combine all documents
    all_docs = ontology_docs + paper_chunk_docs

    # Initialize embedding model
    embedding_model = initialize_embedding_model()

    # Analyze chunk similarities
    analyze_chunk_similarities(all_docs, embedding_model)

    # Create vector store and retriever
    vector_store = InMemoryVectorStore.from_documents(documents=all_docs, embedding=embedding_model)
    retriever = setup_graph_retriever(vector_store)

    # Execute retrieval
    query_text = " ".join(paper_data['paper_keywords'] + paper_data['paper_meshterms'])
    results = retriever.invoke(query_text)

    # Identify matched concepts
    direct_label_hits, synonym_hits = identify_matched_concepts(results, all_docs)

    # Get all matched concepts
    retrieved_concepts = {doc.id for doc in results if doc.metadata.get("type") in ("Class", "Property")}
    all_matched_concepts = retrieved_concepts.union(direct_label_hits).union(synonym_hits)

    # Calculate confidence scores
    confidence_scores = calculate_confidence_scores(all_matched_concepts, query_text, all_docs, embedding_model, label_map)

    # Filter matched classes
    matched_classes = [cid for cid in all_matched_concepts if G.nodes[cid]["type"] == "Class"]

    # Calculate ontology depth
    depth_map, roots = calculate_ontology_depth(G, label_map)

    # Calculate relevance scores
    coverage_score, hierarchy_score, alignment_score, coherence_score, visited = calculate_relevance_scores(
        G, classes, matched_classes, depth_map
    )

    # Calculate terminology scores
    consistency_score, evidence_score, consistent_count = calculate_terminology_scores(
        matched_classes, direct_label_hits, synonym_hits
    )


    total_score = sum([coverage_score, hierarchy_score, alignment_score,
                       coherence_score, consistency_score, evidence_score])
    total_score = min(total_score, 100.0)
    verdict = generate_verdict(total_score)

    print(f"**Executive Verdict:** **{verdict}** (Score: {total_score:.1f}/100)")

    top_concepts = sorted(all_matched_concepts, key=lambda cid: confidence_scores.get(cid, 0), reverse=True)

    print_top_matches(top_concepts, G, label_map, confidence_scores, direct_label_hits, synonym_hits)
    print_reasoned_paths(G, matched_classes, label_map, alignment_score)
    print_gaps_and_next_steps(G, roots, matched_classes, visited, consistent_count, verdict, label_map)

if __name__ == "__main__":
    main()