import re

from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_embedding_model():
    """Initialize the embedding model for semantic search."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def setup_graph_retriever(vector_store, include_papers=False):
    """
    Configure the GraphRetriever for combined vector and graph search.

    With include_papers=True, the retriever will:
    - Start with diverse initial results (papers and ontology)
    - Expand via graph edges to find related content
    - Paper -> mentions -> Ontology concepts
    - Ontology -> mentioned_by -> Papers
    - Ontology -> parents/children -> Related concepts

    Args:
        vector_store: InMemoryVectorStore with documents
        include_papers: If True, optimizes for retrieving both papers and ontology

    Returns:
        GraphRetriever configured with appropriate edges and strategy
    """
    edges = [
        ("mentions", "$id"),                           # paper -> mentioned ontology concepts
        ("mentioned_by", "$id"),                       # ontology -> papers that mention it
        ("parents", "$id"), ("children", "$id"),       # traverse class hierarchy
        ("equivalentClasses", "$id"),                  # traverse equivalent class links
        ("domainOf", "$id"), ("rangeOf", "$id"),       # class -> property
        ("domain", "$id"), ("range", "$id")            # property -> class
    ]

    # Configure for better paper+ontology retrieval
    # - Higher start_k gets more diverse initial results
    # - Higher k returns more total results after expansion
    # - max_depth=2 allows: paper -> concept -> related concept
    if include_papers:
        start_k = 15  # More diverse starting points
        k = 20  # Return more results total
        max_depth = 2  # Traverse 2 hops
    else:
        start_k = 3
        k = 10
        max_depth = 3

    return GraphRetriever(
        store=vector_store,
        edges=edges,
        strategy=Eager(k=k, start_k=start_k, max_depth=max_depth)
    )

def setup_hybrid_retriever(vector_store, k_papers=5, k_ontology=5):
    """
    Create a hybrid retriever that combines paper and ontology results.
    Returns both papers and ontology concepts with graph expansion.
    """
    # Base vector retriever for papers
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": k_papers + k_ontology}
    )

    return base_retriever  # For now, return base retriever - can enhance with ensemble later

def identify_matched_concepts(results, docs):
    """Identify direct label hits and synonym hits from results."""
    direct_label_hits = set()
    synonym_hits = set()

    for doc in results:
        if doc.metadata.get("type") in ("Class", "Property"):
            cid = doc.id
            # If this concept was mentioned in text (from our earlier mention detection)
            for chunk_doc in docs:
                if chunk_doc.metadata.get("type") == "PaperChunk" and cid in chunk_doc.metadata.get("mentions", []):
                    text = chunk_doc.page_content.lower()
                    label = doc.metadata.get("label", "").lower()
                    if label and re.search(rf"\b{re.escape(label)}\b", text):
                        direct_label_hits.add(cid)
                    else:
                        synonym_hits.add(cid)
                    break

    return direct_label_hits, synonym_hits