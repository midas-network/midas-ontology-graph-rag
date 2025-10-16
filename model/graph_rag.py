import re

from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_embedding_model():
    """Initialize the embedding model for semantic search."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def setup_graph_retriever(vector_store):
    """Configure the GraphRetriever for combined vector and graph search."""
    edges = [
        ("mentions", "$id"),                           # paper chunk -> mentioned ontology node
        ("parents", "$id"), ("children", "$id"),       # traverse class hierarchy
        ("equivalentClasses", "$id"),                  # traverse equivalent class links
        ("domainOf", "$id"), ("rangeOf", "$id"),       # class -> property
        ("domain", "$id"), ("range", "$id")            # property -> class
    ]

    return GraphRetriever(
        store=vector_store,
        edges=edges,
        strategy=Eager(k=10, start_k=3, max_depth=3)
    )

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