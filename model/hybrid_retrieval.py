"""
Hybrid retrieval utilities for combining paper and ontology document retrieval.
"""

def retrieve_hybrid(query, vector_store, graph_retriever, k_papers=5, k_ontology=5, retrieve_k=50):
    """
    Retrieve both papers and ontology concepts for a query using GraphRetriever.

    The GraphRetriever will:
    1. Start with top-k similar documents (papers and/or ontology)
    2. Expand via graph edges:
       - From papers -> follow 'mentions' -> reach ontology concepts
       - From ontology -> follow 'mentioned_by' -> reach papers
       - From ontology -> follow 'parents'/'children' -> reach related concepts
    3. Return diverse set of papers + ontology concepts

    Args:
        query: The search query string
        vector_store: The InMemoryVectorStore containing all documents
        graph_retriever: The GraphRetriever configured with edge traversal
        k_papers: Number of papers to return
        k_ontology: Number of ontology concepts to return
        retrieve_k: Not used with graph retriever (kept for API compatibility)

    Returns:
        tuple: (paper_docs, ontology_docs, all_docs)
    """
    # Use GraphRetriever to get documents with graph expansion
    print(f"\n[DEBUG retrieve_hybrid] Using GraphRetriever with edge traversal")
    all_results = graph_retriever.invoke(query)

    print(f"[DEBUG retrieve_hybrid] Graph retriever returned {len(all_results)} documents")

    # Separate papers and ontology from the result set
    paper_results = [doc for doc in all_results
                     if doc.metadata.get('source') == 'paper'][:k_papers]
    
    ontology_results = [doc for doc in all_results
                        if doc.metadata.get('source') == 'ontology'][:k_ontology]
    
    # DEBUG: Check what we got
    source_breakdown = {}
    for doc in all_results:
        source = doc.metadata.get('source', 'MISSING')
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
    print(f"[DEBUG retrieve_hybrid] Source breakdown: {source_breakdown}")
    print(f"[DEBUG retrieve_hybrid] After filtering: {len(paper_results)} papers, {len(ontology_results)} ontology docs")

    # Show how papers and ontology are connected
    if paper_results and ontology_results:
        sample_paper = paper_results[0]
        mentions = sample_paper.metadata.get('mentions', [])
        print(f"[DEBUG retrieve_hybrid] Sample paper mentions {len(mentions)} concepts")

        # Check if retrieved ontology concepts are in the mentions
        retrieved_concept_ids = {doc.id for doc in ontology_results}
        connected = len(set(mentions) & retrieved_concept_ids)
        print(f"[DEBUG retrieve_hybrid] {connected} retrieved concepts are mentioned by sample paper")

    # Combine results
    all_docs = paper_results + ontology_results

    return paper_results, ontology_results, all_docs


def format_context_for_llm(retrieved_docs, include_metadata=False):
    """
    Format retrieved documents into a context string for LLM input.

    Args:
        retrieved_docs: List of Document objects
        include_metadata: Whether to include additional metadata in context

    Returns:
        str: Formatted context string
    """
    context_parts = []

    # Add papers
    papers = [d for d in retrieved_docs if d.metadata.get('source') == 'paper']
    for i, doc in enumerate(papers, 1):
        title = doc.metadata.get('title', 'Untitled')

        if include_metadata:
            paper_id = doc.metadata.get('paper_id', 'unknown')
            section = f"[Paper {i}] (ID: {paper_id})\nTitle: {title}\n\n{doc.page_content}"
        else:
            section = f"[Paper {i}]\nTitle: {title}\n\n{doc.page_content}"

        context_parts.append(section)

    # Add ontology concepts
    concepts = [d for d in retrieved_docs if d.metadata.get('source') == 'ontology']
    if concepts:
        context_parts.append("\n--- Relevant Ontology Concepts ---")

        for i, doc in enumerate(concepts, 1):
            label = doc.metadata.get('label', 'Unknown')

            if include_metadata:
                concept_type = doc.metadata.get('type', 'Concept')
                section = f"[{concept_type} {i}] {label}\n{doc.page_content}"
            else:
                section = f"[Concept {i}] {label}\n{doc.page_content}"

            context_parts.append(section)

    return "\n\n".join(context_parts)


def prepare_llm_prompt(query, context, system_instructions=None):
    """
    Prepare a complete prompt for the LLM.

    Args:
        query: The user's question
        context: The retrieved context string
        system_instructions: Optional system instructions for the LLM

    Returns:
        str: Complete formatted prompt
    """
    if system_instructions is None:
        system_instructions = (
            "You are a helpful research assistant. Use the provided papers and "
            "ontology concepts to answer questions accurately. Cite specific papers "
            "when referencing information from them."
        )

    prompt = f"""{system_instructions}

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def print_retrieval_summary(paper_docs, ontology_docs):
    """
    Print a summary of what was retrieved.

    Args:
        paper_docs: List of paper documents
        ontology_docs: List of ontology documents
    """
    print(f"\n=== RETRIEVAL SUMMARY ===")
    print(f"Papers retrieved: {len(paper_docs)}")
    print(f"Ontology concepts retrieved: {len(ontology_docs)}")

    if paper_docs:
        print("\nPapers:")
        for i, doc in enumerate(paper_docs, 1):
            title = doc.metadata.get('title', 'Untitled')
            print(f"  {i}. {title[:80]}{'...' if len(title) > 80 else ''}")

    if ontology_docs:
        print("\nOntology Concepts:")
        for i, doc in enumerate(ontology_docs, 1):
            label = doc.metadata.get('label', 'Unknown')
            concept_type = doc.metadata.get('type', 'Concept')
            print(f"  {i}. [{concept_type}] {label}")

