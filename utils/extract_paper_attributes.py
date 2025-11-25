from langchain_core.documents import Document

PAPERS_FOR_RAG = "./data/modeling_papers.json"

def create_paper_documents(papers, G):
    """
    Create documents for each paper with metadata and edges to ontology concepts.

    Args:
        papers: List of paper dictionaries
        G: NetworkX graph containing ontology structure

    Returns:
        List of Document objects with 'mentions' edges to ontology concepts
    """
    import re

    # Build lookup structures for ontology matching
    label_to_id = {}  # label -> node_id
    synonym_to_id = {}  # synonym -> node_id

    for node_id, data in G.nodes(data=True):
        if data.get("type") in ("Class", "Property"):
            label = data.get("label", "").lower()
            if label:
                label_to_id[label] = node_id

            # Add synonyms
            for synonym in data.get("synonyms", []):
                synonym = synonym.lower()
                if synonym:
                    synonym_to_id[synonym] = node_id

    paper_docs = []
    total_mentions = 0

    for i, paper in enumerate(papers):
        # Try multiple key formats
        title = paper.get("title") or paper.get("paper_title", "")
        abstract = paper.get("abstract") or paper.get("paper_abstract", "")
        paper_id = paper.get("id") or paper.get("paper_id", f"paper_{i}")

        # Skip if no title and abstract
        if not title and not abstract:
            print(f"WARNING: Paper {i} has no title or abstract, skipping")
            continue

        # Handle abstract that might be a list
        if isinstance(abstract, list):
            abstract = " ".join(abstract)

        # Combine title and abstract for matching
        full_text = f"{title} {abstract}".lower()

        # Find ontology concepts mentioned in the paper
        mentioned_concepts = []

        # Check for exact label matches
        for label, node_id in label_to_id.items():
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{re.escape(label)}\b', full_text):
                if node_id not in mentioned_concepts:
                    mentioned_concepts.append(node_id)

        # Check for synonym matches
        for synonym, node_id in synonym_to_id.items():
            if re.search(rf'\b{re.escape(synonym)}\b', full_text):
                if node_id not in mentioned_concepts:
                    mentioned_concepts.append(node_id)

        total_mentions += len(mentioned_concepts)

        # Combine title and abstract for content
        content = f"Title: {title}\n\nAbstract: {abstract}"

        # Create document with metadata including graph edges
        doc = Document(
            page_content=content,
            metadata={
                "source": "paper",
                "title": title,
                "paper_id": paper_id,
                "type": "research_paper",
                "mentions": mentioned_concepts,  # Edge to ontology concepts
                "mentioned_by": []  # Placeholder (not used for papers)
            },
            id=f"paper_{paper_id}"
        )
        paper_docs.append(doc)

    print(f"  Extracted {total_mentions} total ontology mentions from {len(paper_docs)} papers")
    print(f"  Average: {total_mentions / len(paper_docs):.1f} mentions per paper")

    return paper_docs

def add_bidirectional_edges(ontology_docs, paper_docs):
    """
    Add 'mentioned_by' edges from ontology concepts to papers that mention them.

    Args:
        ontology_docs: List of ontology Document objects
        paper_docs: List of paper Document objects with 'mentions' edges

    Returns:
        Updated ontology_docs with 'mentioned_by' edges
    """
    # Build reverse index: concept_id -> list of paper_ids
    concept_to_papers = {}

    for paper in paper_docs:
        paper_id = paper.id
        for concept_id in paper.metadata.get('mentions', []):
            if concept_id not in concept_to_papers:
                concept_to_papers[concept_id] = []
            concept_to_papers[concept_id].append(paper_id)

    # Add mentioned_by edges to ontology documents
    for onto_doc in ontology_docs:
        concept_id = onto_doc.id
        onto_doc.metadata['mentioned_by'] = concept_to_papers.get(concept_id, [])

    # Print statistics
    mentioned_concepts = sum(1 for papers in concept_to_papers.values() if papers)
    total_edges = sum(len(papers) for papers in concept_to_papers.values())
    print(f"  Created {total_edges} bidirectional edges")
    print(f"  {mentioned_concepts} ontology concepts are mentioned in papers")

    return ontology_docs


