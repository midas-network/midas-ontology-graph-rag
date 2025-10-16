import re

from langchain_core.documents import Document


def split_paper_into_chunks(paper_abstract):
    """Split paper abstract into chunks for processing."""
    raw_chunks = re.split(r"\n\s*\n", paper_abstract)
    chunks = []
    max_words = 200

    for chunk in raw_chunks:
        words = chunk.split()
        if len(words) > max_words:
            # Further split long paragraphs into roughly equal halves if needed
            mid = len(words) // 2
            chunks.append(" ".join(words[:mid]))
            chunks.append(" ".join(words[mid:]))
        else:
            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks

def create_chunk_documents(chunks, G):
    """Create document objects from paper chunks with ontology mentions."""
    docs = []
    for i, chunk_text in enumerate(chunks):
        # Find ontology mentions in this chunk (case-insensitive exact matches of labels or synonyms)
        mentions = []
        text_lower = chunk_text.lower()

        for node_id, data in G.nodes(data=True):
            if data["type"] == "Class" or data["type"] == "Property":
                # check label
                label = data["label"]
                if label and re.search(rf"\b{re.escape(label.lower())}\b", text_lower):
                    mentions.append(node_id)
                    continue
                # check synonyms (for classes)
                for syn in data.get("synonyms", []):
                    if re.search(rf"\b{re.escape(syn.lower())}\b", text_lower):
                        mentions.append(node_id)
                        break

        mentions = list(set(mentions))
        metadata = {"type": "PaperChunk", "mentions": mentions}
        docs.append(Document(page_content=chunk_text, metadata=metadata, id=f"paper_chunk_{i}"))

    return docs
