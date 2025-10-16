import numpy as np


def generate_verdict(total_score):
    """Generate verdict based on total relevance score."""
    if total_score >= 70:
        return "Relevant"
    elif total_score >= 30:
        return "Partially Relevant"
    else:
        return "Not Relevant"

def print_top_matches(top_concepts, G, label_map, confidence_scores, direct_label_hits, synonym_hits):
    """Print top matched ontology terms."""
    top_n = min(len(top_concepts), 10)
    print("\n**Top Matched Terms:**")
    print("| Paper Term | Ontology IRI | Ontology Label | Match Type | Confidence | Ontology Path |")
    print("|---|---|---|---|---|---|")

    for cid in top_concepts[:top_n]:
        if cid not in label_map or G.nodes[cid]["type"] != "Class":
            continue

        label = label_map[cid]
        # Determine match type text
        if cid in direct_label_hits:
            mtype = "label"
            term = label
        elif cid in synonym_hits:
            mtype = "synonym"
            term = G.nodes[cid].get("synonyms", [label])[0]
        else:
            mtype = "embedding"
            term = label

        # Confidence as percentage
        conf_percent = confidence_scores.get(cid, 0) * 100

        # Path snippet
        path_snippet = ""
        parents = G.nodes[cid].get("parents", [])
        if parents:
            parent_id = parents[0]
            parent_label = label_map.get(parent_id, parent_id.split('/')[-1])
            path_snippet = f"{label} — subClassOf → {parent_label}"
        else:
            path_snippet = "(Top-level concept)"

        print(f"| {term} | `{cid}` | {label} | {mtype} | {conf_percent:.0f}% | {path_snippet} |")

def print_reasoned_paths(G, matched_classes, label_map, alignment_score):
    """Print reasoned ontology paths."""
    print("\n**Reasoned Ontology Paths:**")
    reasoned_paths = []

    # Example 1: Property relationship path
    if alignment_score > 0:
        for prop_id, data in G.nodes(data=True):
            if data.get("type") == "Property":
                dom_list = data.get("domain", [])
                rng_list = data.get("range", [])
                if dom_list and rng_list and dom_list[0] in matched_classes and rng_list[0] in matched_classes:
                    dom_label = label_map.get(dom_list[0], dom_list[0].split('/')[-1])
                    rng_label = label_map.get(rng_list[0], rng_list[0].split('/')[-1])
                    prop_label = data["label"]
                    reasoned_paths.append(f"* **{dom_label}** — *{prop_label}* → **{rng_label}**")
                    break

    # Example 2: Sibling concepts under a common parent
    for parent_id in label_map:
        if G.nodes.get(parent_id, {}).get("children"):
            matched_kids = [c for c in G.nodes[parent_id]["children"] if c in matched_classes]
            if len(matched_kids) >= 2:
                sibs = matched_kids[:2]
                sib_labels = [label_map.get(s, s) for s in sibs]
                parent_label = label_map.get(parent_id, parent_id.split('/')[-1])
                reasoned_paths.append(f"* **{sib_labels[0]}** — subClassOf → **{parent_label}** — hasSubClass → **{sib_labels[1]}**")
                break

    # Example 3: Chain of subclasses
    for cid in matched_classes:
        chain_path = None
        for parent in G.nodes.get(cid, {}).get("parents", []):
            if parent in matched_classes:
                for grand in G.nodes.get(parent, {}).get("parents", []):
                    if grand in matched_classes:
                        chain_path = (cid, parent, grand)
                        break
                if chain_path:
                    break
        if chain_path:
            a, b, c = chain_path
            reasoned_paths.append(f"* **{label_map[a]}** — subClassOf → **{label_map[b]}** — subClassOf → **{label_map[c]}**")
            break

    # Print the collected paths
    for p in reasoned_paths[:3]:
        print(p)

def print_gaps_and_next_steps(G, roots, matched_classes, visited, consistent_count, verdict, label_map):
    """Print identified gaps and suggested next steps."""
    print("\n**Gaps & Next Steps:**")
    gaps_points = []

    # Identify top-level ontology areas covered vs not covered
    matched_roots = {cid for cid in roots if cid in visited or cid in matched_classes}
    uncovered_roots = [label_map[r] for r in roots if r not in matched_roots]

    if uncovered_roots:
        gaps_points.append(f"The paper focuses on **{', '.join(label_map[r] for r in matched_roots)}** domain, leaving other ontology areas (e.g., **{', '.join(uncovered_roots)}**) unaddressed.")

    # Terminology gap
    if consistent_count == 0 and len(matched_classes) > 0:
        gaps_points.append("The paper uses different terminology not directly present in the ontology. Consider adding synonyms or mappings for those terms.")

    # Based on verdict
    if verdict == "Not Relevant":
        gaps_points.append("Few to no ontology concepts are covered in the paper, suggesting it may be out of scope or the ontology might need extension to cover its topics.")
    elif verdict == "Partially Relevant":
        gaps_points.append("Only partial alignment was found. To improve coverage, identify missing key concepts in the paper and incorporate them into the ontology or vice-versa.")
    elif verdict == "Relevant":
        gaps_points.append("The ontology can likely be extended or updated with insights from the paper (e.g., new relationships or more specific subclasses noted).")

    for gp in gaps_points:
        print(f"* {gp}")
def analyze_chunk_similarities(docs, embedding_model):
    """Analyze and report similarities between paper chunks."""
    chunk_docs = [(d.id, d.page_content) for d in docs if d.metadata.get("type") == "PaperChunk"]
    if len(chunk_docs) >= 2:
        ids = [cid for cid, _ in chunk_docs]
        texts = [txt for _, txt in chunk_docs]

        # Embed all chunks at once
        chunk_vecs = embedding_model.embed_documents(texts)
        V = np.array(chunk_vecs, dtype=float)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

        # Cosine similarity matrix
        S = V @ V.T
        np.fill_diagonal(S, -1.0)  # ignore self matches

        # Get top-N global pairs (upper triangle)
        n = S.shape[0]
        tri_i, tri_j = np.triu_indices(n, k=1)
        sims = S[tri_i, tri_j]
        topN = 10
        top_idx = np.argsort(-sims)[:topN]

        print("\nTop similar PaperChunk pairs:")
        for idx in top_idx:
            i, j = int(tri_i[idx]), int(tri_j[idx])
            sim = float(sims[idx])
            t_i = texts[i][:120].replace("\n", " ") + ("..." if len(texts[i]) > 120 else "")
            t_j = texts[j][:120].replace("\n", " ") + ("..." if len(texts[j]) > 120 else "")
            print(f"- {ids[i]}  <->  {ids[j]}  |  cos={sim:.3f}")
            print(f"  • {t_i}")
            print(f"  • {t_j}")
    else:
        print("Not enough PaperChunk documents to compare.")


