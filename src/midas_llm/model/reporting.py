import numpy as np


def generate_verdict(total_score):
    """Generate verdict based on total relevance score."""
    if total_score >= 70:
        return "Relevant"
    if total_score >= 30:
        return "Partially Relevant"
    return "Not Relevant"


def _safe_node_attrs(G, node_id):
    """Return node attrs dict if node exists, else empty dict."""
    try:
        if node_id in G.nodes:
            return G.nodes[node_id]
    except Exception:
        pass
    return {}


def _safe_label(label_map, node_id):
    """Get a readable label for a node id."""
    if node_id in label_map:
        return label_map[node_id]
    if isinstance(node_id, str):
        return node_id.split("/")[-1]
    return str(node_id)


def print_top_matches(
    top_concepts,
    G,
    label_map,
    confidence_scores,
    direct_label_hits,
    synonym_hits,
):
    """Print top matched ontology terms."""
    direct_label_hits = set(direct_label_hits or [])
    synonym_hits = set(synonym_hits or [])
    confidence_scores = confidence_scores or {}

    top_n = min(len(top_concepts or []), 25)
    print("\n**Top Matched Terms:**")
    print("| Paper Term | Ontology IRI | Ontology Label | Match Type | Confidence | Ontology Path |")
    print("|---|---|---|---|---|---|")

    for cid in (top_concepts or [])[:top_n]:
        attrs = _safe_node_attrs(G, cid)
        if not attrs or attrs.get("type") != "Class":
            continue

        label = _safe_label(label_map, cid)

        # Determine match type text
        if cid in direct_label_hits:
            mtype = "label"
            term = label
        elif cid in synonym_hits:
            mtype = "synonym"
            syns = attrs.get("synonyms", [])
            if isinstance(syns, str):
                syns = [syns]
            term = syns[0] if syns else label
        else:
            mtype = "embedding"
            term = label

        # Confidence as percentage
        conf_percent = float(confidence_scores.get(cid, 0) or 0) * 100.0

        # Path snippet
        parents = attrs.get("parents", []) or []
        if parents:
            parent_id = parents[0]
            parent_label = _safe_label(label_map, parent_id)
            path_snippet = f"{label} — subClassOf → {parent_label}"
        else:
            path_snippet = "(Top-level concept)"

        print(f"| {term} | `{cid}` | {label} | {mtype} | {conf_percent:.0f}% | {path_snippet} |")


def print_reasoned_paths(G, matched_classes, label_map, alignment_score):
    """Print reasoned ontology paths."""
    print("\n**Reasoned Ontology Paths:**")
    reasoned_paths = []
    matched_classes = set(matched_classes or [])

    # Example 1: Property relationship path
    if alignment_score > 0:
        for prop_id, data in G.nodes(data=True):
            if data.get("type") != "Property":
                continue

            dom_list = data.get("domain", []) or []
            rng_list = data.get("range", []) or []
            if dom_list and rng_list and dom_list[0] in matched_classes and rng_list[0] in matched_classes:
                dom_label = _safe_label(label_map, dom_list[0])
                rng_label = _safe_label(label_map, rng_list[0])
                prop_label = data.get("label", _safe_label(label_map, prop_id))
                reasoned_paths.append(f"* **{dom_label}** — *{prop_label}* → **{rng_label}**")
                break

    # Example 2: Sibling concepts under a common parent
    for parent_id in label_map.keys():
        parent_attrs = _safe_node_attrs(G, parent_id)
        children = parent_attrs.get("children", []) or []
        if not children:
            continue

        matched_kids = [c for c in children if c in matched_classes]
        if len(matched_kids) >= 2:
            sibs = matched_kids[:2]
            sib_labels = [_safe_label(label_map, s) for s in sibs]
            parent_label = _safe_label(label_map, parent_id)
            reasoned_paths.append(
                f"* **{sib_labels[0]}** — subClassOf → **{parent_label}** — hasSubClass → **{sib_labels[1]}**"
            )
            break

    # Example 3: Chain of subclasses
    for cid in matched_classes:
        attrs = _safe_node_attrs(G, cid)
        for parent in (attrs.get("parents", []) or []):
            if parent not in matched_classes:
                continue
            parent_attrs = _safe_node_attrs(G, parent)
            for grand in (parent_attrs.get("parents", []) or []):
                if grand in matched_classes:
                    a, b, c = cid, parent, grand
                    reasoned_paths.append(
                        f"* **{_safe_label(label_map, a)}** — subClassOf → **{_safe_label(label_map, b)}** — "
                        f"subClassOf → **{_safe_label(label_map, c)}**"
                    )
                    break
            if len(reasoned_paths) >= 3:
                break
        if len(reasoned_paths) >= 3:
            break

    if not reasoned_paths:
        print("* No strong multi-hop ontology paths identified from the matched concepts.")
        return

    for p in reasoned_paths[:3]:
        print(p)


def print_gaps_and_next_steps(G, roots, matched_classes, visited, consistent_count, verdict, label_map):
    """Print identified gaps and suggested next steps."""
    print("\n**Gaps & Next Steps:**")
    gaps_points = []

    matched_classes = set(matched_classes or [])
    visited = set(visited or [])
    roots = list(roots or [])

    # Identify top-level ontology areas covered vs not covered
    matched_roots = [r for r in roots if r in visited or r in matched_classes]
    uncovered_roots = [r for r in roots if r not in set(matched_roots)]

    matched_root_labels = [_safe_label(label_map, r) for r in matched_roots]
    uncovered_root_labels = [_safe_label(label_map, r) for r in uncovered_roots]

    if matched_root_labels and uncovered_root_labels:
        gaps_points.append(
            f"The paper focuses on **{', '.join(matched_root_labels)}** domain areas, "
            f"leaving other ontology areas (e.g., **{', '.join(uncovered_root_labels)}**) unaddressed."
        )
    elif uncovered_root_labels and not matched_root_labels:
        gaps_points.append(
            f"No top-level ontology areas were strongly covered. Uncovered areas include "
            f"**{', '.join(uncovered_root_labels)}**."
        )

    # Terminology gap
    if consistent_count == 0 and len(matched_classes) > 0:
        gaps_points.append(
            "The paper uses terminology not directly present in the ontology. "
            "Consider adding synonyms or mappings for those terms."
        )

    # Based on verdict
    if verdict == "Not Relevant":
        gaps_points.append(
            "Few to no ontology concepts are covered in the paper, suggesting it may be out of scope "
            "or the ontology may need extension to cover its topics."
        )
    elif verdict == "Partially Relevant":
        gaps_points.append(
            "Only partial alignment was found. To improve coverage, identify missing key concepts in the paper "
            "and incorporate them into the ontology or vice versa."
        )
    elif verdict == "Relevant":
        gaps_points.append(
            "The ontology can likely be extended or updated with insights from the paper "
            "(for example, new relationships or more specific subclasses)."
        )

    if not gaps_points:
        gaps_points.append("No major gaps were detected with the current matching and coverage heuristics.")

    for gp in gaps_points:
        print(f"* {gp}")


def analyze_chunk_similarities(docs, embedding_model):
    """Analyze and report similarities between paper chunks."""
    docs = docs or []
    chunk_docs = [
        (getattr(d, "id", None), getattr(d, "page_content", ""))
        for d in docs
        if getattr(d, "metadata", {}).get("type") == "PaperChunk"
    ]

    if len(chunk_docs) < 2:
        print("Not enough PaperChunk documents to compare.")
        return

    ids = [cid for cid, _ in chunk_docs]
    texts = [txt or "" for _, txt in chunk_docs]

    # Embed all chunks at once
    chunk_vecs = embedding_model.embed_documents(texts)
    V = np.array(chunk_vecs, dtype=float)

    if V.ndim != 2 or V.shape[0] < 2:
        print("Not enough valid embeddings to compare.")
        return

    # Normalize safely
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / (norms + 1e-12)

    # Cosine similarity matrix
    S = V @ V.T
    np.fill_diagonal(S, -1.0)  # ignore self matches

    # Get top-N global pairs (upper triangle)
    n = S.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    if len(tri_i) == 0:
        print("Not enough PaperChunk pairs to compare.")
        return

    sims = S[tri_i, tri_j]
    topN = min(10, len(sims))
    top_idx = np.argsort(-sims)[:topN]

    print("\nTop similar PaperChunk pairs:")
    for idx in top_idx:
        i, j = int(tri_i[idx]), int(tri_j[idx])
        sim = float(sims[idx])

        t_i_raw = texts[i].replace("\n", " ")
        t_j_raw = texts[j].replace("\n", " ")
        t_i = t_i_raw[:120] + ("..." if len(t_i_raw) > 120 else "")
        t_j = t_j_raw[:120] + ("..." if len(t_j_raw) > 120 else "")

        print(f"- {ids[i]}  <->  {ids[j]}  |  cos={sim:.3f}")
        print(f"  • {t_i}")
        print(f"  • {t_j}")