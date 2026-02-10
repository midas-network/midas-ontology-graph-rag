import numpy as np


def calculate_confidence_scores(all_matched_concepts, query_text, docs, embedding_model, label_map):
    """Calculate embedding similarity confidence for each concept."""
    query_embedding = embedding_model.embed_query(query_text)
    confidence_scores = {}

    for cid in all_matched_concepts:
        if cid in label_map:
            doc_text = ""
            for d in docs:
                if d.id == cid:
                    doc_text = d.page_content
                    break
            if doc_text:
                concept_embedding = embedding_model.embed_query(doc_text)
                sim = np.dot(query_embedding, concept_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(concept_embedding))
                confidence_scores[cid] = float(sim)

    return confidence_scores


def calculate_relevance_scores(G, classes, matched_classes, depth_map):
    """Calculate various relevance scores for ontology match."""
    # 1. Concept Coverage (30%)
    total_classes = len(classes)
    num_matched = len(matched_classes)
    coverage_ratio = min(num_matched, 5) / 5.0
    coverage_score = min(coverage_ratio, 1.0) * 30.0

    # 2. Hierarchy Fit (20%)
    depths = [depth_map.get(cid, 0) for cid in matched_classes if cid in depth_map]
    if depths:
        avg_depth = sum(depths) / len(depths)
        hierarchy_score = min(avg_depth / 3.0, 1.0) * 20.0
    else:
        hierarchy_score = 0.0

    # 3. Property Alignment (20%)
    alignment_score = 0.0
    if num_matched > 1:
        aligned_relations = 0
        for cid in matched_classes:
            for prop_id in G.nodes[cid].get("domainOf", []) + G.nodes[cid].get("rangeOf", []):
                prop_dom = G.nodes[prop_id].get("domain", [])
                prop_rng = G.nodes[prop_id].get("range", [])
                for c in prop_dom + prop_rng:
                    if c in matched_classes and c != cid:
                        aligned_relations += 1
                        break
        if aligned_relations > 0:
            alignment_score = min(aligned_relations, 2) / 2.0 * 20.0

    # 4. Graph Coherence (15%)
    coherence_score = 0.0
    if num_matched > 0:
        subg_nodes = set(matched_classes)
        subg_edges = []
        for u, v, data in G.edges(data=True):
            if data["predicate"] in ("subClassOf", "equivalentClass") and u in subg_nodes and v in subg_nodes:
                subg_edges.append((u, v))
        undirected_edges = [(u,v) for (u,v) in subg_edges] + [(v,u) for (u,v) in subg_edges]
        comp_count = 0
        visited = set()
        for node in subg_nodes:
            if node not in visited:
                comp_count += 1
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        for neigh in G.nodes[n].get("parents", []) + G.nodes[n].get("children", []):
                            if neigh in subg_nodes and neigh not in visited:
                                stack.append(neigh)
        if comp_count == 1:
            coherence_score = 15.0
        else:
            coherence_score = max(0.0, 15.0 * (1 - (comp_count-1)/len(matched_classes)))

    return coverage_score, hierarchy_score, alignment_score, coherence_score, visited

def calculate_terminology_scores(matched_classes, direct_label_hits, synonym_hits):
    """Calculate terminology consistency and evidence quality scores."""
    # 5. Terminology Consistency (10%)
    num_matched = len(matched_classes)
    consistent_count = sum(1 for cid in matched_classes if cid in direct_label_hits or cid in synonym_hits)

    if num_matched > 0:
        consistency_score = (consistent_count / num_matched) * 10.0
    else:
        consistency_score = 0.0

    # 6. Evidence Quality (5%)
    evidence_score = 0.0
    if consistent_count > 0 and num_matched > 1:
        evidence_score = 5.0
    elif consistent_count > 0:
        evidence_score = 4.0
    elif num_matched > 0:
        evidence_score = 2.0

    return consistency_score, evidence_score, consistent_count