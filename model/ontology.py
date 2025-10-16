import networkx as nx
from langchain_core.documents import Document
from rdflib import Graph, OWL, RDF, RDFS, URIRef


def load_ontology(ontology_path):
    """Load and parse the OWL ontology."""
    g = Graph()
    g.parse(ontology_path)

    # Extract ontology elements
    classes = set(g.subjects(RDF.type, OWL.Class))
    properties = set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(g.subjects(RDF.type, OWL.DatatypeProperty))

    print(f"Loaded {len(classes)} classes and {len(properties)} properties from the ontology.")
    return g, classes, properties


def get_literal(g, subject, predicate):
    """Helper function to get literal value if triple exists."""
    obj = None
    for _, _, o in g.triples((subject, predicate, None)):
        obj = str(o)
        break
    return obj


def build_ontology_graph(g, classes, properties):
    """Build a NetworkX graph from ontology elements."""
    G = nx.DiGraph()

    # Add class nodes with attributes
    for cls in classes:
        label = get_literal(g, cls, RDFS.label) or cls.split('/')[-1].split('#')[-1]
        # Gather synonyms (alternate labels)
        synonyms = [str(o) for _, _, o in
                    g.triples((cls, URIRef("http://www.w3.org/2004/02/skos/core#altLabel"), None))]
        synonyms += [str(o) for _, _, o in
                     g.triples((cls, URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"), None))]
        # Gather textual definition or comment if any
        definition = get_literal(g, cls, URIRef("http://www.w3.org/2004/02/skos/core#definition")) or get_literal(g,
                                                                                                                  cls,
                                                                                                                  RDFS.comment)
        # Add node to graph
        G.add_node(str(cls), type="Class", label=label, synonyms=synonyms)
        if definition:
            G.nodes[str(cls)]["definition"] = definition

    # Add property nodes with attributes
    for prop in properties:
        label = get_literal(g, prop, RDFS.label) or prop.split('/')[-1].split('#')[-1]
        G.add_node(str(prop), type="Property", label=label)

    # Add relationships
    add_class_relationships(g, G, classes)
    add_property_relationships(g, G, properties)

    return G


def add_class_relationships(g, G, classes):
    """Add class hierarchy and equivalence relationships to the graph."""
    # Add subclass relationships
    for cls in classes:
        for _, _, parent in g.triples((cls, RDFS.subClassOf, None)):
            if str(parent) in G.nodes:  # ensure parent is a class node
                G.nodes[str(cls)].setdefault("parents", []).append(str(parent))
                G.nodes[str(parent)].setdefault("children", []).append(str(cls))
                G.add_edge(str(cls), str(parent), predicate="subClassOf")
                G.add_edge(str(parent), str(cls), predicate="hasSubClass")

    # Add equivalent class relationships
    for cls in classes:
        for _, _, eq in g.triples((cls, OWL.equivalentClass, None)):
            if str(eq) in G.nodes:
                G.nodes[str(cls)].setdefault("equivalentClasses", []).append(str(eq))
                G.nodes[str(eq)].setdefault("equivalentClasses", []).append(str(cls))
                G.add_edge(str(cls), str(eq), predicate="equivalentClass")
                G.add_edge(str(eq), str(cls), predicate="equivalentClass")


def add_property_relationships(g, G, properties):
    """Add domain and range relationships to the graph."""
    for prop in properties:
        for _, _, dom in g.triples((prop, RDFS.domain, None)):
            if str(dom) in G.nodes:
                G.nodes[str(prop)].setdefault("domain", []).append(str(dom))
                G.nodes[str(dom)].setdefault("domainOf", []).append(str(prop))
                G.add_edge(str(prop), str(dom), predicate="domain")
                G.add_edge(str(dom), str(prop), predicate="domainOf")
        for _, _, rng in g.triples((prop, RDFS.range, None)):
            if str(rng) in G.nodes:
                G.nodes[str(prop)].setdefault("range", []).append(str(rng))
                G.nodes[str(rng)].setdefault("rangeOf", []).append(str(prop))
                G.add_edge(str(prop), str(rng), predicate="range")
                G.add_edge(str(rng), str(prop), predicate="rangeOf")


def create_ontology_documents(G):
    """Create LangChain documents from ontology graph nodes."""
    docs = []
    label_map = {}  # Map from node ID to label for quick lookup

    for node_id, data in G.nodes(data=True):
        if data["type"] == "Class":
            # Compose content text for class: label + definition + synonyms
            text = data["label"]
            if "definition" in data:
                text += ". " + data["definition"]
            if data.get("synonyms"):
                text += ". Synonyms: " + ", ".join(data["synonyms"])
            # Metadata with graph links
            metadata = {
                "type": "Class",
                "label": data["label"],
                "parents": data.get("parents", []),
                "children": data.get("children", []),
                "equivalentClasses": data.get("equivalentClasses", []),
                "domainOf": data.get("domainOf", []),
                "rangeOf": data.get("rangeOf", []),
                "synonyms": data.get("synonyms", [])
            }
            docs.append(Document(page_content=text, metadata=metadata, id=node_id))
            label_map[node_id] = data["label"]
        elif data["type"] == "Property":
            # Compose content for property: label + domain/range context
            dom_labels = [label_map.get(d, d) for d in data.get("domain", [])]
            rng_labels = [label_map.get(r, r) for r in data.get("range", [])]
            text = data["label"]
            if dom_labels or rng_labels:
                text += ". Connects " + (", ".join(dom_labels) if dom_labels else "things")
                text += " to " + (", ".join(rng_labels) if rng_labels else "things") + "."
            metadata = {
                "type": "Property",
                "label": data["label"],
                "domain": data.get("domain", []),
                "range": data.get("range", [])
            }
            docs.append(Document(page_content=text, metadata=metadata, id=node_id))
            label_map[node_id] = data["label"]

    return docs, label_map


def calculate_ontology_depth(G, label_map):
    """Calculate depth of each class in the ontology hierarchy."""
    # Find all root classes (no parent)
    roots = [cid for cid in label_map.keys() if G.nodes[cid].get("type") == "Class" and not G.nodes[cid].get("parents")]
    # BFS from each root to assign depths
    depth_map = {}
    for root in roots:
        depth_map[root] = 0
        queue = [root]
        while queue:
            cur = queue.pop(0)
            cur_depth = depth_map[cur]
            for child in G.nodes[cur].get("children", []):
                if child not in depth_map or depth_map[child] > cur_depth + 1:
                    depth_map[child] = cur_depth + 1
                    queue.append(child)

    return depth_map, roots
