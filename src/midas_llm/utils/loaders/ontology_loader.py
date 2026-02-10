from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...model.ontology import build_ontology_graph, create_ontology_documents, load_ontology

try:
    from ..ontology_linker.ncbi_taxonomy import NCBITaxonomy
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False

LOGGER = logging.getLogger("midas-llm")


def prepare_ontology_context(G, label_map: dict[str, str]) -> str:
    """Convert ontology graph to structured text context for the LLM."""
    context_parts = ["=== MIDAS Ontology Structure ===\n"]

    classes = []
    properties = []

    for node_id in G.nodes():
        node = G.nodes[node_id]
        node_type = node.get('type')
        label = label_map.get(node_id, node_id)

        if node_type == 'Class':
            parents = [label_map.get(p, p) for p in G.predecessors(node_id)]
            parent_str = f" (subclass of: {', '.join(parents)})" if parents else " (root class)"
            synonyms = node.get('synonyms', [])
            synonym_str = f" [synonyms: {', '.join(synonyms)}]" if synonyms else ""
            classes.append(f"  • {label}{parent_str}{synonym_str}")
        elif node_type == 'Property':
            properties.append(f"  • {label}")

    if classes:
        context_parts.append("Classes:")
        context_parts.extend(sorted(classes))

    if properties:
        context_parts.append("\nProperties:")
        context_parts.extend(sorted(properties))

    return "\n".join(context_parts)


def load_midas_ontology(ontology_path: str):
    """Load the MIDAS ontology and prepare context."""
    print("Loading MIDAS ontology...")
    g, classes, properties = load_ontology(ontology_path)
    G = build_ontology_graph(g, classes, properties)
    ontology_docs, label_map = create_ontology_documents(G)
    ontology_context = prepare_ontology_context(G, label_map)
    print(f"✓ Loaded ontology with {len(classes)} classes and {len(properties)} properties\n")
    return g, ontology_context, label_map, classes, properties


def start_background_ontology_loading():
    """Start loading ontologies in background threads."""
    from rdflib import Graph

    print("=" * 80)
    print("STARTING BACKGROUND ONTOLOGY LOADING")
    print("=" * 80)
    print("Pre-loading common ontologies while waiting for LLM response...\n")

    ontology_loader_executor = ThreadPoolExecutor(max_workers=6)

    def load_ncbi_taxonomy():
        if TAXONOMY_AVAILABLE:
            try:
                print("Loading NCBI Taxonomy...")
                tax = NCBITaxonomy()
                print("✓ NCBI Taxonomy loaded")
                return ('ncbi_taxonomy', tax, None)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: Could not load NCBI Taxonomy: {e}")
                return ('ncbi_taxonomy', None, str(e))
        return ('ncbi_taxonomy', None, 'Not available')

    def load_iso_3166():
        try:
            iso_path = "ontologies/iso_3166/iso_3166_countries.json"
            if os.path.exists(iso_path):
                with open(iso_path, 'r', encoding='utf-8') as f:
                    countries = json.load(f)
                print(f"✓ Loaded ISO 3166 ({len(countries)} countries)")
                return ('iso_3166', countries, None)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Could not load ISO 3166: {e}")
            return ('iso_3166', None, str(e))
        return ('iso_3166', None, 'Not found')

    def load_rdf_ontology(name: str, path: str):
        try:
            if os.path.exists(path):
                graph = Graph()
                graph.parse(path, format="xml")
                print(f"✓ Loaded {name} ({len(graph):,} triples)")
                return (name, graph, None)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Could not load {name}: {e}")
            return (name, None, str(e))
        return (name, None, 'Not found')

    background_tasks = {
        'ncbi_taxonomy': ontology_loader_executor.submit(load_ncbi_taxonomy),
        'iso_3166': ontology_loader_executor.submit(load_iso_3166),
        'apollo_sv': ontology_loader_executor.submit(load_rdf_ontology, 'apollo_sv', 'ontologies/apollo_sv/apollo_sv.owl'),
        'doid': ontology_loader_executor.submit(load_rdf_ontology, 'doid', 'ontologies/doid/doid.owl'),
        'ido': ontology_loader_executor.submit(load_rdf_ontology, 'ido', 'ontologies/ido/ido.owl'),
        'obi': ontology_loader_executor.submit(load_rdf_ontology, 'obi', 'ontologies/obi/obi.owl'),
        'stato': ontology_loader_executor.submit(load_rdf_ontology, 'stato', 'ontologies/stato/stato.owl'),
        'vo': ontology_loader_executor.submit(load_rdf_ontology, 'vo', 'ontologies/vo/vo.owl'),
    }

    return ontology_loader_executor, background_tasks


def finalize_ontology_loading(background_tasks: dict[str, Any], executor: ThreadPoolExecutor, midas_graph):
    """Wait for background ontology loading to complete and organize results."""
    if background_tasks is None or executor is None:
        return None

    print("Waiting for any remaining ontology loads to complete...")
    for ont_name, future in background_tasks.items():
        try:
            future.result(timeout=5)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: {ont_name} loading encountered an issue: {e}")

    ontologies = {
        'taxonomy': None,
        'iso_countries': None,
        'apollo_graph': None,
        'midas_graph': midas_graph,
        'doid_graph': None,
        'ido_graph': None,
        'obi_graph': None,
        'stato_graph': None,
        'vo_graph': None,
        'gaz_graph': None,
    }

    print("Collecting background-loaded ontologies...")
    for ont_name, future in background_tasks.items():
        try:
            result_name, result_data, error = future.result()
            if result_name == 'ncbi_taxonomy':
                ontologies['taxonomy'] = result_data
            elif result_name == 'apollo_sv':
                ontologies['apollo_graph'] = result_data
            elif result_name == 'doid':
                ontologies['doid_graph'] = result_data
            elif result_name == 'ido':
                ontologies['ido_graph'] = result_data
            elif result_name == 'iso_3166':
                ontologies['iso_countries'] = result_data
            elif result_name == 'obi':
                ontologies['obi_graph'] = result_data
            elif result_name == 'stato':
                ontologies['stato_graph'] = result_data
            elif result_name == 'vo':
                ontologies['vo_graph'] = result_data
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Error collecting {ont_name}: {e}")

    executor.shutdown(wait=False)
    print("\n✓ All required ontologies ready")
    return ontologies
