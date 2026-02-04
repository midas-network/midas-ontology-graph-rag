"""
Shared ontology lookup utilities for concept-to-ontology mapping.
"""
import re
from typing import Dict, List, Optional

try:
    from utils.ontology_linker.ncbi_taxonomy import NCBITaxonomy
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False


def lookup_in_ncbi_taxonomy(value: str, taxonomy) -> Optional[Dict]:
    """
    Look up a value in NCBI Taxonomy.

    Args:
        value: The value to look up
        taxonomy: NCBITaxonomy instance

    Returns:
        Match information or None
    """
    if not taxonomy or not TAXONOMY_AVAILABLE:
        return None

    # Clean the value
    clean_value = re.sub(r'\s*\([^)]*\)', '', value)
    clean_value = re.sub(r'\s+(vector|host)$', '', clean_value, flags=re.IGNORECASE)
    clean_value = clean_value.strip()

    try:
        # Try common name
        results = taxonomy.lookup_common_name(clean_value)
        if not results:
            # Try scientific name
            results = taxonomy.lookup_species(clean_value)
        if not results:
            # Try fuzzy search
            results = taxonomy.search_fuzzy(clean_value, max_results=1)

        if results:
            best = results[0]

            # Build definition from lineage if available
            definition = None
            if 'lineage' in best:
                definition = f"Taxonomic lineage: {best['lineage']}"
            elif 'rank' in best:
                definition = f"Taxonomic rank: {best['rank']}"

            return {
                'ontology': 'NCBI Taxonomy',
                'identifier': f"NCBI:txid{best['tax_id']}",
                'matched_term': best.get('scientific_name'),
                'rank': best.get('rank'),
                'definition': definition,
                'status': 'found'
            }
    except Exception as e:
        return {
            'ontology': 'NCBI Taxonomy',
            'status': 'error',
            'error': str(e)
        }

    return None


def lookup_in_iso3166(value: str, iso_countries: List[Dict]) -> Optional[Dict]:
    """
    Look up a value in ISO 3166 country codes.

    Args:
        value: The value to look up
        iso_countries: ISO 3166 country data

    Returns:
        Match information or None
    """
    if not iso_countries:
        return None

    value_lower = value.lower()

    for country in iso_countries:
        country_name = country.get('name', '').lower()
        alpha2 = country.get('alpha-2', '')
        alpha3 = country.get('alpha-3', '')

        if (value_lower in country_name or
            country_name in value_lower or
            value_lower == alpha2.lower() or
            value_lower == alpha3.lower()):

            # Build definition
            definition = f"Country code: {alpha2} ({alpha3}). Official name: {country.get('name')}"

            return {
                'ontology': 'ISO 3166',
                'identifier': f"ISO:{alpha2}",
                'matched_term': country.get('name'),
                'rank': 'country',
                'definition': definition,
                'status': 'found'
            }

    return None


def lookup_in_rdf_ontology(value: str, ontology_graph, ontology_name: str) -> Optional[Dict]:
    """
    Look up a value in an RDF/OWL ontology (Apollo SV, MIDAS, DOID, etc.).

    Searches multiple properties including labels, synonyms, and definitions.

    Args:
        value: The value to look up
        ontology_graph: RDFLib graph
        ontology_name: Name of the ontology (e.g., "Apollo SV", "MIDAS Ontology")

    Returns:
        Match information or None
    """
    if not ontology_graph:
        return None

    from rdflib import RDFS, Namespace

    # Define OBO namespace for IAO terms
    OBO = Namespace("http://purl.obolibrary.org/obo/")

    value_lower = value.lower()

    # Properties to search (in order of preference)
    properties_to_search = [
        (RDFS.label, 'label', 4),                    # Primary label - highest priority
        (OBO.IAO_0000118, 'synonym', 3),             # Synonym/alternative term
        (OBO.IAO_0000111, 'editor_term', 3),         # Editor preferred term
        (OBO.IAO_0000115, 'definition', 1),          # Definition (lowest priority)
    ]

    best_match = None
    best_score = 0

    # Tokenize the search value for partial matching
    value_tokens = set(value_lower.split())
    # Remove very common words that don't help matching
    stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    value_tokens = value_tokens - stop_words

    for prop, prop_name, base_score in properties_to_search:
        for subj, pred, obj in ontology_graph.triples((None, prop, None)):
            label = str(obj).lower()

            # Skip very short labels/synonyms to avoid false matches (e.g., "o", "a", "i")
            if len(label.strip()) <= 2:
                continue

            # Calculate match score
            if value_lower == label:
                # Exact match
                score = base_score * 2
            elif value_lower in label:
                # Value is substring of label
                score = base_score * 1.5
            elif label in value_lower:
                # Label is substring of value
                score = base_score
            else:
                # Token-based matching: check if significant tokens from value appear in label
                label_tokens = set(label.split()) - stop_words
                if value_tokens and label_tokens:
                    # Calculate overlap: how many value tokens are in the label
                    overlap = value_tokens & label_tokens
                    if overlap:
                        # Partial match based on token overlap
                        overlap_ratio = len(overlap) / len(value_tokens)
                        if overlap_ratio >= 0.5:  # At least 50% of tokens match
                            score = base_score * overlap_ratio * 0.8  # Lower than exact match
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            if score > best_score:
                best_score = score

                # Get the preferred label for display
                pref_label = ontology_graph.value(subj, RDFS.label)
                if not pref_label:
                    pref_label = obj  # Use the matched value if no label

                # Get the definition if available
                definition = ontology_graph.value(subj, OBO.IAO_0000115)
                if definition:
                    definition = str(definition)
                else:
                    definition = None

                best_match = {
                    'ontology': ontology_name,
                    'identifier': str(subj),
                    'matched_term': str(pref_label),
                    'matched_via': str(obj) if str(obj) != str(pref_label) else None,
                    'property': prop_name,
                    'definition': definition,
                    'status': 'found'
                }

    return best_match


def lookup_in_all_ontologies(
    value: str,
    taxonomy=None,
    iso_countries: Optional[List[Dict]] = None,
    apollo_graph=None,
    midas_graph=None,
    doid_graph=None,
    ido_graph=None,
    obi_graph=None,
    stato_graph=None,
    vo_graph=None,
    gaz_graph=None,
    mesh_graph=None
) -> List[Dict]:
    """
    Search ALL available ontologies for a concept and return ALL matches.

    Args:
        value: The value to search for
        All ontology graphs and resources

    Returns:
        List of all matches found across all ontologies
    """
    all_matches = []

    # NCBI Taxonomy
    if taxonomy and TAXONOMY_AVAILABLE:
        match = lookup_in_ncbi_taxonomy(value, taxonomy)
        if match and match.get('status') == 'found':
            all_matches.append(match)

    # ISO 3166
    if iso_countries:
        match = lookup_in_iso3166(value, iso_countries)
        if match and match.get('status') == 'found':
            all_matches.append(match)

    # All RDF ontologies
    rdf_ontologies = [
        (apollo_graph, "Apollo SV"),
        (midas_graph, "MIDAS Ontology"),
        (doid_graph, "Disease Ontology (DOID)"),
        (ido_graph, "Infectious Disease Ontology (IDO)"),
        (obi_graph, "Ontology for Biomedical Investigations (OBI)"),
        (stato_graph, "Statistical Methods Ontology (STATO)"),
        (vo_graph, "Vaccine Ontology (VO)"),
        (gaz_graph, "Gazetteer (GAZ)"),
        (mesh_graph, "Medical Subject Headings (MeSH)")
    ]

    for ont_graph, ont_name in rdf_ontologies:
        if ont_graph:
            match = lookup_in_rdf_ontology(value, ont_graph, ont_name)
            if match and match.get('status') == 'found':
                all_matches.append(match)

    return all_matches


def lookup_multiple_concepts(
    attribute: str,
    value: str,
    taxonomy=None,
    iso_countries: Optional[List[Dict]] = None,
    apollo_graph=None,
    midas_graph=None,
    doid_graph=None,
    ido_graph=None,
    obi_graph=None,
    stato_graph=None,
    vo_graph=None,
    gaz_graph=None,
    mesh_graph=None
) -> List[Dict]:
    """
    Look up multiple concepts from a comma-separated list.

    Returns a list of results, one for each item in the value.
    """
    # Clean up list notation
    clean_value = value.strip()
    if clean_value.startswith('[') and clean_value.endswith(']'):
        clean_value = clean_value[1:-1]
    clean_value = clean_value.strip('"\'')

    # Split by comma
    if ',' in clean_value:
        items = [item.strip().strip('"\'') for item in clean_value.split(',')]
    else:
        items = [clean_value]

    # Lookup each item
    results = []
    for item in items:
        if not item or item.lower() in ['unspecified', 'unknown', 'not specified', 'n/a', 'none']:
            continue

        result = lookup_concept_in_ontologies(
            attribute=attribute,
            value=item,
            taxonomy=taxonomy,
            iso_countries=iso_countries,
            apollo_graph=apollo_graph,
            midas_graph=midas_graph,
            doid_graph=doid_graph,
            ido_graph=ido_graph,
            obi_graph=obi_graph,
            stato_graph=stato_graph,
            vo_graph=vo_graph,
            gaz_graph=gaz_graph,
            mesh_graph=mesh_graph,
            _is_list_item=True  # Flag to prevent recursive splitting
        )

        results.append(result)

    return results if results else [lookup_concept_in_ontologies(attribute, value, taxonomy, iso_countries, apollo_graph, midas_graph, doid_graph, ido_graph, obi_graph, stato_graph, vo_graph, gaz_graph, mesh_graph)]


def lookup_concept_in_ontologies(
    attribute: str,
    value: str,
    taxonomy=None,
    iso_countries: Optional[List[Dict]] = None,
    apollo_graph=None,
    midas_graph=None,
    doid_graph=None,
    ido_graph=None,
    obi_graph=None,
    stato_graph=None,
    vo_graph=None,
    gaz_graph=None,
    mesh_graph=None,
    _is_list_item: bool = False
) -> Dict[str, any]:
    """
    Look up a concept in appropriate ontologies based on the attribute type.

    Args:
        attribute: The attribute name (e.g., "host_species", "key_outcome_measures")
        value: The extracted value
        taxonomy: NCBI Taxonomy instance
        iso_countries: ISO 3166 country data
        apollo_graph: Apollo SV RDF graph
        midas_graph: MIDAS ontology RDF graph
        doid_graph: Disease Ontology RDF graph
        ido_graph: Infectious Disease Ontology RDF graph
        obi_graph: Ontology for Biomedical Investigations RDF graph
        stato_graph: Statistical Methods Ontology RDF graph
        vo_graph: Vaccine Ontology RDF graph
        gaz_graph: Gazetteer RDF graph
        mesh_graph: Medical Subject Headings RDF graph
        _is_list_item: Internal flag to prevent recursive list splitting

    Returns:
        Dictionary with lookup results
    """
    result = {
        'attribute': attribute,
        'value': value,
        'ontology': None,
        'identifier': None,
        'matched_term': None,
        'rank': None,
        'status': 'not_looked_up'
    }

    # Skip if value indicates unspecified
    if not value or any(marker in value.lower() for marker in ['unspecified', 'unknown', 'not specified', 'n/a', 'none']):
        result['status'] = 'unspecified'
        return result

    # Boolean or simple text fields that don't need ontology mapping
    if attribute in ['intervention_present', 'calibration_mentioned', 'code_available',
                     'historical_vs_hypothetical', 'extraction_notes',
                     'study_dates_start', 'study_dates_end']:
        result['status'] = 'no_ontology_needed'
        return result

    # Handle comma-separated lists only if not already processing a list item
    if not _is_list_item and ',' in value:
        # Clean up list notation if present
        clean_value = value.strip()
        if clean_value.startswith('[') and clean_value.endswith(']'):
            clean_value = clean_value[1:-1]
        clean_value = clean_value.strip('"\'')

        # Extract first item for single-result mode
        items = [item.strip().strip('"\'') for item in clean_value.split(',')]
        for item in items:
            if item and item.lower() not in ['unspecified', 'unknown', 'not specified', 'n/a', 'none']:
                value = item
                break

    # Host species or pathogen -> NCBI Taxonomy
    if attribute in ['host_species', 'pathogen_name']:
        match = lookup_in_ncbi_taxonomy(value, taxonomy)
        if match:
            return {**result, **match}
        else:
            result['ontology'] = 'NCBI Taxonomy'
            result['status'] = 'not_found'
            return result

    # Geographic locations -> ISO 3166 / GeoNames / GAZ
    elif attribute in ['geographic_units', 'geographic_scope']:
        # Try ISO 3166 first for countries
        match = lookup_in_iso3166(value, iso_countries)
        if match:
            return {**result, **match}

        # Try GAZ for more detailed geographic locations
        if gaz_graph:
            match = lookup_in_rdf_ontology(value, gaz_graph, 'Gazetteer (GAZ)')
            if match:
                return {**result, **match}

        result['ontology'] = 'ISO 3166 / GeoNames / GAZ'
        result['status'] = 'not_found'
        return result

    # Disease names -> Try DOID then IDO
    elif attribute in ['disease_name']:
        # Try DOID first (Human Disease Ontology)
        if doid_graph:
            match = lookup_in_rdf_ontology(value, doid_graph, 'Disease Ontology (DOID)')
            if match:
                return {**result, **match}

        # Try IDO (Infectious Disease Ontology)
        if ido_graph:
            match = lookup_in_rdf_ontology(value, ido_graph, 'Infectious Disease Ontology (IDO)')
            if match:
                return {**result, **match}

        result['ontology'] = 'DOID / IDO'
        result['status'] = 'not_found'
        return result

    # Pathogen type -> Try IDO then DOID
    elif attribute in ['pathogen_type']:
        # Try IDO first (more specific for infectious diseases)
        if ido_graph:
            match = lookup_in_rdf_ontology(value, ido_graph, 'Infectious Disease Ontology (IDO)')
            if match:
                return {**result, **match}

        # Try DOID
        if doid_graph:
            match = lookup_in_rdf_ontology(value, doid_graph, 'Disease Ontology (DOID)')
            if match:
                return {**result, **match}

        result['ontology'] = 'IDO / DOID'
        result['status'] = 'not_found'
        return result

    # Vaccination/intervention types -> Try VO then Apollo/MIDAS
    elif attribute in ['intervention_types']:
        # Check if value mentions vaccination
        if 'vaccin' in value.lower():
            if vo_graph:
                match = lookup_in_rdf_ontology(value, vo_graph, 'Vaccine Ontology (VO)')
                if match:
                    return {**result, **match}

        # Fall through to Apollo/MIDAS below
        pass

    # Statistical/calibration methods -> Try STATO then MeSH then Apollo/MIDAS
    elif attribute in ['calibration_techniques', 'statistical_methods']:
        # Try STATO first
        if stato_graph:
            match = lookup_in_rdf_ontology(value, stato_graph, 'Statistical Methods Ontology (STATO)')
            if match:
                return {**result, **match}

        # Try MeSH for medical/statistical terms
        if mesh_graph:
            match = lookup_in_rdf_ontology(value, mesh_graph, 'Medical Subject Headings (MeSH)')
            if match:
                return {**result, **match}

        # Fall through to Apollo/MIDAS below
        pass

    # Study design/methods -> Try OBI then Apollo/MIDAS
    elif attribute in ['study_design', 'data_collection_methods',
                       'population_setting_type', 'primary_population']:
        # Try OBI first
        if obi_graph:
            match = lookup_in_rdf_ontology(value, obi_graph, 'Ontology for Biomedical Investigations (OBI)')
            if match:
                return {**result, **match}

        # Try Apollo/MIDAS
        if apollo_graph:
            match = lookup_in_rdf_ontology(value, apollo_graph, 'Apollo SV')
            if match:
                return {**result, **match}

        if midas_graph:
            match = lookup_in_rdf_ontology(value, midas_graph, 'MIDAS Ontology')
            if match:
                return {**result, **match}

        result['ontology'] = 'OBI / Apollo SV / MIDAS'
        result['status'] = 'not_found'
        return result

    # Model type, outcomes, etc -> Try Apollo SV then MIDAS then STATO
    if attribute in ['model_type', 'intervention_types', 'study_goal_category',
                       'model_determinism', 'calibration_techniques',
                       'key_outcome_measures', 'data_used', 'statistical_methods',
                       'study_design', 'data_collection_methods']:
        # Try Apollo SV first
        if apollo_graph:
            match = lookup_in_rdf_ontology(value, apollo_graph, 'Apollo SV')
            if match:
                return {**result, **match}

        # Try MIDAS ontology
        if midas_graph:
            match = lookup_in_rdf_ontology(value, midas_graph, 'MIDAS Ontology')
            if match:
                return {**result, **match}

        # Try STATO for statistical/parameter terms (especially for study_goal_category, calibration_techniques)
        if stato_graph and attribute in ['study_goal_category', 'calibration_techniques', 'statistical_methods']:
            match = lookup_in_rdf_ontology(value, stato_graph, 'Statistical Methods Ontology (STATO)')
            if match:
                return {**result, **match}

        # Try MeSH for medical/epidemiological/statistical terms
        if mesh_graph and attribute in ['study_goal_category', 'calibration_techniques', 'statistical_methods', 'key_outcome_measures']:
            match = lookup_in_rdf_ontology(value, mesh_graph, 'Medical Subject Headings (MeSH)')
            if match:
                return {**result, **match}

        result['ontology'] = 'Apollo SV / MIDAS'
        result['status'] = 'not_found'
        return result

    # Default: no mapping defined for this attribute
    return result
