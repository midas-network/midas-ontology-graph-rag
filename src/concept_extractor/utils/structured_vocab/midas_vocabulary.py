"""
Parse MIDAS OWL structured_vocab into constrained vocabularies for LLM extraction.

Pipeline:
  1. Parse OWL/RDF → extract classes with labels, alt terms, definitions, hierarchy
  2. Map structured_vocab classes to extraction schema fields
  3. Build JSON schema with enum constraints for TensorRT-LLM guided decoding
  4. Build synonym map for post-extraction normalization

Usage:
  vocab = MIDASVocabulary.from_owl("midas-data.owl")
  schema = vocab.build_json_schema()        # for constrained decoding
  synonyms = vocab.build_synonym_map()      # for post-normalization
  prompt_text = vocab.build_prompt_section() # for prompt injection
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# OWL Parser
# ═══════════════════════════════════════════════════════════════════════

# XML namespaces used in the MIDAS OWL file
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "obo": "http://purl.obolibrary.org/obo/",
    "datasets": "http://w3id.org/midas-metadata/",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "apollo_sv": "http://purl.obolibrary.org/obo/apollo_sv.owl/",
}

# IAO annotation property IRIs
IAO_DEFINITION = "http://purl.obolibrary.org/obo/IAO_0000115"
IAO_ALT_TERM = "http://purl.obolibrary.org/obo/IAO_0000118"
IAO_ELUCIDATION = "http://purl.obolibrary.org/obo/IAO_0000600"


@dataclass
class OntologyClass:
    """A single class from the MIDAS OWL structured_vocab."""
    iri: str
    label: str = ""
    alt_terms: list[str] = field(default_factory=list)
    definition: str = ""
    parent_iri: str | None = None
    deprecated: bool = False

    @property
    def short_id(self) -> str:
        """Extract short ID from IRI (e.g., 'APOLLO_SV_00000142')."""
        return self.iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

    @property
    def all_terms(self) -> list[str]:
        """All terms: label + alt_terms, deduplicated."""
        terms = [self.label] + self.alt_terms if self.label else list(self.alt_terms)
        seen = set()
        result = []
        for t in terms:
            t_lower = t.lower().strip()
            if t_lower and t_lower not in seen:
                seen.add(t_lower)
                result.append(t.strip())
        return result


def parse_owl(owl_path: str | Path) -> list[OntologyClass]:
    """Parse OWL/RDF-XML file and extract all non-deprecated classes."""
    tree = ET.parse(owl_path)
    root = tree.getroot()
    classes = []

    for cls_elem in root.findall("owl:Class", NS):
        iri = cls_elem.get(f"{{{NS['rdf']}}}about", "")
        if not iri:
            continue

        oc = OntologyClass(iri=iri)

        # Check deprecated
        dep = cls_elem.find("owl:deprecated", NS)
        if dep is not None and dep.text and dep.text.lower() == "true":
            oc.deprecated = True

        # rdfs:label
        for label_el in cls_elem.findall("rdfs:label", NS):
            text = (label_el.text or "").strip()
            if text and not oc.label:
                oc.label = text

        # rdfs:subClassOf → parent
        sub = cls_elem.find("rdfs:subClassOf", NS)
        if sub is not None:
            oc.parent_iri = sub.get(f"{{{NS['rdf']}}}resource", "")

        # IAO_0000118 → alternative terms
        for elem in cls_elem:
            tag_uri = elem.tag
            if IAO_ALT_TERM in tag_uri or tag_uri == f"{{{NS['obo']}}}IAO_0000118":
                text = (elem.text or "").strip()
                if text:
                    oc.alt_terms.append(text)

        # IAO_0000115 → definition
        for elem in cls_elem:
            if IAO_DEFINITION in elem.tag or elem.tag == f"{{{NS['obo']}}}IAO_0000115":
                text = (elem.text or "").strip()
                if text and not oc.definition:
                    oc.definition = text

        if not oc.deprecated:
            classes.append(oc)

    return classes


# ═══════════════════════════════════════════════════════════════════════
# Field mapping: which structured_vocab classes map to which schema fields
# ═══════════════════════════════════════════════════════════════════════

# Map parent IRIs / class patterns to extraction schema fields
FIELD_MAPPING_RULES = {
    # Data types (children of midas2 or APOLLO_SV_00000617)
    "data_used": {
        "parent_iris": [
            "http://w3id.org/midas-metadata/midas2",
            "http://purl.obolibrary.org/obo/APOLLO_SV_00000617",
            "http://purl.obolibrary.org/obo/EUPATH_0000587",
            "http://purl.obolibrary.org/obo/STATO_0000107",
        ],
        "label_patterns": ["data", "census", "count", "survey"],
    },
    # Interventions / control strategies
    "intervention_types": {
        "parent_iris": [
            "http://purl.obolibrary.org/obo/APOLLO_SV_00000604",
            "http://purl.obolibrary.org/obo/APOLLO_SV_00000142",
        ],
        "label_patterns": ["vaccination", "treatment", "closure", "control"],
    },
    # Model types
    "model_type": {
        "parent_iris": [
            "http://w3id.org/midas-metadata/midas15",
        ],
        "label_patterns": ["model", "transmission", "phylogenetic"],
    },
    # Diseases
    "disease_name": {
        "parent_iris": [
            "http://purl.obolibrary.org/obo/DOID_4",
        ],
        "label_patterns": [],
    },
    # Behavioral / social data
    "data_used_behavioral": {
        "parent_iris": [
            "http://w3id.org/midas-metadata/midas79",
            "http://w3id.org/midas-metadata/midas27",
        ],
        "label_patterns": ["movement", "contact", "intent", "behavior"],
    },
}


class MIDASVocabulary:
    """Controlled vocabulary extracted from the MIDAS OWL structured_vocab."""

    def __init__(self, classes: list[OntologyClass]):
        self.classes = classes
        self._by_iri = {c.iri: c for c in classes}
        self._field_terms: dict[str, list[dict]] = {}
        self._build_field_mapping()

    @classmethod
    def from_owl(cls, owl_path: str | Path) -> MIDASVocabulary:
        classes = parse_owl(owl_path)
        return cls(classes)

    def _build_field_mapping(self):
        """Map each structured_vocab class to an extraction field."""
        for field_name, rules in FIELD_MAPPING_RULES.items():
            terms = []
            for oc in self.classes:
                matched = False

                # Check parent IRI
                if oc.parent_iri in rules["parent_iris"]:
                    matched = True

                # Check grandparent (one level up)
                if not matched and oc.parent_iri:
                    parent = self._by_iri.get(oc.parent_iri)
                    if parent and parent.parent_iri in rules["parent_iris"]:
                        matched = True

                # Check label patterns
                if not matched and rules.get("label_patterns"):
                    label_lower = oc.label.lower()
                    for pat in rules["label_patterns"]:
                        if pat in label_lower:
                            matched = True
                            break

                if matched:
                    terms.append({
                        "iri": oc.iri,
                        "label": oc.label,
                        "alt_terms": oc.alt_terms,
                        "definition": oc.definition,
                        "all_terms": oc.all_terms,
                    })

            self._field_terms[field_name] = terms

        # Merge behavioral data into data_used
        if "data_used_behavioral" in self._field_terms:
            self._field_terms.setdefault("data_used", []).extend(
                self._field_terms.pop("data_used_behavioral")
            )

    def get_terms(self, field: str) -> list[str]:
        """Get all canonical terms for a field (labels only)."""
        return [t["label"] for t in self._field_terms.get(field, [])]

    def get_all_terms(self, field: str) -> list[str]:
        """Get all terms including alternates for a field."""
        result = []
        for t in self._field_terms.get(field, []):
            result.extend(t["all_terms"])
        return result

    def get_enum_values(self, field: str) -> list[str]:
        """Get deduplicated enum values: alt_terms preferred (shorter),
        falling back to labels."""
        values = []
        seen = set()
        for entry in self._field_terms.get(field, []):
            # Prefer alt_term (typically shorter/cleaner)
            preferred = entry["alt_terms"][0] if entry["alt_terms"] else entry["label"]
            if preferred.lower() not in seen:
                seen.add(preferred.lower())
                values.append(preferred)
        return sorted(values)

    # ─── Output builders ───────────────────────────────────────────

    def build_json_schema(self) -> dict:
        """Build JSON schema for TensorRT-LLM constrained decoding.

        Fields with MIDAS structured_vocab terms use enum+additionalItems
        so the LLM prefers structured_vocab terms but can add novel ones.
        """
        # Fixed categorical vocabularies (not from structured_vocab)
        fixed_enums = {
            "model_determinism": [
                "deterministic", "stochastic", "both",
                "not mentioned", "not applicable"
            ],
            "pathogen_type": [
                "virus", "bacterium", "parasite", "fungus",
                "prion", "multiple", "not mentioned"
            ],
            "geographic_scope": [
                "facility-level", "city-level", "sub-national",
                "national", "multi-country", "global",
                "not specified", "not applicable"
            ],
            "historical_vs_hypothetical": [
                "historical", "hypothetical", "mixed", "not mentioned"
            ],
            "study_goal_category": [
                "forecast/nowcast", "scenario analysis",
                "intervention evaluation", "parameter estimation",
                "methodological", "transmission dynamics/drivers",
                "burden estimation", "risk assessment"
            ],
            "intervention_present": ["yes", "no", "not mentioned"],
            "calibration_mentioned": ["yes", "no", "not mentioned"],
            "code_available": ["yes", "no", "not mentioned", "not specified"],
        }

        properties = {}

        # Fixed enum fields → strict enum constraint
        for field_name, enum_vals in fixed_enums.items():
            properties[field_name] = {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"enum": enum_vals}
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["values", "reasoning"]
            }

        # MIDAS structured_vocab-sourced fields → enum with free-text fallback
        for field_name in ["data_used", "intervention_types", "model_type"]:
            ontology_terms = self.get_enum_values(field_name)
            if ontology_terms:
                properties[field_name] = {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"enum": ontology_terms},
                                    {"type": "string"}
                                ]
                            },
                            "description": (
                                f"Prefer these MIDAS structured_vocab terms: "
                                f"{', '.join(ontology_terms[:15])}. "
                                f"Use free text only if none match."
                            )
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["values", "reasoning"]
                }

        # Free-text fields
        free_fields = [
            "pathogen_name", "disease_name", "host_species",
            "primary_population", "population_setting_type",
            "geographic_units", "data_source",
            "calibration_techniques", "key_outcome_measures",
            "study_dates_start", "study_dates_end",
        ]
        for field_name in free_fields:
            if field_name not in properties:
                properties[field_name] = {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["values", "reasoning"]
                }

        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys())
        }

    def build_synonym_map(self) -> dict[str, dict[str, str]]:
        """Build synonym → canonical term map for post-normalization."""
        syn_map: dict[str, dict[str, str]] = {}

        for field_name, entries in self._field_terms.items():
            field_syns: dict[str, str] = {}
            for entry in entries:
                canonical = entry["alt_terms"][0] if entry["alt_terms"] else entry["label"]
                # Map all terms to canonical
                for term in entry["all_terms"]:
                    if term.lower() != canonical.lower():
                        field_syns[term.lower()] = canonical
            if field_syns:
                syn_map[field_name] = field_syns

        return syn_map

    def build_prompt_section(self) -> str:
        """Build a prompt section listing preferred structured_vocab terms."""
        lines = [
            "",
            "═══════════════════════════════════════════════",
            "MIDAS ONTOLOGY — PREFERRED VOCABULARY",
            "═══════════════════════════════════════════════",
            "",
            "When extracting values for the following fields,",
            "PREFER these exact terms from the MIDAS structured_vocab.",
            "Only use free text if no structured_vocab term matches.",
            "",
        ]

        for field_name in ["data_used", "intervention_types", "model_type"]:
            terms = self.get_enum_values(field_name)
            if terms:
                lines.append(f"  {field_name}:")
                for t in terms:
                    lines.append(f"    - \"{t}\"")
                lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """Print summary of extracted vocabulary."""
        lines = ["MIDAS Ontology Vocabulary Summary", "=" * 40]
        for field_name, entries in sorted(self._field_terms.items()):
            terms = self.get_enum_values(field_name)
            lines.append(f"\n{field_name}: {len(terms)} terms")
            for t in terms[:10]:
                lines.append(f"  - {t}")
            if len(terms) > 10:
                lines.append(f"  ... and {len(terms) - 10} more")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    owl_path = sys.argv[1] if len(sys.argv) > 1 else "midas-data.owl"
    print(f"Parsing: {owl_path}")

    vocab = MIDASVocabulary.from_owl(owl_path)
    print(vocab.summary())

    # Write outputs
    schema = vocab.build_json_schema()
    Path("midas_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    print("\nWrote: midas_schema.json")

    synonyms = vocab.build_synonym_map()
    Path("midas_synonyms.json").write_text(
        json.dumps(synonyms, indent=2), encoding="utf-8"
    )
    print("Wrote: midas_synonyms.json")

    prompt = vocab.build_prompt_section()
    Path("midas_prompt_vocab.txt").write_text(prompt, encoding="utf-8")
    print("Wrote: midas_prompt_vocab.txt")