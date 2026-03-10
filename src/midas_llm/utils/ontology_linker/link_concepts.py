from __future__ import annotations

import difflib
import logging
import os
from typing import Any

import numpy as np

from .ontology_lookup import lookup_in_all_ontologies
from ..config import ExtractionConfig

SentenceTransformer: Any | None = None
_SENTENCE_TRANSFORMERS_AVAILABLE: bool | None = None
_EMBED_MODEL: Any | None = None
_EMBED_MODEL_NAME: str | None = None
_EMBED_MODEL_FAILED_NAMES: set[str] = set()

LOGGER = logging.getLogger("midas-llm")


def _get_embedder(model_name: str | None = None) -> Any | None:
    global SentenceTransformer, _SENTENCE_TRANSFORMERS_AVAILABLE
    global _EMBED_MODEL, _EMBED_MODEL_NAME, _EMBED_MODEL_FAILED_NAMES

    if _SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
            SentenceTransformer = _SentenceTransformer
            _SENTENCE_TRANSFORMERS_AVAILABLE = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "sentence-transformers unavailable in ontology linker: %s. "
                "Falling back to difflib.",
                exc,
            )
            _SENTENCE_TRANSFORMERS_AVAILABLE = False
            SentenceTransformer = None

    if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
        return None

    # Use config default if no model_name provided
    if model_name is None:
        config = ExtractionConfig()
        model_name = config.embedding_model

    # Prepend "sentence-transformers/" if not already present
    if model_name and "/" not in model_name and not model_name.startswith("sentence-transformers/"):
        model_name = f"sentence-transformers/{model_name}"

    if model_name in _EMBED_MODEL_FAILED_NAMES:
        return None

    # Reload model if model name changed
    if _EMBED_MODEL is None or _EMBED_MODEL_NAME != model_name:
        try:
            _EMBED_MODEL = SentenceTransformer(
                model_name,
                model_kwargs={"use_safetensors": True},
            )
            _EMBED_MODEL_NAME = model_name
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failed to load sentence-transformer '%s' with safetensors: %s. "
                "Caching this failure for the current process.",
                model_name,
                exc,
            )
            if model_name:
                _EMBED_MODEL_FAILED_NAMES.add(model_name)
            _EMBED_MODEL = None
            _EMBED_MODEL_NAME = None
    return _EMBED_MODEL


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_definition_similarity(
    llm_definition: str | None,
    ontology_definition: str | None,
) -> tuple[float | None, str]:
    """Compute similarity between LLM concept text and ontology definition."""
    if not llm_definition or not ontology_definition:
        return None, "none"

    embedder = _get_embedder()
    if embedder is not None:
        try:
            vecs = embedder.encode([llm_definition, ontology_definition], convert_to_numpy=True, normalize_embeddings=False)
            score = _cosine_sim(vecs[0], vecs[1])
            score01 = (score + 1.0) / 2.0
            return round(score01, 3), "embeddings"
        except Exception:  # noqa: BLE001
            pass

    score = difflib.SequenceMatcher(None, llm_definition.lower(), ontology_definition.lower()).ratio()
    return round(score, 3), "difflib"


def link_concepts_to_ontologies(
    extracted_data: dict[str, Any],
    ontologies: dict[str, Any],
    *,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, Any]]:
    """Link extracted concepts to ontology terms across all loaded ontologies."""
    logger.info("LINKING CONCEPTS TO ALL ONTOLOGIES")
    logger.info("NOTE: Searching ALL loaded ontologies for each term.")

    lookup_results = []

    skip_values = {
        'unspecified', 'unknown', 'not specified', 'n/a', 'none', 'no', 'yes',
        'not available', 'not applicable', 'not mentioned', 'unclear', 'not stated',
        'not reported', 'not given', 'not provided', 'not described'
    }

    for attr, data in extracted_data.items():
        value = data['value']
        recommended_onts = data.get('ontologies', [])

        value_lower = value.lower().strip()
        if value_lower in skip_values:
            logger.info("[%s]: %s - Skipping (non-informative value)", attr, value)
            continue

        logger.info("[%s]: %s", attr, value[:50] + ("..." if len(value) > 50 else ""))
        if recommended_onts and any(recommended_onts) and recommended_onts != ['None']:
            logger.info("Ignoring LLM recommendations; searching all ontologies")
        logger.info("Searching ALL ontologies")

        clean_value = value.strip()
        if clean_value.startswith('[') and clean_value.endswith(']'):
            clean_value = clean_value[1:-1]
        clean_value = clean_value.strip('"\'')
        llm_definition = data.get('definition')

        def _handle_match(match_value: str):
            matches = lookup_in_all_ontologies(
                value=match_value,
                taxonomy=ontologies['taxonomy'],
                iso_countries=ontologies['iso_countries'],
                apollo_graph=ontologies['apollo_graph'],
                midas_graph=ontologies['midas_graph'],
                doid_graph=ontologies['doid_graph'],
                ido_graph=ontologies['ido_graph'],
                obi_graph=ontologies['obi_graph'],
                stato_graph=ontologies['stato_graph'],
                vo_graph=ontologies['vo_graph'],
                gaz_graph=ontologies['gaz_graph'],
            )

            if matches:
                for match in matches:
                    sim, sim_method = compute_definition_similarity(llm_definition, match.get('definition'))
                    result = {**match, 'attribute': attr, 'value': match_value}
                    result['provenance'] = data.get('provenance')
                    result['definition'] = llm_definition
                    result['reasoning'] = data.get('reasoning')
                    result['domains'] = data.get('domains', [])
                    result['recommended_ontologies'] = recommended_onts
                    result['llm_definition'] = llm_definition
                    result['ontology_definition'] = match.get('definition')
                    result['similarity'] = sim
                    result['similarity_method'] = sim_method
                    lookup_results.append(result)
                    logger.info("      FOUND: %s: %s (%s)", match['ontology'], match['identifier'], match['matched_term'])
                    logger.info("         LLM definition: %s", llm_definition)
                    logger.info("         Ontology definition: %s", match.get('definition') if match.get('definition') else 'N/A')
                    if sim is not None:
                        logger.info("         Similarity score: %s (method: %s)", sim, sim_method)
            else:
                logger.info("      NOT FOUND in any ontology")

        if ',' in clean_value:
            items = [item.strip().strip('"\'"') for item in clean_value.split(',')]
            for item in items:
                if not item or item.lower().strip() in skip_values:
                    continue
                logger.info("      Searching for: %s", item)
                _handle_match(item)
        else:
            _handle_match(value)

    return lookup_results
