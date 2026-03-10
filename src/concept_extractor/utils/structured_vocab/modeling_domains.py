"""Modeling domain classification helpers."""
from __future__ import annotations
from typing import List

MODELING_DOMAINS = {
    'model_parameters': [
        'model_type', 'model_determinism', 'calibration_techniques',
        'statistical_methods', 'parameter_estimation'
    ],
    'location': [
        'geographic_scope', 'geographic_units', 'location', 'country',
        'region', 'population_setting_type'
    ],
    'pathogen_disease': [
        'pathogen_name', 'pathogen_type', 'disease_name', 'transmission_mode'
    ],
    'population': [
        'host_species', 'primary_population', 'age_groups', 'demographic_groups'
    ],
    'intervention': [
        'intervention_present', 'intervention_types', 'control_measures',
        'vaccination', 'pharmaceutical_interventions', 'non_pharmaceutical_interventions'
    ],
    'data_sources': [
        'data_used', 'data_source', 'surveillance_data', 'case_data'
    ],
    'outcomes': [
        'key_outcome_measures', 'health_outcomes', 'economic_outcomes',
        'mortality', 'morbidity', 'incidence', 'prevalence'
    ],
    'study_metadata': [
        'study_goal_category', 'study_design', 'historical_vs_hypothetical',
        'study_dates_start', 'study_dates_end', 'code_available'
    ],
}


def identify_modeling_domains(attribute: str) -> List[str]:
    """Return modeling domains for a given attribute name."""
    domains: List[str] = []

    for domain, attributes in MODELING_DOMAINS.items():
        if attribute in attributes:
            domains.append(domain)

    if not domains:
        attr_lower = attribute.lower()
        if any(kw in attr_lower for kw in ['model', 'calibration', 'parameter']):
            domains.append('model_parameters')
        if any(kw in attr_lower for kw in ['location', 'geographic', 'country', 'region']):
            domains.append('location')
        if any(kw in attr_lower for kw in ['pathogen', 'disease', 'virus', 'bacteria']):
            domains.append('pathogen_disease')
        if any(kw in attr_lower for kw in ['population', 'host', 'demographic', 'age']):
            domains.append('population')
        if any(kw in attr_lower for kw in ['intervention', 'control', 'treatment', 'vaccin']):
            domains.append('intervention')
        if any(kw in attr_lower for kw in ['resources', 'source', 'surveillance']):
            domains.append('data_sources')
        if any(kw in attr_lower for kw in ['outcome', 'mortality', 'morbidity', 'incidence', 'case']):
            domains.append('outcomes')
        if any(kw in attr_lower for kw in ['study', 'goal', 'design', 'date']):
            domains.append('study_metadata')

    if not domains:
        domains.append('other')

    return domains
