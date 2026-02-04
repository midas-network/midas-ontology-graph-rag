from typing import List, Dict, Any


def format_paper_attributes(paper: Dict[str, Any]) -> str:
    """Format the target attributes from a paper into a readable block."""
    attributes = [
        'model_type',
        'model_determinism',
        'pathogen_name',
        'pathogen_type',
        'host_species',
        'primary_population',
        'population_setting_type',
        'geographic_scope',
        'geographic_units',
        'historical_vs_hypothetical',
        'study_goal_category',
        'intervention_present',
        'intervention_types',
        'data_used',
        'key_outcome_measures',
        'code_available'
    ]

    lines: List[str] = []
    for attr in attributes:
        value = paper.get(attr, "unspecified")
        if isinstance(value, list):
            value = f"[{', '.join(str(v) for v in value)}]"
        lines.append(f"  {attr}: {value}")

    return "\n".join(lines)


def format_paper_examples(papers: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """Format a few papers as few-shot examples for the LLM."""
    examples: List[str] = []

    for i, paper in enumerate(papers[:num_examples], 1):
        title = paper.get("title", paper.get("paper_title", "Untitled"))
        abstract = paper.get("abstract", paper.get("paper_abstract", ""))
        if isinstance(abstract, list):
            abstract = " ".join(abstract)

        example = f"""
=== Example {i} ===
Title: {title}
Abstract: {abstract[:500]}{'...' if len(abstract) > 500 else ''}

Extracted Attributes:
{format_paper_attributes(paper)}
"""
        examples.append(example)

    return "\n".join(examples)
