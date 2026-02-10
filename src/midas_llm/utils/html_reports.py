"""
HTML report generation utilities for concept extraction and ontology mapping.
"""
from datetime import datetime
from typing import Dict, List


def generate_html_report(
    extracted_data: Dict[str, Dict],
    lookup_results: List[Dict],
    llm_response: str,
    output_path: str = "extraction_report.html"
):
    """
    Generate an HTML report showing extracted concepts and their ontology mappings.

    Args:
        extracted_data: Parsed LLM response (attribute -> {value, citation, reasoning} mapping)
        lookup_results: Results from ontology lookups
        llm_response: Raw LLM response text
        output_path: Where to save the HTML file
    """
    # Create lookup map for easy access
    lookup_map = {r['attribute']: r for r in lookup_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Concept Extraction & Ontology Mapping Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }}
        h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #3498db; }}
        .timestamp {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        thead {{ background: #34495e; color: white; }}
        th {{ padding: 15px; text-align: left; font-weight: 600; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
        tbody tr:hover {{ background: #f8f9fa; }}
        .status-found {{ color: #27ae60; font-weight: bold; }}
        .status-not-found {{ color: #e74c3c; }}
        .status-unspecified {{ color: #95a5a6; font-style: italic; }}
        .identifier {{ font-family: 'Courier New', monospace; background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        .ontology-badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 500; }}
        .ncbi {{ background: #e3f2fd; color: #1565c0; }}
        .iso {{ background: #f3e5f5; color: #6a1b9a; }}
        .midas {{ background: #fff3e0; color: #e65100; }}
        .doid {{ background: #e8f5e9; color: #2e7d32; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .raw-response {{ background: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 4px solid #3498db; margin: 20px 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em; max-height: 400px; overflow-y: auto; }}
        details {{ margin: 20px 0; }}
        summary {{ cursor: pointer; font-weight: bold; padding: 10px; background: #ecf0f1; border-radius: 5px; }}
        summary:hover {{ background: #d5dbdb; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Concept Extraction & Ontology Mapping Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{len(extracted_data)}</div>
                <div class="stat-label">Concepts Extracted</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(1 for r in lookup_results if r['status'] == 'found')}</div>
                <div class="stat-label">Ontology Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(r['ontology'] for r in lookup_results if r['ontology']))}</div>
                <div class="stat-label">Ontologies Used</div>
            </div>
        </div>
        
        <h2>📋 Extracted Concepts & Ontology Mappings</h2>
        <table>
            <thead>
                <tr>
                    <th>Attribute</th>
                    <th>Modeling Domain(s)</th>
                    <th>Extracted Value</th>
                    <th>Provenance (Source)</th>
                    <th>Concept Definition</th>
                    <th>Reasoning</th>
                    <th>Ontology</th>
                    <th>Identifier</th>
                    <th>Matched Term</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

    for attr, data in sorted(extracted_data.items()):
        value = data.get('value', data if isinstance(data, str) else '')
        provenance = data.get('provenance', '') if isinstance(data, dict) else ''
        concept = data.get('concept', '') if isinstance(data, dict) else ''
        reasoning = data.get('reasoning', '') if isinstance(data, dict) else ''
        domains = data.get('domains', []) if isinstance(data, dict) else []

        lookup = lookup_map.get(attr, {})
        ontology = lookup.get('ontology', 'N/A')
        identifier = lookup.get('identifier', '-')
        matched = lookup.get('matched_term', '-')
        status = lookup.get('status', 'unknown')

        # Style the ontology badge
        ontology_class = ''
        if 'NCBI' in str(ontology):
            ontology_class = 'ncbi'
        elif 'ISO' in str(ontology) or 'GeoNames' in str(ontology):
            ontology_class = 'iso'
        elif 'MIDAS' in str(ontology) or 'Apollo' in str(ontology):
            ontology_class = 'midas'
        elif 'DOID' in str(ontology) or 'Disease' in str(ontology):
            ontology_class = 'doid'

        # Style the status
        status_class = ''
        status_text = status
        if status == 'found':
            status_class = 'status-found'
            status_text = '✓ Found'
        elif status == 'not_found':
            status_class = 'status-not-found'
            status_text = '✗ Not Found'
        elif status == 'unspecified':
            status_class = 'status-unspecified'
            status_text = 'Unspecified'

        ontology_badge = f'<span class="ontology-badge {ontology_class}">{ontology}</span>' if ontology != 'N/A' else 'N/A'
        identifier_display = f'<span class="identifier">{identifier}</span>' if identifier != '-' else '-'

        # Format domains, provenance, concept, and reasoning for display
        domains_display = ', '.join(domains) if domains else '-'
        provenance_display = provenance[:200] + '...' if provenance and len(provenance) > 200 else (provenance or '-')
        concept_display = concept[:250] + '...' if concept and len(concept) > 250 else (concept or '-')
        reasoning_display = reasoning[:200] + '...' if reasoning and len(reasoning) > 200 else (reasoning or '-')

        html += f"""
                <tr>
                    <td><strong>{attr}</strong></td>
                    <td style="font-size: 0.85em; color: #7f8c8d;">
                        {domains_display}
                    </td>
                    <td>{value[:100]}{'...' if len(value) > 100 else ''}</td>
                    <td style="font-style: italic; font-size: 0.9em; color: #555; background: #fffbf0;">
                        {provenance_display}
                    </td>
                    <td style="font-size: 0.9em; color: #2c3e50; background: #f0f7ff;">
                        {concept_display}
                    </td>
                    <td style="font-size: 0.9em; color: #666;">
                        {reasoning_display}
                    </td>
                    <td>{ontology_badge}</td>
                    <td>{identifier_display}</td>
                    <td>{matched}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
        
        <h2>🏷️ Modeling Domain Distribution</h2>
"""

    # Group by modeling domain
    domain_groups = {}
    for attr, data in extracted_data.items():
        domains = data.get('domains', []) if isinstance(data, dict) else []
        for domain in domains:
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(attr)

    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0;">'

    for domain, attrs in sorted(domain_groups.items()):
        html += f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 1em;">{domain.replace('_', ' ').title()}</h3>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
"""
        for attr in sorted(attrs):
            html += f"                <li>{attr}</li>\n"
        html += """
            </ul>
        </div>
"""

    html += '</div>'

    html += """
        
        <h2>🗺️ Ontology Usage Summary</h2>
"""

    # Group by ontology
    ontology_groups = {}
    for lookup in lookup_results:
        ont = lookup.get('ontology')
        if ont:
            if ont not in ontology_groups:
                ontology_groups[ont] = []
            ontology_groups[ont].append(lookup)

    for ontology, items in sorted(ontology_groups.items()):
        found_count = sum(1 for item in items if item['status'] == 'found')
        html += f"""
        <details open>
            <summary>{ontology} ({found_count}/{len(items)} matched)</summary>
            <table style="margin-top: 10px;">
                <thead>
                    <tr>
                        <th>Attribute</th>
                        <th>Value</th>
                        <th>Identifier</th>
                        <th>Matched Term</th>
                    </tr>
                </thead>
                <tbody>
"""
        for item in items:
            if item['status'] == 'found':
                html += f"""
                    <tr>
                        <td>{item['attribute']}</td>
                        <td>{item['value']}</td>
                        <td><span class="identifier">{item['identifier']}</span></td>
                        <td>{item['matched_term']}</td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </details>
"""

    html += f"""
        <h2>📄 Raw LLM Response</h2>
        <details>
            <summary>Click to view full LLM response</summary>
            <div class="raw-response">{llm_response}</div>
        </details>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML report generated: {output_path}")
