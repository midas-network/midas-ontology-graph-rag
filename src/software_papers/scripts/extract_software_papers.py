#!/usr/bin/env python3
"""
Script to extract paperID, title, and pmcID from downloaded paper data
where pubmedMeshTerms contains "Software".
"""

import json
import glob
from pathlib import Path
from typing import Any


def extract_software_papers(data_dir: Path) -> list[dict[str, Any]]:
    """
    Process all midasquery_papers-*.json files and extract papers
    where pubmedMeshTerms contains "Software".

    Returns a list of dicts with paperID, title, and pmcID.
    """
    results = []
    json_files = sorted(glob.glob(str(data_dir / "midasquery_papers-*.json")))

    print(f"Found {len(json_files)} JSON files to process")

    # Debug: Print structure of first file
    first_file_checked = False

    for filepath in json_files:
        print(f"Processing: {Path(filepath).name}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Error parsing {filepath}: {e}")
            continue

        # Handle both single objects and lists of papers
        papers = data if isinstance(data, list) else [data]

        # Debug: Find and print a paper with non-empty mesh terms
        if not first_file_checked and papers:
            for p in papers:
                if isinstance(p, dict) and p.get("paperPubMedMeshTerms"):
                    print(f"\n  DEBUG - Sample paper with mesh terms:")
                    print(f"  DEBUG - paperPubMedMeshTerms: {p.get('paperPubMedMeshTerms')[:3]}...")
                    first_file_checked = True
                    break

        for paper in papers:
            if not isinstance(paper, dict):
                continue

            # Check for paperPubMedMeshTerms containing "Software"
            mesh_terms = paper.get("paperPubMedMeshTerms", [])

            # Handle both string and list formats
            if isinstance(mesh_terms, str):
                has_software = "Software" in mesh_terms
            elif isinstance(mesh_terms, list):
                has_software = any(
                    "Software" in str(term)
                    for term in mesh_terms
                )
            else:
                has_software = False

            if has_software:
                result = {
                    "paperID": paper.get("paperID"),
                    "title": paper.get("title"),
                    "pmcid": paper.get("pmcid"),
                }
                results.append(result)

    return results


def main():
    # Path to the downloaded paper data - use repo root relative paths
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = repo_root / "downloaded_data" / "paper_data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print(f"Searching in: {data_dir}\n")

    results = extract_software_papers(data_dir)

    print(f"\n{'='*60}")
    print(f"Found {len(results)} papers with 'Software' in pubmedMeshTerms")
    print(f"{'='*60}\n")

    # Print results
    for i, paper in enumerate(results, 1):
        print(f"{i}. Paper ID: {paper['paperID']}")
        print(f"   Title: {paper['title']}")
        print(f"   PMC ID: {paper['pmcid']}")
        print()

    # Optionally save to a file
    output_path = repo_root / "output" / "software_papers.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

