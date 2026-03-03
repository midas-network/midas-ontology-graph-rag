#!/usr/bin/env python3
"""
Script to download full text from Entrez for all papers in software_papers.json
that have a valid pmcid.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests


def fetch_full_text(pmcid: str, email: str = "test@example.com") -> Optional[dict]:
    """
    Fetch full text from NCBI Entrez for a given PMC ID.

    Returns a dict with title, abstract, and body text, or None if not available.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "email": email,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    Error fetching {pmcid}: {e}")
        return None

    # Parse the XML
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        print(f"    Error parsing XML for {pmcid}: {e}")
        return None

    # Check if we got an article back
    article = root.find(".//article")
    if article is None:
        return None

    # Extract title
    title_el = article.find(".//article-title")
    title = "".join(title_el.itertext()) if title_el is not None else ""

    # Extract abstract
    abstract_el = article.find(".//abstract")
    abstract_text = ""
    if abstract_el is not None:
        abstract_text = " ".join(
            "".join(p.itertext()).strip()
            for p in abstract_el.findall(".//p")
        )

    # Extract body paragraphs
    body_text = ""
    body = article.find(".//body")
    if body is not None:
        paragraphs = body.findall(".//p")
        body_text = "\n\n".join(
            "".join(p.itertext()).strip()
            for p in paragraphs
        )

    return {
        "title": title,
        "abstract": abstract_text,
        "body": body_text,
        "xml_size": len(resp.text),
    }


def main():
    # Paths - use repo root relative paths
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    input_path = repo_root / "output" / "software_papers.json"
    output_dir = repo_root / "output" / "fulltext"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load software papers
    print(f"Loading papers from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers total")

    # Filter to papers with valid pmcid
    papers_with_pmcid = [p for p in papers if p.get("pmcid")]
    print(f"Papers with PMC ID: {len(papers_with_pmcid)}")
    print()

    # Track results
    success_count = 0
    failed_count = 0
    skipped_count = 0
    results_summary = []

    for i, paper in enumerate(papers_with_pmcid, 1):
        paper_id = paper.get("paperID")
        pmcid = paper.get("pmcid")
        title = paper.get("title", "Unknown")

        # Output file for this paper
        output_file = output_dir / f"{paper_id}_{pmcid}.json"

        # Skip if already downloaded
        if output_file.exists():
            print(f"[{i}/{len(papers_with_pmcid)}] Skipping {pmcid} (already exists)")
            skipped_count += 1
            continue

        print(f"[{i}/{len(papers_with_pmcid)}] Fetching {pmcid} - {title[:50]}...")

        result = fetch_full_text(pmcid)

        if result:
            # Save the full text
            output_data = {
                "paperID": paper_id,
                "pmcid": pmcid,
                "original_title": title,
                **result
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"    ✓ Saved ({result['xml_size']:,} chars, {len(result['body'])} body chars)")
            success_count += 1
            results_summary.append({
                "paperID": paper_id,
                "pmcid": pmcid,
                "status": "success",
                "body_length": len(result['body'])
            })
        else:
            print(f"    ✗ No full text available (may not be open access)")
            failed_count += 1
            results_summary.append({
                "paperID": paper_id,
                "pmcid": pmcid,
                "status": "failed"
            })

        # Rate limiting: NCBI recommends no more than 3 requests per second
        time.sleep(0.4)

    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total papers processed: {len(papers_with_pmcid)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed (not open access): {failed_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Output directory: {output_dir}")

    # Save summary
    summary_path = output_dir / "_download_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(papers_with_pmcid),
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "results": results_summary
        }, f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

