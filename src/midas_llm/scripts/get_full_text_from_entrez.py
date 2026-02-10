#!/usr/bin/env python3
"""
Simple script to fetch full text from NCBI Entrez for a single PMC ID.
"""

import requests
import xml.etree.ElementTree as ET

PMCID = "PMC12111387"  # numeric part only

url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
params = {
    "db": "pmc",
    "id": PMCID,
    "rettype": "xml",
    "email": "test@example.com",  # NCBI asks you to provide an email
}

print(f"Fetching PMC{PMCID}...")
resp = requests.get(url, params=params)
resp.raise_for_status()

# Parse the XML
root = ET.fromstring(resp.text)

# Check if we got an article back
article = root.find(".//article")
if article is None:
    print("No full-text article found. It may not be open access.")
    print("Raw response (first 500 chars):")
    print(resp.text[:500])
else:
    # Extract title
    title_el = article.find(".//article-title")
    title = "".join(title_el.itertext()) if title_el is not None else "N/A"
    print(f"Title: {title}\n")

    # Extract abstract
    abstract_el = article.find(".//abstract")
    if abstract_el is not None:
        abstract_text = " ".join("".join(p.itertext()).strip() for p in abstract_el.findall(".//p"))
        print(f"Abstract:\n{abstract_text[:500]}...\n")

    # Extract body paragraphs
    body = article.find(".//body")
    if body is not None:
        paragraphs = body.findall(".//p")
        print(f"Body: {len(paragraphs)} paragraphs found\n")
        # Print first 3 paragraphs as a sample
        for i, p in enumerate(paragraphs[:3000]):
            text = "".join(p.itertext()).strip()
            print(f"--- Paragraph {i+1} ---")
            print(text[:300])
            print()
    else:
        print("No body text found.")

    # Show total XML size
    print(f"Total XML response size: {len(resp.text):,} characters")

