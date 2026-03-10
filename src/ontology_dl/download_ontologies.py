#!/usr/bin/env python3
"""Download epidemiology-relevant ontologies.

This script downloads ontologies commonly used for infectious disease
modeling metadata extraction and standardization.

Usage:
    midas-download-ontologies --output-dir ./resources/ontologies
    midas-download-ontologies --skip ncbi_taxonomy mesh
    midas-download-ontologies --only apollo_sv ido doid
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Ontology definitions
ONTOLOGIES: dict[str, dict[str, Any]] = {
    "midas_data": {
        "name": "MIDAS Data Ontology",
        "best_for": "MIDAS modeling metadata standards",
        "url": "https://raw.githubusercontent.com/midas-network/midas-data/main/midas-data.owl",
        "filename": "midas-resources.owl",
        "format": "owl",
        "extract": False,
        "notes": "MIDAS Network resources types and modeling metadata ontology",
        "alternative_urls": [
            "https://raw.githubusercontent.com/midas-network/midas-data/master/midas-data.owl"
        ],
    },
    "ncbi_taxonomy": {
        "name": "NCBI Taxonomy Database",
        "best_for": "Pathogens, host species",
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz",
        "filename": "taxdump.tar.gz",
        "format": "tar.gz",
        "extract": True,
        "notes": "NCBI Taxonomy dump files",
    },
    "apollo_sv": {
        "name": "Apollo Structured Vocabulary",
        "best_for": "Epidemic modeling concepts",
        "url": "http://purl.obolibrary.org/obo/apollo_sv.owl",
        "filename": "apollo_sv.owl",
        "format": "owl",
        "extract": False,
        "notes": "Epidemic modeling and simulation vocabulary",
        "alternative_urls": [
            "https://raw.githubusercontent.com/ApolloDev/apollo-sv/master/src/ontology/apollo_sv.owl"
        ],
    },
    "ido": {
        "name": "Infectious Disease Ontology",
        "best_for": "Disease processes, transmission",
        "url": "http://purl.obolibrary.org/obo/ido.owl",
        "filename": "ido.owl",
        "format": "owl",
        "extract": False,
        "notes": "Core infectious disease concepts",
        "alternative_urls": [
            "https://raw.githubusercontent.com/infectious-disease-ontology/infectious-disease-ontology/master/src/ontology/ido.owl"
        ],
    },
    "doid": {
        "name": "Human Disease Ontology",
        "best_for": "Disease names",
        "url": "http://purl.obolibrary.org/obo/doid.owl",
        "filename": "doid.owl",
        "format": "owl",
        "extract": False,
        "notes": "Comprehensive disease nomenclature",
        "alternative_urls": [
            "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.owl"
        ],
    },
    "vo": {
        "name": "Vaccine Ontology",
        "best_for": "Vaccination, vaccines",
        "url": "http://purl.obolibrary.org/obo/vo.owl",
        "filename": "vo.owl",
        "format": "owl",
        "extract": False,
        "notes": "Vaccine types and vaccination processes",
        "alternative_urls": [
            "https://raw.githubusercontent.com/vaccineontology/VO/master/src/VO_merged.owl"
        ],
    },
    "gaz": {
        "name": "Gazetteer",
        "best_for": "Geographic locations",
        "url": "http://purl.obolibrary.org/obo/gaz.obo",
        "filename": "gaz.obo",
        "format": "obo",
        "extract": False,
        "notes": "Geographic feature vocabulary (OBO format)",
        "alternative_urls": [
            "https://raw.githubusercontent.com/EnvironmentOntology/gaz/master/gaz.obo"
        ],
    },
    "stato": {
        "name": "Statistical Methods Ontology",
        "best_for": "Statistical techniques",
        "url": "http://purl.obolibrary.org/obo/stato.owl",
        "filename": "stato.owl",
        "format": "owl",
        "extract": False,
        "notes": "Statistical methods and measures",
    },
    "obi": {
        "name": "Ontology for Biomedical Investigations",
        "best_for": "Study designs, methods",
        "url": "http://purl.obolibrary.org/obo/obi.owl",
        "filename": "obi.owl",
        "format": "owl",
        "extract": False,
        "notes": "Biomedical investigation terminology",
    },
    "iso_3166": {
        "name": "ISO 3166 Country Codes",
        "best_for": "Country identifiers",
        "url": "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.json",
        "filename": "iso_3166_countries.json",
        "format": "json",
        "extract": False,
        "notes": "Country codes and names",
    },
    "geonames": {
        "name": "GeoNames Geographic Database",
        "best_for": "Sub-national geography",
        "url": "http://download.geonames.org/export/dump/countryInfo.txt",
        "filename": "geonames_countries.txt",
        "format": "txt",
        "extract": False,
        "notes": "GeoNames country and geographic information",
    },
    "mesh": {
        "name": "Medical Subject Headings (MeSH)",
        "best_for": "Medical/epidemiological terms, statistical methods",
        "url": "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2025/mesh2025.nt.gz",
        "filename": "mesh.nt",
        "format": "nt",
        "extract": False,
        "notes": "NLM's controlled vocabulary in RDF format. Large file (~1.2GB).",
        "alternative_urls": [
            "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/mesh.nt.gz",
            "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2024/mesh2024.nt.gz",
        ],
    },
}


def download_file(
    url: str,
    output_path: Path,
    alternative_urls: list[str] | None = None,
) -> bool:
    """Download a file from URL to output path.

    Args:
        url: URL to download from.
        output_path: Path to save file.
        alternative_urls: List of alternative URLs to try if main URL fails.

    Returns:
        True if successful, False otherwise.
    """
    urls_to_try = [url]
    if alternative_urls:
        urls_to_try.extend(alternative_urls)

    for attempt_num, try_url in enumerate(urls_to_try, 1):
        try:
            if attempt_num > 1:
                print(f"  Trying alternative URL #{attempt_num - 1}...")

            print(f"  Downloading from {try_url}")

            response = requests.get(
                try_url,
                stream=True,
                timeout=300,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OntologyDownloader/1.0)"},
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (8192 * 100) == 0:
                            percent = (downloaded / total_size) * 100
                            print(f"     Progress: {percent:.1f}%", end="\r")

            file_size = output_path.stat().st_size
            print(f"  Downloaded {file_size:,} bytes to {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            if attempt_num < len(urls_to_try):
                print(f"  Download failed: {e}")
                continue
            print(f"  All download attempts failed. Last error: {e}")
            return False

    return False


def extract_tar_gz(tar_path: Path, output_dir: Path, cleanup: bool = False) -> bool:
    """Extract a tar.gz file to output directory.

    Args:
        tar_path: Path to tar.gz file.
        output_dir: Directory to extract to.
        cleanup: Whether to remove tar.gz after successful extraction.

    Returns:
        True if successful, False otherwise.
    """
    try:
        print(f"  Extracting {tar_path} to {output_dir}")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

        print("  Extracted successfully")

        if cleanup:
            tar_path.unlink()
            print("  Cleanup complete")

        return True

    except OSError as e:
        print(f"  Extraction failed: {e}")
        return False


def download_ontology(
    ontology_id: str,
    config: dict[str, Any],
    output_dir: Path,
    *,
    no_extract: bool = False,
    cleanup: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    """Download a single ontology.

    Args:
        ontology_id: Ontology identifier.
        config: Ontology configuration dictionary.
        output_dir: Base output directory.
        no_extract: Skip extraction of tar.gz files.
        cleanup: Remove tar.gz after successful extraction.
        force: Force re-download even if file already exists.

    Returns:
        Dictionary with download metadata, or None if failed.
    """
    print(f"\n{'=' * 60}")
    print(f"{config['name']}")
    print(f"   Best for: {config['best_for']}")
    print(f"={'=' * 60}")

    ontology_dir = output_dir / ontology_id
    ontology_dir.mkdir(parents=True, exist_ok=True)

    output_path = ontology_dir / config["filename"]

    if output_path.exists() and not force:
        file_size = output_path.stat().st_size
        print(f"  Already exists: {output_path} ({file_size:,} bytes)")
        print("  Skipping download (use --force to re-download)")

        return {
            "ontology_id": ontology_id,
            "name": config["name"],
            "best_for": config["best_for"],
            "url": config["url"],
            "filename": config["filename"],
            "format": config["format"],
            "download_path": str(output_path),
            "download_timestamp": None,
            "file_size": file_size,
            "notes": config.get("notes", ""),
            "skipped": True,
        }

    alternative_urls = config.get("alternative_urls")
    success = download_file(config["url"], output_path, alternative_urls=alternative_urls)

    if not success:
        return None

    if config.get("extract", False) and not no_extract:
        if config["format"] == "tar.gz" or str(output_path).endswith(".tar.gz"):
            extract_tar_gz(output_path, ontology_dir, cleanup=cleanup)

    return {
        "ontology_id": ontology_id,
        "name": config["name"],
        "best_for": config["best_for"],
        "url": config["url"],
        "filename": config["filename"],
        "format": config["format"],
        "download_path": str(output_path),
        "download_timestamp": datetime.now().isoformat(),
        "file_size": output_path.stat().st_size,
        "notes": config.get("notes", ""),
        "skipped": False,
    }


def save_download_manifest(downloads: list[dict], output_dir: Path) -> None:
    """Save a manifest of all downloaded ontologies."""
    manifest = {
        "download_date": datetime.now().isoformat(),
        "total_ontologies": len(downloads),
        "ontologies": downloads,
    }

    manifest_path = output_dir / "download_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownload manifest saved to {manifest_path}")


def main() -> None:
    """Main download script."""
    import argparse

    parser = argparse.ArgumentParser(description="Download epidemiology-relevant ontologies")
    parser.add_argument(
        "--output-dir",
        default="./resources/ontologies",
        help="Output directory for downloaded ontologies (default: ./resources/ontologies)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Skip specific ontologies (e.g., --skip ncbi_taxonomy gaz)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Download only specific ontologies (e.g., --only apollo_sv ido)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download but do not extract tar.gz files",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove tar.gz files after successful extraction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ONTOLOGY DOWNLOAD TOOL")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine which ontologies to download
    if args.only:
        to_download = {k: v for k, v in ONTOLOGIES.items() if k in args.only}
    else:
        skip_list = args.skip or []
        to_download = {k: v for k, v in ONTOLOGIES.items() if k not in skip_list}

    print(f"\nWill download {len(to_download)} ontologies")

    downloads = []
    successful = 0
    failed = 0
    skipped = 0

    for ontology_id, config in to_download.items():
        metadata = download_ontology(
            ontology_id,
            config,
            output_dir,
            no_extract=args.no_extract,
            cleanup=args.cleanup,
            force=args.force,
        )

        if metadata:
            downloads.append(metadata)
            if metadata.get("skipped"):
                skipped += 1
            else:
                successful += 1
        else:
            failed += 1

        time.sleep(1)

    if downloads:
        save_download_manifest(downloads, output_dir)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Downloaded: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
