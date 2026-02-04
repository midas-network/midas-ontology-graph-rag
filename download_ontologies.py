#!/usr/bin/env python3
"""
Download the latest versions of epidemiology-relevant ontologies.

This script downloads ontologies commonly used for infectious disease
modeling metadata extraction and standardization.
"""

import os
import sys
import requests
import json
from datetime import datetime
from typing import Dict, Optional
import time


# Ontology definitions
ONTOLOGIES = {
    "midas_data": {
        "name": "MIDAS Data Ontology",
        "best_for": "MIDAS modeling metadata standards",
        "url": "https://raw.githubusercontent.com/midas-network/midas-data/main/midas-data.owl",
        "filename": "midas-data.owl",
        "format": "owl",
        "extract": False,
        "notes": "MIDAS Network data types and modeling metadata ontology",
        "alternative_urls": [
            "https://raw.githubusercontent.com/midas-network/midas-data/master/midas-data.owl"
        ]
    },
    "ncbi_taxonomy": {
        "name": "NCBI Taxonomy Database",
        "best_for": "Pathogens, host species",
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz",
        "filename": "taxdump.tar.gz",
        "format": "tar.gz",
        "extract": True,
        "notes": "NCBI Taxonomy dump files"
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
        ]
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
        ]
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
        ]
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
        ]
    },
    "gaz": {
        "name": "Gazetteer",
        "best_for": "Geographic locations",
        "url": "http://purl.obolibrary.org/obo/gaz.obo",
        "filename": "gaz.obo",
        "format": "obo",
        "extract": False,
        "notes": "Geographic feature vocabulary (OBO format - more reliable than OWL)",
        "alternative_urls": [
            "http://purl.obolibrary.org/obo/gaz.obo",
            "https://raw.githubusercontent.com/EnvironmentOntology/gaz/master/gaz.obo"
        ]
    },
    "stato": {
        "name": "Statistical Methods Ontology",
        "best_for": "Statistical techniques",
        "url": "http://purl.obolibrary.org/obo/stato.owl",
        "filename": "stato.owl",
        "format": "owl",
        "extract": False,
        "notes": "Statistical methods and measures"
    },
    "obi": {
        "name": "Ontology for Biomedical Investigations",
        "best_for": "Study designs, methods",
        "url": "http://purl.obolibrary.org/obo/obi.owl",
        "filename": "obi.owl",
        "format": "owl",
        "extract": False,
        "notes": "Biomedical investigation terminology"
    },
    "iso_3166": {
        "name": "ISO 3166 Country Codes",
        "best_for": "Country identifiers",
        "url": "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.json",
        "filename": "iso_3166_countries.json",
        "format": "json",
        "extract": False,
        "notes": "Country codes and names"
    },
    "geonames": {
        "name": "GeoNames Geographic Database",
        "best_for": "Sub-national geography",
        "url": "http://download.geonames.org/export/dump/countryInfo.txt",
        "filename": "geonames_countries.txt",
        "format": "txt",
        "extract": False,
        "notes": "GeoNames country and geographic information"
    },
    "mesh": {
        "name": "Medical Subject Headings (MeSH)",
        "best_for": "Medical/epidemiological terms, statistical methods",
        "url": "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2025/mesh2025.nt.gz",
        "filename": "mesh.nt",
        "format": "nt",
        "extract": False,
        "notes": "NLM's controlled vocabulary in RDF format. Contains terms like 'Likelihood Functions', 'Models, Statistical', 'Epidemiologic Methods'. Note: Despite .nt.gz URL, file is already uncompressed NT format. Large file (~1.2GB).",
        "alternative_urls": [
            "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/mesh.nt.gz",
            "https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2024/mesh2024.nt.gz",
            "ftp://nlmpubs.nlm.nih.gov/online/mesh/rdf/mesh.nt.gz"
        ]
    }
}


def download_file(url: str, output_path: str, description: str = "", alternative_urls: list = None) -> bool:
    """
    Download a file from URL to output path.

    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress display
        alternative_urls: List of alternative URLs to try if main URL fails

    Returns:
        True if successful, False otherwise
    """
    urls_to_try = [url]
    if alternative_urls:
        urls_to_try.extend(alternative_urls)

    last_error = None

    for attempt_num, try_url in enumerate(urls_to_try, 1):
        try:
            if attempt_num > 1:
                print(f"  🔄 Trying alternative URL #{attempt_num - 1}...")
                print(f"     {try_url}")
            else:
                print(f"  📥 Downloading from {try_url}")

            # Stream download for large files, allow redirects
            response = requests.get(
                try_url,
                stream=True,
                timeout=300,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; OntologyDownloader/1.0)'}
            )
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress
            downloaded = 0
            chunk_size = 8192

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Show progress for large files
                        if total_size > 0 and downloaded % (chunk_size * 100) == 0:
                            percent = (downloaded / total_size) * 100
                            print(f"     Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='\r')

            if total_size > 0:
                print(f"     Progress: 100.0% ({total_size:,} / {total_size:,} bytes)")

            file_size = os.path.getsize(output_path)
            print(f"  ✅ Downloaded {file_size:,} bytes to {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt_num < len(urls_to_try):
                print(f"  ⚠️  Download failed: {e}")
                continue  # Try next URL
            else:
                print(f"  ❌ All download attempts failed. Last error: {e}")
                return False
        except Exception as e:
            last_error = e
            if attempt_num < len(urls_to_try):
                print(f"  ⚠️  Error: {e}")
                continue  # Try next URL
            else:
                print(f"  ❌ All download attempts failed. Last error: {e}")
                return False

    return False


def check_disk_space(path: str, required_mb: int) -> bool:
    """
    Check if there's enough disk space available.

    Args:
        path: Path to check
        required_mb: Required space in megabytes

    Returns:
        True if enough space, False otherwise
    """
    try:
        import shutil
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)

        print(f"  💾 Disk space check:")
        print(f"     Available: {available_mb:,.1f} MB")
        print(f"     Required: {required_mb:,.1f} MB")

        if available_mb < required_mb:
            print(f"  ⚠️  WARNING: Insufficient disk space!")
            return False

        print(f"  ✅ Sufficient disk space available")
        return True

    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Proceed anyway if we can't check


def extract_tar_gz(tar_path: str, output_dir: str, cleanup: bool = False) -> bool:
    """
    Extract a tar.gz file to output directory.

    Args:
        tar_path: Path to tar.gz file
        output_dir: Directory to extract to
        cleanup: Whether to remove tar.gz after successful extraction

    Returns:
        True if successful, False otherwise
    """
    try:
        import tarfile

        # Get compressed file size
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)

        # Estimate uncompressed size (typically 2-3x for text files)
        estimated_size_mb = tar_size_mb * 3

        # Check disk space
        if not check_disk_space(output_dir, estimated_size_mb):
            print(f"  💡 TIP: You can use --skip ncbi_taxonomy to skip this large download")
            print(f"  💡 Or keep the .tar.gz file and extract manually when space is available")
            return False

        print(f"  📦 Extracting {tar_path} to {output_dir}")

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)

        print(f"  ✅ Extracted successfully")

        # Cleanup if requested
        if cleanup:
            print(f"  🗑️  Removing {tar_path}")
            os.remove(tar_path)
            print(f"  ✅ Cleanup complete")

        return True

    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"  ❌ Extraction failed: DISK FULL")
            print(f"  💡 The compressed file is saved at: {tar_path}")
            print(f"  💡 You can extract it manually later when more space is available:")
            print(f"     tar -xzf {tar_path} -C {output_dir}")
            return False
        else:
            print(f"  ❌ Extraction failed: {e}")
            return False
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def extract_gz(gz_path: str, output_path: str, cleanup: bool = False) -> bool:
    """
    Extract a .gz file to output path.

    Args:
        gz_path: Path to .gz file
        output_path: Path for extracted file
        cleanup: Whether to remove .gz after successful extraction

    Returns:
        True if successful, False otherwise
    """
    try:
        import gzip
        import shutil

        print(f"  📦 Extracting {gz_path}")

        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        file_size = os.path.getsize(output_path)
        print(f"  ✅ Extracted to {output_path} ({file_size:,} bytes)")

        # Cleanup if requested
        if cleanup:
            print(f"  🗑️  Removing {gz_path}")
            os.remove(gz_path)
            print(f"  ✅ Cleanup complete")

        return True

    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def download_ontology(
    ontology_id: str,
    config: Dict,
    output_dir: str,
    no_extract: bool = False,
    cleanup: bool = False,
    force: bool = False
) -> Optional[Dict]:
    """
    Download a single ontology.

    Args:
        ontology_id: Ontology identifier
        config: Ontology configuration dictionary
        output_dir: Base output directory
        no_extract: Skip extraction of tar.gz files
        cleanup: Remove tar.gz after successful extraction
        force: Force re-download even if file already exists

    Returns:
        Dictionary with download metadata, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"📚 {config['name']}")
    print(f"   Best for: {config['best_for']}")
    print(f"   Format: {config['format']}")
    print(f"{'='*80}")

    # Create subdirectory for this ontology
    ontology_dir = os.path.join(output_dir, ontology_id)
    os.makedirs(ontology_dir, exist_ok=True)

    # Check if file already exists
    output_path = os.path.join(ontology_dir, config['filename'])

    if os.path.exists(output_path) and not force:
        file_size = os.path.getsize(output_path)
        print(f"  ✓ Already exists: {output_path} ({file_size:,} bytes)")
        print(f"  ⏭️  Skipping download (use --force to re-download)")

        # Return metadata for existing file
        metadata = {
            'ontology_id': ontology_id,
            'name': config['name'],
            'best_for': config['best_for'],
            'url': config['url'],
            'filename': config['filename'],
            'format': config['format'],
            'download_path': output_path,
            'download_timestamp': None,  # Wasn't downloaded this run
            'file_size': file_size,
            'notes': config.get('notes', ''),
            'skipped': True,
            'extraction_attempted': False,
            'extraction_successful': None
        }
        return metadata

    # Download the file
    alternative_urls = config.get('alternative_urls', None)
    success = download_file(config['url'], output_path, config['name'], alternative_urls=alternative_urls)

    if not success:
        return None

    # Extract if needed
    extraction_attempted = False
    extraction_successful = None

    if config.get('extract', False):
        if no_extract:
            print(f"  ⏭️  Skipping extraction (--no-extract flag)")
            extraction_attempted = False
        else:
            extraction_attempted = True

            if config['format'] == 'tar.gz' or output_path.endswith('.tar.gz'):
                extraction_successful = extract_tar_gz(output_path, ontology_dir, cleanup=cleanup)
            elif output_path.endswith('.gz') and not output_path.endswith('.tar.gz'):
                # Extract .gz file (like mesh.nt.gz)
                extracted_name = config['filename'].replace('.gz', '')
                extracted_path = os.path.join(ontology_dir, extracted_name)
                extraction_successful = extract_gz(output_path, extracted_path, cleanup=cleanup)
            else:
                print(f"  ⚠️  Unknown compression format, skipping extraction")
                extraction_attempted = False

    # Return metadata
    metadata = {
        'ontology_id': ontology_id,
        'name': config['name'],
        'best_for': config['best_for'],
        'url': config['url'],
        'filename': config['filename'],
        'format': config['format'],
        'download_path': output_path,
        'download_timestamp': datetime.now().isoformat(),
        'file_size': os.path.getsize(output_path),
        'notes': config.get('notes', ''),
        'extraction_attempted': extraction_attempted,
        'extraction_successful': extraction_successful
    }

    return metadata


def save_download_manifest(downloads: list, output_dir: str):
    """
    Save a manifest of all downloaded ontologies.

    Args:
        downloads: List of download metadata dictionaries
        output_dir: Output directory for manifest
    """
    manifest = {
        'download_date': datetime.now().isoformat(),
        'total_ontologies': len(downloads),
        'ontologies': downloads
    }

    manifest_path = os.path.join(output_dir, 'download_manifest.json')

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Download manifest saved to {manifest_path}")


def create_readme(output_dir: str):
    """Create a README file explaining the downloaded ontologies."""

    readme_content = """# Downloaded Ontologies

This directory contains ontologies downloaded for infectious disease modeling metadata extraction.

## Ontologies Included

### Core Infectious Disease Ontologies

**Apollo Structured Vocabulary (APOLLO_SV)**
- Best for: Epidemic modeling concepts
- Format: OWL
- URI: http://purl.obolibrary.org/obo/APOLLO_SV
- Use: Standardized terminology for epidemic modeling and simulation

**Infectious Disease Ontology (IDO)**
- Best for: Disease processes, transmission
- Format: OWL
- URI: http://purl.obolibrary.org/obo/IDO
- Use: Core concepts for infectious disease representation

**Human Disease Ontology (DOID)**
- Best for: Disease names
- Format: OWL
- URI: http://purl.obolibrary.org/obo/DOID
- Use: Comprehensive disease nomenclature

**Vaccine Ontology (VO)**
- Best for: Vaccination, vaccines
- Format: OWL
- URI: http://purl.obolibrary.org/obo/VO
- Use: Vaccine types and vaccination processes

### Taxonomic & Geographic

**NCBI Taxonomy**
- Best for: Pathogens, host species
- Format: tar.gz (dump files)
- URI: ncbi.nlm.nih.gov/Taxonomy
- Use: Standardized organism taxonomy IDs

**Gazetteer (GAZ)**
- Best for: Geographic locations
- Format: OWL
- URI: http://purl.obolibrary.org/obo/GAZ
- Use: Geographic feature vocabulary

### Methodological

**Statistical Methods Ontology (STATO)**
- Best for: Statistical techniques
- Format: OWL
- URI: http://purl.obolibrary.org/obo/STATO
- Use: Statistical methods and measures terminology

**Ontology for Biomedical Investigations (OBI)**
- Best for: Study designs, methods
- Format: OWL
- URI: http://purl.obolibrary.org/obo/OBI
- Use: Biomedical investigation methodology

### Reference Data

**ISO 3166 Country Codes**
- Best for: Country identifiers
- Format: JSON
- Use: Standardized country codes

**GeoNames**
- Best for: Sub-national geography
- Format: TXT
- Use: Geographic names and hierarchies

## File Structure

```
ontologies/
├── apollo_sv/
│   └── apollo_sv.owl
├── ido/
│   └── ido.owl
├── doid/
│   └── doid.owl
├── vo/
│   └── vo.owl
├── gaz/
│   └── gaz.owl
├── stato/
│   └── stato.owl
├── obi/
│   └── obi.owl
├── ncbi_taxonomy/
│   ├── taxdump.tar.gz
│   └── [extracted files]
├── iso_3166/
│   └── iso_3166_countries.json
├── geonames_countries/
│   └── geonames_countries.txt
└── download_manifest.json
```

## Usage

### Loading OWL Ontologies

```python
from rdflib import Graph

# Load an ontology
g = Graph()
g.parse("ontologies/apollo_sv/apollo_sv.owl", format="xml")

# Query classes
for s, p, o in g.triples((None, None, None)):
    print(s, p, o)
```

### Using NCBI Taxonomy

```python
from utils.ncbi_taxonomy import NCBITaxonomy

taxonomy = NCBITaxonomy("ontologies/ncbi_taxonomy")
results = taxonomy.lookup_common_name("human")
print(results[0]['tax_id'])  # 9606
```

### Using ISO 3166 Country Codes

```python
import json

with open("ontologies/iso_3166/iso_3166_countries.json") as f:
    countries = json.load(f)

# Find country by code
usa = next(c for c in countries if c['alpha-2'] == 'US')
print(usa['name'])  # United States of America
```

## Updating

To download the latest versions:

```bash
python download_ontologies.py --output-dir ./ontologies
```

## References

- OBO Foundry: http://obofoundry.org/
- NCBI Taxonomy: https://www.ncbi.nlm.nih.gov/taxonomy
- ISO 3166: https://www.iso.org/iso-3166-country-codes.html
- GeoNames: https://www.geonames.org/
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"📄 README created at {readme_path}")


def main():
    """Main download script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download epidemiology-relevant ontologies"
    )
    parser.add_argument(
        '--output-dir',
        default='./ontologies',
        help='Output directory for downloaded ontologies (default: ./ontologies)'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        help='Skip specific ontologies (e.g., --skip ncbi_taxonomy gaz)'
    )
    parser.add_argument(
        '--only',
        nargs='+',
        help='Download only specific ontologies (e.g., --only apollo_sv ido)'
    )
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Download but do not extract tar.gz files (saves disk space)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove tar.gz files after successful extraction'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files already exist'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("🌍 ONTOLOGY DOWNLOAD TOOL")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Determine which ontologies to download
    to_download = {}

    if args.only:
        # Download only specified ontologies
        for ont_id in args.only:
            if ont_id in ONTOLOGIES:
                to_download[ont_id] = ONTOLOGIES[ont_id]
            else:
                print(f"⚠️  Warning: Unknown ontology '{ont_id}', skipping")
    else:
        # Download all ontologies except those in skip list
        skip_list = args.skip or []
        to_download = {
            ont_id: config
            for ont_id, config in ONTOLOGIES.items()
            if ont_id not in skip_list
        }

    print(f"\n📦 Will download {len(to_download)} ontologies")

    # Download each ontology
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
            force=args.force
        )

        if metadata:
            downloads.append(metadata)
            if metadata.get('skipped'):
                skipped += 1
            else:
                successful += 1
        else:
            failed += 1

        # Be nice to servers - wait a bit between downloads
        time.sleep(1)

    # Save manifest
    if downloads:
        save_download_manifest(downloads, output_dir)
        create_readme(output_dir)

    # Print summary
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)
    print(f"✅ Downloaded: {successful}")
    print(f"⏭️  Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    if failed > 0:
        print("\n⚠️  Some downloads failed. Check the output above for details.")
        sys.exit(1)
    elif skipped > 0 and successful == 0:
        print("\n✨ All files already exist!")
        print(f"\n💡 To re-download existing files, use: --force")
        print(f"\nNext steps:")
        print(f"  1. Check {output_dir}/README.md for usage instructions")
        print(f"  2. Explore {output_dir}/download_manifest.json for details")
        print(f"  3. Use the ontologies in your extraction pipeline")
    else:
        print("\n✨ All downloads completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Check {output_dir}/README.md for usage instructions")
        print(f"  2. Explore {output_dir}/download_manifest.json for details")
        print(f"  3. Use the ontologies in your extraction pipeline")


if __name__ == "__main__":
    main()
