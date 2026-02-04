"""
NCBI Taxonomy lookup utility for finding tax_id of species names.

This module provides functionality to search the NCBI Taxonomy database
dump files to find taxonomy IDs for species names.
"""

import os
from typing import Dict, List, Optional, Tuple


class NCBITaxonomy:
    """Handler for NCBI Taxonomy database lookups."""

    def __init__(self, taxdump_dir: str = "./data/taxdump"):
        """
        Initialize the taxonomy lookup.

        Args:
            taxdump_dir: Path to directory containing NCBI taxdump files
        """
        self.taxdump_dir = taxdump_dir
        self.names_file = os.path.join(taxdump_dir, "names.dmp")
        self.nodes_file = os.path.join(taxdump_dir, "nodes.dmp")

        # Cache for loaded data
        self._name_to_taxid: Optional[Dict[str, List[Tuple[int, str]]]] = None
        self._taxid_to_names: Optional[Dict[int, List[Tuple[str, str]]]] = None
        self._taxid_to_rank: Optional[Dict[int, str]] = None

    def _load_names(self):
        """Load the names.dmp file into memory (lazy loading)."""
        if self._name_to_taxid is not None:
            return

        print(f"Loading NCBI Taxonomy names from {self.names_file}...", flush=True)

        self._name_to_taxid = {}
        self._taxid_to_names = {}

        with open(self.names_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"\r  Loading names: {line_num:,} entries...", end='', flush=True)

                # Parse the pipe-delimited format
                parts = line.strip().split('\t|\t')
                if len(parts) < 4:
                    continue

                tax_id = int(parts[0].strip())
                name_txt = parts[1].strip()
                name_class = parts[3].replace('\t|', '').strip()

                # Index by name (lowercase for case-insensitive search)
                name_lower = name_txt.lower()
                if name_lower not in self._name_to_taxid:
                    self._name_to_taxid[name_lower] = []
                self._name_to_taxid[name_lower].append((tax_id, name_class))

                # Index by tax_id
                if tax_id not in self._taxid_to_names:
                    self._taxid_to_names[tax_id] = []
                self._taxid_to_names[tax_id].append((name_txt, name_class))

        print(f"\r✓ Loaded {len(self._name_to_taxid):,} unique names for {len(self._taxid_to_names):,} taxa" + " " * 20)

    def _load_nodes(self):
        """Load the nodes.dmp file to get rank information (lazy loading)."""
        if self._taxid_to_rank is not None:
            return

        print(f"Loading NCBI Taxonomy nodes from {self.nodes_file}...", flush=True)

        self._taxid_to_rank = {}

        with open(self.nodes_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"\r  Loading nodes: {line_num:,} entries...", end='', flush=True)

                parts = line.strip().split('\t|\t')
                if len(parts) < 3:
                    continue

                tax_id = int(parts[0].strip())
                rank = parts[2].strip()

                self._taxid_to_rank[tax_id] = rank

        print(f"\r✓ Loaded {len(self._taxid_to_rank):,} taxonomy nodes" + " " * 20)

    def lookup_species(self, species_name: str, rank_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Look up taxonomy ID(s) for a species name.

        Args:
            species_name: The species name to search for (e.g., "human", "Homo sapiens")
            rank_filter: Optional list of ranks to filter by (e.g., ["species", "genus"])

        Returns:
            List of dictionaries with keys: tax_id, name, name_class, rank
        """
        self._load_names()
        self._load_nodes()

        species_lower = species_name.lower().strip()

        if species_lower not in self._name_to_taxid:
            return []

        results = []
        for tax_id, name_class in self._name_to_taxid[species_lower]:
            rank = self._taxid_to_rank.get(tax_id, "unknown")

            # Apply rank filter if specified
            if rank_filter and rank not in rank_filter:
                continue

            results.append({
                'tax_id': tax_id,
                'name': species_name,
                'name_class': name_class,
                'rank': rank
            })

        return results

    def get_scientific_name(self, tax_id: int) -> Optional[str]:
        """
        Get the scientific name for a given taxonomy ID.

        Args:
            tax_id: The taxonomy ID

        Returns:
            The scientific name, or None if not found
        """
        self._load_names()

        if tax_id not in self._taxid_to_names:
            return None

        # Find the scientific name
        for name_txt, name_class in self._taxid_to_names[tax_id]:
            if name_class == "scientific name":
                return name_txt

        # Fallback to first name if no scientific name found
        if self._taxid_to_names[tax_id]:
            return self._taxid_to_names[tax_id][0][0]

        return None

    def lookup_common_name(self, common_name: str) -> List[Dict]:
        """
        Look up species by common name (e.g., "human", "cattle").

        Args:
            common_name: The common name to search for

        Returns:
            List of matches with tax_id, scientific_name, and rank
        """
        self._load_names()
        self._load_nodes()

        results = self.lookup_species(common_name)

        # Enhance results with scientific names
        enhanced = []
        for result in results:
            scientific = self.get_scientific_name(result['tax_id'])
            enhanced.append({
                'tax_id': result['tax_id'],
                'scientific_name': scientific,
                'common_name': common_name,
                'name_class': result['name_class'],
                'rank': result['rank']
            })

        return enhanced

    def search_fuzzy(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform a fuzzy search for species names.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of matches with tax_id, name, name_class, and rank
        """
        self._load_names()
        self._load_nodes()

        query_lower = query.lower().strip()
        results = []

        # Search for partial matches
        for name_lower, tax_entries in self._name_to_taxid.items():
            if query_lower in name_lower:
                for tax_id, name_class in tax_entries:
                    rank = self._taxid_to_rank.get(tax_id, "unknown")
                    results.append({
                        'tax_id': tax_id,
                        'name': name_lower,
                        'name_class': name_class,
                        'rank': rank
                    })

                    if len(results) >= max_results:
                        return results

        return results


def demo():
    """Demonstrate taxonomy lookup functionality."""
    taxonomy = NCBITaxonomy()

    print("\n" + "="*60)
    print("NCBI Taxonomy Lookup Demo")
    print("="*60)

    # Example 1: Look up common names
    test_species = ["human", "cattle", "mosquito", "influenza"]

    for species in test_species:
        print(f"\n🔍 Searching for: {species}")
        results = taxonomy.lookup_common_name(species)

        if results:
            print(f"   Found {len(results)} match(es):")
            for i, result in enumerate(results[:5], 1):  # Show top 5
                print(f"   {i}. tax_id={result['tax_id']:8d} | "
                      f"{result['scientific_name']:30s} | "
                      f"rank={result['rank']:15s} | "
                      f"class={result['name_class']}")
        else:
            print(f"   ❌ No matches found")

    # Example 2: Look up scientific name
    print(f"\n🔍 Searching for: Homo sapiens")
    results = taxonomy.lookup_species("Homo sapiens", rank_filter=["species"])
    if results:
        print(f"   Found {len(results)} match(es):")
        for result in results:
            print(f"   tax_id={result['tax_id']:8d} | rank={result['rank']}")

    # Example 3: Get scientific name from tax_id
    print(f"\n🔍 Getting scientific name for tax_id=9606")
    scientific = taxonomy.get_scientific_name(9606)
    print(f"   Scientific name: {scientific}")


if __name__ == "__main__":
    demo()
