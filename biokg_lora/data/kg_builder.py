"""
Knowledge Graph Builder for Biological Data.

Constructs unified knowledge graph from multiple sources:
- MGI (Mouse Genome Informatics)
- GO (Gene Ontology)
- KEGG (Pathways)
- STRING (Protein interactions)
- MPO (Mammalian Phenotype Ontology)
- GTEx (Tissue expression)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalKGBuilder:
    """
    Builder for constructing biological knowledge graphs.
    
    Entity Types:
        - gene: Gene symbols (e.g., Thbd, Bmp4)
        - pathway: Biological pathways (e.g., Coagulation_Cascade)
        - go_term: Gene Ontology terms
        - phenotype: Mammalian phenotypes (MP terms)
        - tissue: Tissue/organ types
        - protein: Protein IDs
    
    Relation Types:
        - regulates / regulated_by
        - part_of / has_part
        - expressed_in / expresses
        - causes / caused_by
        - interacts_with (symmetric)
        - located_in / location_of
        - has_function / function_of
        - participates_in
        - associated_with
    """
    
    ENTITY_TYPES = {
        "gene": 0,
        "pathway": 1,
        "go_term": 2,
        "phenotype": 3,
        "tissue": 4,
        "protein": 5,
    }
    
    RELATION_TYPES = {
        "regulates": 0,
        "regulated_by": 1,
        "part_of": 2,
        "has_part": 3,
        "expressed_in": 4,
        "expresses": 5,
        "causes": 6,
        "caused_by": 7,
        "interacts_with": 8,
        "located_in": 9,
        "location_of": 10,
        "has_function": 11,
        "function_of": 12,
        "participates_in": 13,
        "associated_with": 14,
    }
    
    def __init__(self):
        """Initialize empty KG builder."""
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.entity_types: Dict[int, int] = {}
        self.entity_metadata: Dict[int, Dict] = {}
        
        self.triples: List[Tuple[int, int, int]] = []
        self.relation2id = self.RELATION_TYPES
        
        self.stats = defaultdict(int)
        
        logger.info("Initialized BiologicalKGBuilder")
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add an entity to the KG.
        
        Args:
            name: Entity name (e.g., "Thbd", "GO:0007596")
            entity_type: One of ENTITY_TYPES keys
            metadata: Optional metadata dict
        
        Returns:
            entity_id: Unique integer ID
        """
        if name in self.entity2id:
            return self.entity2id[name]
        
        entity_id = len(self.entity2id)
        self.entity2id[name] = entity_id
        self.id2entity[entity_id] = name
        self.entity_types[entity_id] = self.ENTITY_TYPES[entity_type]
        
        if metadata:
            self.entity_metadata[entity_id] = metadata
        
        self.stats[f"entity_{entity_type}"] += 1
        
        return entity_id
    
    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        add_inverse: bool = False
    ):
        """
        Add a triple (head, relation, tail) to the KG.
        
        Args:
            head: Head entity name
            relation: Relation type
            tail: Tail entity name
            add_inverse: If True, also add inverse relation
        """
        if head not in self.entity2id or tail not in self.entity2id:
            logger.warning(f"Entities not found: {head} or {tail}")
            return
        
        if relation not in self.relation2id:
            logger.warning(f"Unknown relation: {relation}")
            return
        
        head_id = self.entity2id[head]
        tail_id = self.entity2id[tail]
        relation_id = self.relation2id[relation]
        
        self.triples.append((head_id, relation_id, tail_id))
        self.stats[f"relation_{relation}"] += 1
        
        # Add inverse relation if specified
        if add_inverse:
            inverse_map = {
                "regulates": "regulated_by",
                "part_of": "has_part",
                "expressed_in": "expresses",
                "causes": "caused_by",
                "located_in": "location_of",
                "has_function": "function_of",
            }
            if relation in inverse_map:
                inverse_rel = inverse_map[relation]
                inverse_id = self.relation2id[inverse_rel]
                self.triples.append((tail_id, inverse_id, head_id))
                self.stats[f"relation_{inverse_rel}"] += 1
    
    def build(self) -> Data:
        """
        Build PyTorch Geometric Data object.
        
        Returns:
            PyG Data object with:
                - edge_index: (2, num_edges)
                - edge_type: (num_edges,)
                - entity_type: (num_entities,)
                - num_nodes: int
        """
        logger.info(f"Building PyG Data object from {len(self.triples)} triples...")
        
        # Convert triples to edge_index and edge_type
        edge_index = torch.tensor(
            [(h, t) for h, r, t in self.triples],
            dtype=torch.long
        ).T  # (2, num_edges)
        
        edge_type = torch.tensor(
            [r for h, r, t in self.triples],
            dtype=torch.long
        )  # (num_edges,)
        
        # Entity types
        entity_type = torch.tensor(
            [self.entity_types[i] for i in range(len(self.entity2id))],
            dtype=torch.long
        )  # (num_entities,)
        
        kg_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            entity_type=entity_type,
            num_nodes=len(self.entity2id),
        )
        
        logger.info(f"Built KG: {kg_data.num_nodes} nodes, {kg_data.edge_index.size(1)} edges")
        
        return kg_data
    
    def save(self, output_dir: str, kg_data: Data):
        """
        Save KG data and metadata.
        
        Args:
            output_dir: Directory to save files
            kg_data: PyG Data object
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyG data
        torch.save(kg_data, output_dir / "biological_kg.pt")
        logger.info(f"Saved KG data to {output_dir / 'biological_kg.pt'}")
        
        # Save entity mappings
        with open(output_dir / "entity2id.json", "w") as f:
            json.dump(self.entity2id, f, indent=2)
        
        with open(output_dir / "id2entity.json", "w") as f:
            json.dump(self.id2entity, f, indent=2)
        
        with open(output_dir / "entity_metadata.json", "w") as f:
            json.dump(self.entity_metadata, f, indent=2)
        
        # Save statistics
        with open(output_dir / "kg_stats.json", "w") as f:
            json.dump(dict(self.stats), f, indent=2)
        
        logger.info(f"Saved metadata to {output_dir}")
    
    def print_stats(self):
        """Print KG statistics."""
        print("\n" + "="*60)
        print("Knowledge Graph Statistics")
        print("="*60)
        
        print(f"\nTotal Entities: {len(self.entity2id):,}")
        for entity_type in self.ENTITY_TYPES:
            count = self.stats.get(f"entity_{entity_type}", 0)
            pct = 100 * count / len(self.entity2id) if len(self.entity2id) > 0 else 0
            print(f"  {entity_type:12s}: {count:6,} ({pct:5.1f}%)")
        
        print(f"\nTotal Triples: {len(self.triples):,}")
        for relation in self.RELATION_TYPES:
            count = self.stats.get(f"relation_{relation}", 0)
            pct = 100 * count / len(self.triples) if len(self.triples) > 0 else 0
            print(f"  {relation:18s}: {count:7,} ({pct:5.1f}%)")
        
        print("="*60 + "\n")


def create_dummy_kg(
    num_genes: int = 100,
    num_phenotypes: int = 50,
    num_go_terms: int = 200,
    num_pathways: int = 20,
    num_tissues: int = 10,
    avg_edges_per_gene: int = 5,
    seed: int = 42
) -> Tuple[Data, Dict]:
    """
    Create a dummy knowledge graph for testing.
    
    Args:
        num_genes: Number of genes
        num_phenotypes: Number of phenotypes
        num_go_terms: Number of GO terms
        num_pathways: Number of pathways
        num_tissues: Number of tissues
        avg_edges_per_gene: Average edges per gene
        seed: Random seed
    
    Returns:
        kg_data: PyG Data object
        metadata: Dict with entity2id, id2entity
    """
    np.random.seed(seed)
    builder = BiologicalKGBuilder()
    
    logger.info("Creating dummy KG...")
    
    # Add genes
    genes = [f"Gene{i:04d}" for i in range(num_genes)]
    for gene in genes:
        builder.add_entity(gene, "gene")
    
    # Add phenotypes
    phenotypes = [f"MP:{i:07d}" for i in range(num_phenotypes)]
    for pheno in phenotypes:
        builder.add_entity(pheno, "phenotype")
    
    # Add GO terms
    go_terms = [f"GO:{i:07d}" for i in range(num_go_terms)]
    for go_term in go_terms:
        builder.add_entity(go_term, "go_term")
    
    # Add pathways
    pathways = [f"Pathway_{i}" for i in range(num_pathways)]
    for pathway in pathways:
        builder.add_entity(pathway, "pathway")
    
    # Add tissues
    tissues = [f"Tissue_{i}" for i in range(num_tissues)]
    for tissue in tissues:
        builder.add_entity(tissue, "tissue")
    
    # Add random edges
    for gene in tqdm(genes, desc="Adding edges"):
        # Gene -> Phenotype (causes)
        num_phenos = np.random.poisson(2)
        for pheno in np.random.choice(phenotypes, min(num_phenos, len(phenotypes)), replace=False):
            builder.add_triple(gene, "causes", pheno, add_inverse=True)
        
        # Gene -> GO term (has_function)
        num_go = np.random.poisson(3)
        for go in np.random.choice(go_terms, min(num_go, len(go_terms)), replace=False):
            builder.add_triple(gene, "has_function", go, add_inverse=True)
        
        # Gene -> Pathway (participates_in)
        num_paths = np.random.poisson(1)
        for pathway in np.random.choice(pathways, min(num_paths, len(pathways)), replace=False):
            builder.add_triple(gene, "participates_in", pathway)
        
        # Gene -> Tissue (expressed_in)
        num_tiss = np.random.poisson(2)
        for tissue in np.random.choice(tissues, min(num_tiss, len(tissues)), replace=False):
            builder.add_triple(gene, "expressed_in", tissue, add_inverse=True)
        
        # Gene -> Gene (regulates)
        if np.random.rand() < 0.3:
            other_gene = np.random.choice([g for g in genes if g != gene])
            builder.add_triple(gene, "regulates", other_gene, add_inverse=True)
    
    # Add GO hierarchy
    for i in range(num_go_terms // 10):
        child = go_terms[np.random.randint(num_go_terms)]
        parent = go_terms[np.random.randint(num_go_terms)]
        if child != parent:
            builder.add_triple(child, "part_of", parent, add_inverse=True)
    
    builder.print_stats()
    
    kg_data = builder.build()
    
    metadata = {
        "entity2id": builder.entity2id,
        "id2entity": builder.id2entity,
        "relation2id": builder.relation2id,
        "entity_types": BiologicalKGBuilder.ENTITY_TYPES,
    }
    
    return kg_data, metadata


if __name__ == "__main__":
    # Test dummy KG creation
    kg_data, metadata = create_dummy_kg(
        num_genes=100,
        num_phenotypes=50,
        num_go_terms=200
    )
    
    print(f"\nDummy KG created successfully!")
    print(f"Nodes: {kg_data.num_nodes}")
    print(f"Edges: {kg_data.edge_index.size(1)}")
