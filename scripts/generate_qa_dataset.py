#!/usr/bin/env python3
"""
Generate QA dataset from Knowledge Graph for LoRA training.

Implements 5 question types:
1. Gene → Phenotype (direct causation)
2. Phenotype → Gene (reverse lookup)
3. Clinical Parameter interpretation
4. Multi-hop reasoning
5. Comparative questions

Usage:
    python scripts/generate_qa_dataset.py \
        --kg_path data/kg/biological_kg.pt \
        --entity2id_path data/kg/entity2id.json \
        --output data/qa_dataset.json \
        --num_samples 10000
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QAGenerator:
    """Generate question-answer pairs from biological knowledge graph."""
    
    # Entity and relation types from KG Builder
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
    
    # Reverse mapping
    ID2RELATION = {v: k for k, v in RELATION_TYPES.items()}
    
    def __init__(self, kg_path: str, entity2id_path: str, metadata_path: Optional[str] = None):
        """
        Initialize QA generator.
        
        Args:
            kg_path: Path to biological_kg.pt
            entity2id_path: Path to entity2id.json
            metadata_path: Optional path to entity_metadata.json
        """
        logger.info("Loading knowledge graph...")
        self.kg_data = torch.load(kg_path)
        
        logger.info("Loading entity mappings...")
        with open(entity2id_path, 'r') as f:
            self.entity2id = json.load(f)
        
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        
        # Load metadata if available
        self.entity_metadata = {}
        if metadata_path and Path(metadata_path).exists():
            logger.info("Loading entity metadata...")
            with open(metadata_path, 'r') as f:
                self.entity_metadata = json.load(f)
        
        # Build graph index for fast lookup
        logger.info("Building graph indices...")
        self._build_indices()
        
        logger.info(f"KG loaded: {len(self.entity2id)} entities, "
                   f"{self.kg_data.edge_index.size(1)} edges")
    
    def _build_indices(self):
        """Build indices for fast graph queries."""
        edge_index = self.kg_data.edge_index
        edge_type = self.kg_data.edge_type
        
        # Outgoing edges: entity -> [(relation, target), ...]
        self.outgoing = defaultdict(list)
        # Incoming edges: entity -> [(relation, source), ...]
        self.incoming = defaultdict(list)
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            rel = edge_type[i].item()
            
            self.outgoing[src].append((rel, dst))
            self.incoming[dst].append((rel, src))
        
        # Entity type index
        self.entities_by_type = defaultdict(list)
        if hasattr(self.kg_data, 'entity_type'):
            for ent_id, ent_type in enumerate(self.kg_data.entity_type.tolist()):
                self.entities_by_type[ent_type].append(ent_id)
        else:
            # Infer from entity names
            for entity, ent_id in self.entity2id.items():
                if entity.startswith("MP:"):
                    self.entities_by_type[self.ENTITY_TYPES["phenotype"]].append(ent_id)
                elif entity.startswith("GO:"):
                    self.entities_by_type[self.ENTITY_TYPES["go_term"]].append(ent_id)
                elif "_" in entity or entity.endswith("_Cascade"):
                    self.entities_by_type[self.ENTITY_TYPES["pathway"]].append(ent_id)
                elif entity[0].isupper() and len(entity) < 20:
                    self.entities_by_type[self.ENTITY_TYPES["gene"]].append(ent_id)
                else:
                    self.entities_by_type[self.ENTITY_TYPES["tissue"]].append(ent_id)
    
    def _get_entity_name(self, entity_id: int) -> str:
        """Get human-readable entity name."""
        entity = self.id2entity.get(entity_id, f"Entity_{entity_id}")
        
        # Get metadata if available
        if str(entity_id) in self.entity_metadata:
            meta = self.entity_metadata[str(entity_id)]
            if "name" in meta:
                return meta["name"]
        
        # Clean up entity name
        if entity.startswith("MP:"):
            return entity  # Keep phenotype IDs
        elif entity.startswith("GO:"):
            return entity
        else:
            return entity.replace("_", " ")
    
    def _find_path(self, start: int, end: int, max_length: int = 4) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find path between two entities using BFS.
        
        Returns:
            List of (src, relation, dst) tuples forming the path, or None
        """
        if start == end:
            return []
        
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            curr, path = queue.pop(0)
            
            if len(path) >= max_length:
                continue
            
            for rel, neighbor in self.outgoing[curr]:
                if neighbor in visited:
                    continue
                
                new_path = path + [(curr, rel, neighbor)]
                
                if neighbor == end:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return None
    
    def generate_gene_phenotype_qa(self, num_samples: int) -> List[Dict]:
        """Generate Gene → Phenotype questions."""
        logger.info(f"Generating {num_samples} Gene → Phenotype QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        
        if not genes:
            logger.warning("No genes found in KG!")
            return []
        
        for _ in tqdm(range(num_samples), desc="Gene→Phenotype"):
            gene_id = random.choice(genes)
            gene_name = self._get_entity_name(gene_id)
            
            # Find phenotypes caused by this gene
            phenotypes = []
            for rel, target in self.outgoing[gene_id]:
                if rel == self.RELATION_TYPES["causes"]:
                    phenotypes.append(target)
            
            if not phenotypes:
                continue
            
            pheno_id = random.choice(phenotypes)
            pheno_name = self._get_entity_name(pheno_id)
            
            # Generate question
            question = f"What phenotypes are associated with {gene_name} knockout?"
            
            # Generate answer with reasoning
            # Find intermediate pathway/mechanism
            mechanisms = []
            for rel, intermediate in self.outgoing[gene_id]:
                if rel in [self.RELATION_TYPES["regulates"], 
                          self.RELATION_TYPES["participates_in"],
                          self.RELATION_TYPES["has_function"]]:
                    mechanisms.append((rel, intermediate))
            
            if mechanisms:
                rel, mech_id = random.choice(mechanisms)
                mech_name = self._get_entity_name(mech_id)
                rel_name = self.ID2RELATION[rel].replace("_", " ")
                
                answer = (f"{gene_name} knockout results in {pheno_name}. "
                         f"This occurs because {gene_name} {rel_name} {mech_name}, "
                         f"and disruption of this process leads to the observed phenotype.")
            else:
                answer = f"{gene_name} knockout is associated with {pheno_name}."
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "gene_to_phenotype",
                "entities": [gene_name, pheno_name],
                "gene": gene_name,
                "phenotype": pheno_name
            })
        
        return qa_pairs
    
    def generate_phenotype_gene_qa(self, num_samples: int) -> List[Dict]:
        """Generate Phenotype → Gene questions."""
        logger.info(f"Generating {num_samples} Phenotype → Gene QA pairs...")
        
        qa_pairs = []
        phenotypes = self.entities_by_type.get(self.ENTITY_TYPES["phenotype"], [])
        
        if not phenotypes:
            logger.warning("No phenotypes found in KG!")
            return []
        
        for _ in tqdm(range(num_samples), desc="Phenotype→Gene"):
            pheno_id = random.choice(phenotypes)
            pheno_name = self._get_entity_name(pheno_id)
            
            # Find genes that cause this phenotype
            genes = []
            for rel, source in self.incoming[pheno_id]:
                if rel == self.RELATION_TYPES["causes"]:
                    genes.append(source)
            
            if not genes:
                continue
            
            if len(genes) == 1:
                gene_name = self._get_entity_name(genes[0])
                question = f"Which gene knockout causes {pheno_name}?"
                answer = f"{gene_name} knockout causes {pheno_name}."
            else:
                gene_names = [self._get_entity_name(g) for g in genes[:3]]
                question = f"Which genes are associated with {pheno_name}?"
                answer = (f"Multiple genes are associated with {pheno_name}, including "
                         f"{', '.join(gene_names[:-1])}, and {gene_names[-1]}.")
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "phenotype_to_gene",
                "entities": [pheno_name] + [self._get_entity_name(g) for g in genes[:3]],
                "phenotype": pheno_name
            })
        
        return qa_pairs
    
    def generate_clinical_parameter_qa(self, num_samples: int) -> List[Dict]:
        """Generate clinical parameter interpretation questions."""
        logger.info(f"Generating {num_samples} Clinical Parameter QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        
        # Clinical parameters (simulated - in real data this would come from annotations)
        clinical_params = {
            "ALT": "alanine aminotransferase, a liver enzyme",
            "AST": "aspartate aminotransferase, a liver enzyme",
            "creatinine": "a kidney function marker",
            "glucose": "blood sugar level",
            "albumin": "a serum protein",
            "BUN": "blood urea nitrogen, a kidney function marker"
        }
        
        for _ in tqdm(range(num_samples), desc="Clinical"):
            gene_id = random.choice(genes)
            gene_name = self._get_entity_name(gene_id)
            param_name, param_desc = random.choice(list(clinical_params.items()))
            
            # Find tissues affected by this gene
            tissues = []
            for rel, target in self.outgoing[gene_id]:
                if rel == self.RELATION_TYPES["expressed_in"]:
                    tissues.append(target)
            
            question = f"What is the significance of elevated {param_name} in {gene_name} knockout mice?"
            
            if tissues:
                tissue_name = self._get_entity_name(random.choice(tissues))
                answer = (f"Elevated {param_name} ({param_desc}) in {gene_name} knockout suggests "
                         f"dysfunction in {tissue_name}. {gene_name} is normally expressed in "
                         f"{tissue_name}, and its loss leads to cellular damage, resulting in "
                         f"elevated {param_name} levels.")
            else:
                answer = (f"Elevated {param_name} ({param_desc}) in {gene_name} knockout may "
                         f"indicate tissue damage or metabolic dysfunction caused by loss of "
                         f"{gene_name} function.")
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "clinical_parameter",
                "entities": [gene_name, param_name],
                "gene": gene_name,
                "parameter": param_name
            })
        
        return qa_pairs
    
    def generate_multihop_qa(self, num_samples: int) -> List[Dict]:
        """Generate multi-hop reasoning questions."""
        logger.info(f"Generating {num_samples} Multi-hop Reasoning QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        tissues = self.entities_by_type.get(self.ENTITY_TYPES["tissue"], [])
        
        if not tissues:
            logger.warning("No tissues found in KG!")
            return []
        
        for _ in tqdm(range(num_samples), desc="Multi-hop"):
            gene_id = random.choice(genes)
            tissue_id = random.choice(tissues)
            
            # Try to find path
            path = self._find_path(gene_id, tissue_id, max_length=4)
            
            if not path or len(path) < 2:
                continue
            
            gene_name = self._get_entity_name(gene_id)
            tissue_name = self._get_entity_name(tissue_id)
            
            question = f"How does {gene_name} knockout affect {tissue_name}?"
            
            # Build answer from path
            answer_parts = [f"{gene_name} knockout affects {tissue_name} through the following pathway:"]
            for i, (src, rel, dst) in enumerate(path, 1):
                src_name = self._get_entity_name(src)
                dst_name = self._get_entity_name(dst)
                rel_name = self.ID2RELATION[rel].replace("_", " ")
                answer_parts.append(f"{i}. {src_name} {rel_name} {dst_name}")
            
            answer = " ".join(answer_parts)
            
            entity_names = [self._get_entity_name(src) for src, _, _ in path]
            entity_names.append(self._get_entity_name(path[-1][2]))
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "multihop_reasoning",
                "entities": entity_names,
                "gene": gene_name,
                "tissue": tissue_name,
                "path_length": len(path)
            })
        
        return qa_pairs
    
    def generate_comparative_qa(self, num_samples: int) -> List[Dict]:
        """Generate comparative questions between two genes."""
        logger.info(f"Generating {num_samples} Comparative QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        
        for _ in tqdm(range(num_samples), desc="Comparative"):
            if len(genes) < 2:
                continue
            
            gene1_id, gene2_id = random.sample(genes, 2)
            gene1_name = self._get_entity_name(gene1_id)
            gene2_name = self._get_entity_name(gene2_id)
            
            # Find phenotypes for each
            pheno1 = set()
            for rel, target in self.outgoing[gene1_id]:
                if rel == self.RELATION_TYPES["causes"]:
                    pheno1.add(target)
            
            pheno2 = set()
            for rel, target in self.outgoing[gene2_id]:
                if rel == self.RELATION_TYPES["causes"]:
                    pheno2.add(target)
            
            if not pheno1 or not pheno2:
                continue
            
            # Find shared and unique phenotypes
            shared = pheno1 & pheno2
            unique1 = pheno1 - pheno2
            unique2 = pheno2 - pheno1
            
            question = f"Compare the knockout phenotypes of {gene1_name} and {gene2_name}."
            
            answer_parts = []
            if shared:
                shared_names = [self._get_entity_name(p) for p in list(shared)[:2]]
                answer_parts.append(f"Both {gene1_name} and {gene2_name} knockout show {', '.join(shared_names)}.")
            
            if unique1:
                unique1_name = self._get_entity_name(list(unique1)[0])
                answer_parts.append(f"{gene1_name} knockout uniquely shows {unique1_name}.")
            
            if unique2:
                unique2_name = self._get_entity_name(list(unique2)[0])
                answer_parts.append(f"{gene2_name} knockout uniquely shows {unique2_name}.")
            
            if not answer_parts:
                answer = f"{gene1_name} and {gene2_name} have distinct knockout phenotypes."
            else:
                answer = " ".join(answer_parts)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "comparative",
                "entities": [gene1_name, gene2_name],
                "gene1": gene1_name,
                "gene2": gene2_name
            })
        
        return qa_pairs
    
    def generate_dataset(
        self,
        total_samples: int,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05
    ) -> Dict[str, List[Dict]]:
        """
        Generate complete QA dataset with train/val/test splits.
        
        Args:
            total_samples: Total number of QA pairs to generate
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        
        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        # Distribution across question types
        type_distribution = {
            "gene_to_phenotype": 0.30,
            "phenotype_to_gene": 0.20,
            "clinical_parameter": 0.25,
            "multihop_reasoning": 0.15,
            "comparative": 0.10
        }
        
        all_qa_pairs = []
        
        # Generate each type
        all_qa_pairs.extend(self.generate_gene_phenotype_qa(
            int(total_samples * type_distribution["gene_to_phenotype"])
        ))
        all_qa_pairs.extend(self.generate_phenotype_gene_qa(
            int(total_samples * type_distribution["phenotype_to_gene"])
        ))
        all_qa_pairs.extend(self.generate_clinical_parameter_qa(
            int(total_samples * type_distribution["clinical_parameter"])
        ))
        all_qa_pairs.extend(self.generate_multihop_qa(
            int(total_samples * type_distribution["multihop_reasoning"])
        ))
        all_qa_pairs.extend(self.generate_comparative_qa(
            int(total_samples * type_distribution["comparative"])
        ))
        
        # Shuffle
        random.shuffle(all_qa_pairs)
        
        # Split
        n = len(all_qa_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        dataset = {
            "train": all_qa_pairs[:train_end],
            "val": all_qa_pairs[train_end:val_end],
            "test": all_qa_pairs[val_end:]
        }
        
        logger.info(f"\nDataset statistics:")
        logger.info(f"  Total: {n} QA pairs")
        logger.info(f"  Train: {len(dataset['train'])} ({train_ratio*100:.1f}%)")
        logger.info(f"  Val: {len(dataset['val'])} ({val_ratio*100:.1f}%)")
        logger.info(f"  Test: {len(dataset['test'])} ({test_ratio*100:.1f}%)")
        
        # Type distribution
        type_counts = defaultdict(int)
        for qa in all_qa_pairs:
            type_counts[qa["type"]] += 1
        
        logger.info(f"\nQuestion type distribution:")
        for qtype, count in sorted(type_counts.items()):
            logger.info(f"  {qtype}: {count} ({count/n*100:.1f}%)")
        
        return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate QA dataset from KG")
    
    parser.add_argument("--kg_path", type=str, required=True,
                       help="Path to biological_kg.pt")
    parser.add_argument("--entity2id_path", type=str, required=True,
                       help="Path to entity2id.json")
    parser.add_argument("--metadata_path", type=str, default=None,
                       help="Path to entity_metadata.json (optional)")
    parser.add_argument("--output", type=str, default="data/qa_dataset.json",
                       help="Output JSON file path")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of QA pairs to generate")
    parser.add_argument("--train_ratio", type=float, default=0.85,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                       help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create generator
    generator = QAGenerator(
        kg_path=args.kg_path,
        entity2id_path=args.entity2id_path,
        metadata_path=args.metadata_path
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        total_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info("✓ Dataset generation complete!")
    logger.info(f"\nTo use this dataset:")
    logger.info(f"  python scripts/stage3_train_lora.py --qa_dataset {output_path}")


if __name__ == "__main__":
    main()
