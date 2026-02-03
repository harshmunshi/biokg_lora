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
        self.kg_data = torch.load(kg_path, weights_only=False)
        
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
        """Generate Gene → Phenotype questions with deep mechanistic reasoning."""
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
            pheno_id_str = self.id2entity.get(pheno_id, str(pheno_id))
            
            # Collect all related information
            pathways = []
            functions = []
            tissues = []
            regulated_genes = []
            
            for rel, target in self.outgoing[gene_id]:
                if rel == self.RELATION_TYPES["participates_in"]:
                    pathways.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["has_function"]:
                    functions.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["expressed_in"]:
                    tissues.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["regulates"]:
                    regulated_genes.append(self._get_entity_name(target))
            
            # Generate question
            question = f"What phenotypes are associated with {gene_name} knockout?"
            
            # Build detailed multi-step answer
            answer_parts = [f"{gene_name} knockout results in {pheno_name}. This occurs through the following mechanism:"]
            
            step = 1
            
            # Step 1: Gene function
            if functions:
                func_name = functions[0]
                answer_parts.append(f"{step}. {gene_name} has molecular function: {func_name}")
                step += 1
            
            # Step 2: Pathway involvement
            if pathways:
                pathway_name = pathways[0]
                answer_parts.append(f"{step}. {gene_name} participates in {pathway_name}")
                step += 1
            
            # Step 3: Tissue expression
            if tissues:
                tissue_name = tissues[0]
                answer_parts.append(f"{step}. Loss of {gene_name} expression in {tissue_name} disrupts normal tissue function")
                step += 1
            
            # Step 4: Downstream effects
            if regulated_genes:
                reg_gene = regulated_genes[0]
                answer_parts.append(f"{step}. {gene_name} normally regulates {reg_gene}, and its loss causes dysregulation")
                step += 1
            
            # Step 5: Phenotype manifestation
            if pheno_id_str.startswith("MP:"):
                answer_parts.append(f"{step}. These disruptions manifest as phenotype {pheno_id_str}")
            else:
                answer_parts.append(f"{step}. These disruptions result in the observed phenotype: {pheno_name}")
            
            # Related information
            related_info = []
            if pathways:
                related_info.append(f"Related pathways: {', '.join(pathways[:3])}")
            if regulated_genes:
                related_info.append(f"Regulated genes: {', '.join(regulated_genes[:3])}")
            
            if related_info:
                answer_parts.append("\n" + "\n".join(related_info))
            
            answer = "\n".join(answer_parts)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "gene_to_phenotype",
                "entities": [gene_name, pheno_name],
                "gene": gene_name,
                "phenotype": pheno_name,
                "phenotype_id": pheno_id_str if pheno_id_str.startswith("MP:") else None,
                "pathways": pathways[:3] if pathways else [],
                "tissues": tissues[:2] if tissues else []
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
        """Generate clinical parameter interpretation questions with deep mechanistic reasoning."""
        logger.info(f"Generating {num_samples} Clinical Parameter QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        
        # Clinical parameters with tissue mapping
        clinical_params = {
            "ALT": {
                "full_name": "alanine aminotransferase",
                "type": "liver enzyme",
                "tissue": "Liver",
                "mechanism": "hepatocellular damage causes ALT release into circulation"
            },
            "AST": {
                "full_name": "aspartate aminotransferase",
                "type": "liver enzyme",
                "tissue": "Liver",
                "mechanism": "hepatic injury leads to AST leakage"
            },
            "creatinine": {
                "full_name": "creatinine",
                "type": "kidney function marker",
                "tissue": "Kidney",
                "mechanism": "impaired glomerular filtration causes creatinine accumulation"
            },
            "glucose": {
                "full_name": "glucose",
                "type": "blood sugar",
                "tissue": "Pancreas",
                "mechanism": "β-cell dysfunction impairs insulin secretion"
            },
            "albumin": {
                "full_name": "albumin",
                "type": "serum protein",
                "tissue": "Liver",
                "mechanism": "reduced hepatic synthesis decreases albumin levels"
            },
            "BUN": {
                "full_name": "blood urea nitrogen",
                "type": "kidney function marker",
                "tissue": "Kidney",
                "mechanism": "decreased renal clearance elevates BUN"
            }
        }
        
        for _ in tqdm(range(num_samples), desc="Clinical"):
            gene_id = random.choice(genes)
            gene_name = self._get_entity_name(gene_id)
            param_name = random.choice(list(clinical_params.keys()))
            param_info = clinical_params[param_name]
            
            # Find pathways, functions, and tissues
            pathways = []
            functions = []
            tissues = []
            phenotypes = []
            
            for rel, target in self.outgoing[gene_id]:
                if rel == self.RELATION_TYPES["participates_in"]:
                    pathways.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["has_function"]:
                    functions.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["expressed_in"]:
                    tissues.append(self._get_entity_name(target))
                elif rel == self.RELATION_TYPES["causes"]:
                    phenotypes.append(target)
            
            question = f"What is the significance of elevated {param_name} in {gene_name} knockout mice?"
            
            # Build detailed answer with numbered steps
            answer_parts = [f"Elevated {param_name} ({param_info['full_name']}, {param_info['type']}) in {gene_name} knockout is significant because:"]
            
            # Step 1: Gene function/location
            if functions:
                func_name = functions[0]
                answer_parts.append(f"1. {gene_name} has function: {func_name}")
            elif tissues:
                tissue_name = tissues[0]
                answer_parts.append(f"1. {gene_name} is expressed in {tissue_name}")
            else:
                answer_parts.append(f"1. {gene_name} plays a regulatory role in cellular function")
            
            # Step 2: Pathway disruption
            if pathways:
                pathway_name = pathways[0]
                answer_parts.append(f"2. Knockout disrupts {pathway_name}")
            else:
                answer_parts.append(f"2. Knockout disrupts normal cellular processes")
            
            # Step 3: Tissue damage mechanism
            answer_parts.append(f"3. This leads to {param_info['tissue'].lower()} damage where {param_info['mechanism']}")
            
            # Step 4: Phenotype connection
            if phenotypes:
                pheno_id = phenotypes[0]
                pheno_name = self._get_entity_name(pheno_id)
                if pheno_id in self.id2entity and self.id2entity[pheno_id].startswith("MP:"):
                    answer_parts.append(f"4. This connects to phenotype {self.id2entity[pheno_id]} ({pheno_name})")
                else:
                    answer_parts.append(f"4. This connects to observed phenotype: {pheno_name}")
            else:
                answer_parts.append(f"4. This manifests as elevated {param_name} in clinical chemistry")
            
            # Related information
            if pathways:
                answer_parts.append(f"\nRelated pathways: {', '.join(pathways[:3])}")
            
            answer = "\n".join(answer_parts)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "clinical_parameter",
                "entities": [gene_name, param_name],
                "gene": gene_name,
                "parameter": param_name,
                "pathways": pathways[:3] if pathways else [],
                "phenotypes": [self.id2entity.get(p, str(p)) for p in phenotypes[:2]]
            })
        
        return qa_pairs
    
    def generate_multihop_qa(self, num_samples: int) -> List[Dict]:
        """Generate multi-hop reasoning questions with detailed path explanations."""
        logger.info(f"Generating {num_samples} Multi-hop Reasoning QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        tissues = self.entities_by_type.get(self.ENTITY_TYPES["tissue"], [])
        phenotypes = self.entities_by_type.get(self.ENTITY_TYPES["phenotype"], [])
        
        # Try both tissue and phenotype targets
        targets = tissues + phenotypes
        
        if not targets:
            logger.warning("No target entities found in KG!")
            return []
        
        for _ in tqdm(range(num_samples), desc="Multi-hop"):
            gene_id = random.choice(genes)
            target_id = random.choice(targets)
            
            # Try to find path
            path = self._find_path(gene_id, target_id, max_length=4)
            
            if not path or len(path) < 2:
                continue
            
            gene_name = self._get_entity_name(gene_id)
            target_name = self._get_entity_name(target_id)
            target_type = "tissue" if target_id in tissues else "phenotype"
            
            question = f"How does {gene_name} knockout affect {target_name}?"
            
            # Build detailed answer with numbered reasoning
            answer_parts = [f"{gene_name} knockout affects {target_name} through the following mechanistic pathway:\n"]
            
            for i, (src, rel, dst) in enumerate(path, 1):
                src_name = self._get_entity_name(src)
                dst_name = self._get_entity_name(dst)
                rel_name = self.ID2RELATION[rel].replace("_", " ")
                
                # Add explanation for each step
                if rel_name == "regulates":
                    explanation = f"which controls"
                elif rel_name == "participates in":
                    explanation = f"which is involved in"
                elif rel_name == "expressed in":
                    explanation = f"where it is normally expressed in"
                elif rel_name == "causes":
                    explanation = f"leading to"
                elif rel_name == "has function":
                    explanation = f"with molecular function"
                else:
                    explanation = f"which {rel_name}"
                
                answer_parts.append(f"{i}. {src_name} {explanation} {dst_name}")
            
            # Add conclusion
            answer_parts.append(f"\nTherefore, loss of {gene_name} disrupts this {len(path)}-step pathway, ultimately affecting {target_name}.")
            
            # Find related phenotypes for context
            related_phenos = []
            for rel, pheno_id in self.outgoing[gene_id]:
                if rel == self.RELATION_TYPES["causes"]:
                    pheno_name = self.id2entity.get(pheno_id, "")
                    if pheno_name.startswith("MP:"):
                        related_phenos.append(pheno_name)
            
            if related_phenos:
                answer_parts.append(f"Related phenotypes: {', '.join(related_phenos[:2])}")
            
            answer = "\n".join(answer_parts)
            
            entity_names = [self._get_entity_name(src) for src, _, _ in path]
            entity_names.append(self._get_entity_name(path[-1][2]))
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "multihop_reasoning",
                "entities": entity_names,
                "gene": gene_name,
                "target": target_name,
                "target_type": target_type,
                "path_length": len(path)
            })
        
        return qa_pairs
    
    def generate_comparative_qa(self, num_samples: int) -> List[Dict]:
        """Generate comparative questions between two genes with detailed analysis."""
        logger.info(f"Generating {num_samples} Comparative QA pairs...")
        
        qa_pairs = []
        genes = self.entities_by_type.get(self.ENTITY_TYPES["gene"], [])
        
        for _ in tqdm(range(num_samples), desc="Comparative"):
            if len(genes) < 2:
                continue
            
            gene1_id, gene2_id = random.sample(genes, 2)
            gene1_name = self._get_entity_name(gene1_id)
            gene2_name = self._get_entity_name(gene2_id)
            
            # Collect comprehensive information for both genes
            def get_gene_info(gene_id):
                phenos = set()
                pathways = set()
                tissues = set()
                functions = set()
                
                for rel, target in self.outgoing[gene_id]:
                    if rel == self.RELATION_TYPES["causes"]:
                        phenos.add(target)
                    elif rel == self.RELATION_TYPES["participates_in"]:
                        pathways.add(target)
                    elif rel == self.RELATION_TYPES["expressed_in"]:
                        tissues.add(target)
                    elif rel == self.RELATION_TYPES["has_function"]:
                        functions.add(target)
                
                return phenos, pathways, tissues, functions
            
            pheno1, path1, tissue1, func1 = get_gene_info(gene1_id)
            pheno2, path2, tissue2, func2 = get_gene_info(gene2_id)
            
            if not pheno1 or not pheno2:
                continue
            
            # Find overlaps
            shared_phenos = pheno1 & pheno2
            shared_pathways = path1 & path2
            unique_pheno1 = pheno1 - pheno2
            unique_pheno2 = pheno2 - pheno1
            
            question = f"Compare the knockout phenotypes of {gene1_name} and {gene2_name}."
            
            # Build structured answer
            answer_parts = [f"Comparison of {gene1_name} and {gene2_name} knockout phenotypes:\n"]
            
            # Shared features
            if shared_pathways:
                pathway_names = [self._get_entity_name(p) for p in list(shared_pathways)[:2]]
                answer_parts.append(f"**Shared mechanisms**: Both genes participate in {', '.join(pathway_names)}")
            
            if shared_phenos:
                shared_names = [self._get_entity_name(p) for p in list(shared_phenos)[:2]]
                pheno_ids = [self.id2entity.get(p, "") for p in list(shared_phenos)[:2]]
                shared_str = ", ".join([f"{n} ({i})" if i.startswith("MP:") else n 
                                       for n, i in zip(shared_names, pheno_ids)])
                answer_parts.append(f"**Common phenotypes**: {shared_str}")
            
            # Gene 1 specific
            if unique_pheno1:
                unique1_names = [self._get_entity_name(p) for p in list(unique_pheno1)[:2]]
                answer_parts.append(f"\n**{gene1_name}-specific phenotypes**: {', '.join(unique1_names)}")
                if tissue1:
                    tissue1_name = self._get_entity_name(list(tissue1)[0])
                    answer_parts.append(f"  - Expressed in {tissue1_name}")
            
            # Gene 2 specific
            if unique_pheno2:
                unique2_names = [self._get_entity_name(p) for p in list(unique_pheno2)[:2]]
                answer_parts.append(f"\n**{gene2_name}-specific phenotypes**: {', '.join(unique2_names)}")
                if tissue2:
                    tissue2_name = self._get_entity_name(list(tissue2)[0])
                    answer_parts.append(f"  - Expressed in {tissue2_name}")
            
            # Conclusion
            if shared_phenos:
                answer_parts.append(f"\n**Interpretation**: While both genes show overlapping phenotypes suggesting shared pathway involvement, each gene has unique knockout effects reflecting their specific biological roles.")
            else:
                answer_parts.append(f"\n**Interpretation**: {gene1_name} and {gene2_name} have distinct knockout phenotypes, indicating they function in different biological pathways.")
            
            answer = "\n".join(answer_parts)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "type": "comparative",
                "entities": [gene1_name, gene2_name],
                "gene1": gene1_name,
                "gene2": gene2_name,
                "shared_phenotypes": len(shared_phenos),
                "shared_pathways": len(shared_pathways)
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
