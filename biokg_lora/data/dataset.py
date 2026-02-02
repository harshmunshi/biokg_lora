"""
PyTorch datasets for KG training and QA fine-tuning.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Data class (same as kg_builder for consistency, avoids PyG dependency)
class Data:
    """Simple data container for knowledge graph"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def num_nodes(self):
        if hasattr(self, 'x') and self.x is not None:
            return self.x.size(0)
        elif hasattr(self, 'edge_index') and self.edge_index is not None:
            return int(self.edge_index.max()) + 1
        return 0


class KGDataset(Dataset):
    """
    Dataset for Knowledge Graph link prediction training.
    
    Returns triples (head, relation, tail) with negative samples.
    """
    
    def __init__(
        self,
        kg_path: str,
        entity2id_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        neg_sample_size: int = 128,
        seed: int = 42,
    ):
        """
        Args:
            kg_path: Path to KG PyG Data file
            entity2id_path: Path to entity2id.json
            split: "train", "val", or "test"
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            neg_sample_size: Number of negative samples per positive
            seed: Random seed
        """
        self.kg_path = kg_path
        self.split = split
        self.neg_sample_size = neg_sample_size
        
        # Load KG (weights_only=False needed for PyG Data objects)
        self.kg_data = torch.load(kg_path, weights_only=False)
        
        # Load entity mappings
        with open(entity2id_path) as f:
            self.entity2id = json.load(f)
        
        self.num_entities = self.kg_data.num_nodes
        self.num_relations = self.kg_data.edge_type.max().item() + 1
        
        # Convert to triples
        edge_index = self.kg_data.edge_index.T  # (num_edges, 2)
        edge_type = self.kg_data.edge_type  # (num_edges,)
        
        self.triples = torch.cat([
            edge_index[:, 0:1],  # head
            edge_type.unsqueeze(1),  # relation
            edge_index[:, 1:2],  # tail
        ], dim=1)  # (num_edges, 3)
        
        # Split into train/val/test
        num_triples = len(self.triples)
        indices = np.random.RandomState(seed).permutation(num_triples)
        
        train_end = int(train_ratio * num_triples)
        val_end = train_end + int(val_ratio * num_triples)
        
        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
        
        logger.info(f"KGDataset {split}: {len(self.indices)} triples, "
                   f"{self.num_entities} entities, {self.num_relations} relations")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - head: (,) head entity ID
                - relation: (,) relation type ID
                - tail: (,) tail entity ID
                - negative_sample: (neg_sample_size,) negative entity IDs
                - mode: str, "head-batch" or "tail-batch"
        """
        triple_idx = self.indices[idx]
        head, relation, tail = self.triples[triple_idx]
        
        # Generate negative samples
        # Randomly corrupt either head or tail
        mode = random.choice(["head-batch", "tail-batch"])
        
        negative_sample = torch.randint(
            0, self.num_entities, (self.neg_sample_size,)
        )
        
        return {
            "head": head,
            "relation": relation,
            "tail": tail,
            "negative_sample": negative_sample,
            "mode": mode,
        }


class QADataset(Dataset):
    """
    Dataset for Question-Answering with entity annotations.
    
    Used for Stage 3 (LoRA fine-tuning).
    """
    
    def __init__(
        self,
        qa_pairs_path: str,
        entity2id_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
    ):
        """
        Args:
            qa_pairs_path: Path to QA pairs JSON file
            entity2id_path: Path to entity2id.json
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
            split: "train", "val", or "test"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load QA pairs
        with open(qa_pairs_path) as f:
            data = json.load(f)
        
        # Filter by split
        self.qa_pairs = [item for item in data if item.get("split", "train") == split]
        
        # Load entity mappings
        with open(entity2id_path) as f:
            self.entity2id = json.load(f)
        
        logger.info(f"QADataset {split}: {len(self.qa_pairs)} samples")
    
    def __len__(self) -> int:
        return len(self.qa_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a QA sample.
        
        Returns:
            Dict with:
                - input_ids: (max_length,) token IDs
                - attention_mask: (max_length,) attention mask
                - labels: (max_length,) target token IDs
                - entity_spans: List of (start, end, entity_id) tuples
        """
        item = self.qa_pairs[idx]
        
        question = item["question"]
        answer = item["answer"]
        entities = item.get("entities", [])
        
        # Format as prompt
        prompt = f"Question: {question}\n\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Labels for causal LM (shift by 1)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Entity spans (for KG augmentation)
        entity_spans = []
        for entity_name, start_char, end_char in entities:
            if entity_name in self.entity2id:
                # Convert char positions to token positions (approximate)
                # This is simplified; in practice, use tokenizer.encode_plus with return_offsets_mapping
                entity_id = self.entity2id[entity_name]
                entity_spans.append((0, 0, entity_id))  # Placeholder
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_spans": entity_spans,
        }


def collate_kg_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for KG dataset.
    
    Args:
        batch: List of samples from KGDataset
    
    Returns:
        Batched dict
    """
    return {
        "head": torch.stack([item["head"] for item in batch]),
        "relation": torch.stack([item["relation"] for item in batch]),
        "tail": torch.stack([item["tail"] for item in batch]),
        "negative_sample": torch.stack([item["negative_sample"] for item in batch]),
        "mode": batch[0]["mode"],  # Assume same mode for all in batch
    }


def collate_qa_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for QA dataset.
    
    Args:
        batch: List of samples from QADataset
    
    Returns:
        Batched dict
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "entity_spans": [item["entity_spans"] for item in batch],
    }


if __name__ == "__main__":
    # Test KG dataset
    from biokg_lora.data.kg_builder import create_dummy_kg
    
    # Create dummy KG
    kg_data, metadata = create_dummy_kg(num_genes=100, num_phenotypes=50)
    
    # Save temporarily
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        kg_path = Path(tmpdir) / "test_kg.pt"
        entity2id_path = Path(tmpdir) / "entity2id.json"
        
        torch.save(kg_data, kg_path)
        with open(entity2id_path, "w") as f:
            json.dump(metadata["entity2id"], f)
        
        # Create dataset
        dataset = KGDataset(
            kg_path=str(kg_path),
            entity2id_path=str(entity2id_path),
            split="train",
        )
        
        # Test __getitem__
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Head: {sample['head']}")
        print(f"Relation: {sample['relation']}")
        print(f"Tail: {sample['tail']}")
        print(f"Negative sample shape: {sample['negative_sample'].shape}")
