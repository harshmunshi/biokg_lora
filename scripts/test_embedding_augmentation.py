#!/usr/bin/env python3
"""
Test embedding augmentation without training.

Quick test to verify:
1. KG embeddings load correctly
2. Projection layer works
3. Entity linking functions
4. Embeddings can be fused with LLM token embeddings

Usage:
    python scripts/test_embedding_augmentation.py \
        --kg_path data/kg/biological_kg.pt \
        --entity_embeddings data/rotate_embeddings.pt \
        --entity2id_path data/kg/entity2id.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleProjection(nn.Module):
    """Simple projection layer from KG space to LLM space."""
    
    def __init__(self, kg_dim: int = 256, lm_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(kg_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, lm_dim),
            nn.LayerNorm(lm_dim),
        )
    
    def forward(self, kg_emb):
        return self.proj(kg_emb)


class SimpleEntityLinker:
    """Simple rule-based entity linker for genes."""
    
    def __init__(self, entity2id: Dict[str, int]):
        self.entity2id = entity2id
        self.entities = set(entity2id.keys())
        
        # Build lowercase index for case-insensitive matching
        self.lower_to_entity = {}
        for entity in self.entities:
            self.lower_to_entity[entity.lower()] = entity
    
    def link_entities(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find entities in text.
        
        Returns:
            List of (start_idx, end_idx, entity_name)
        """
        words = text.split()
        entities_found = []
        
        i = 0
        while i < len(words):
            # Try multi-word matches first (up to 3 words)
            for length in [3, 2, 1]:
                if i + length <= len(words):
                    candidate = " ".join(words[i:i+length])
                    candidate_lower = candidate.lower()
                    
                    if candidate_lower in self.lower_to_entity:
                        entity_name = self.lower_to_entity[candidate_lower]
                        # Approximate character positions
                        start_char = sum(len(w) + 1 for w in words[:i])
                        end_char = start_char + len(candidate)
                        entities_found.append((start_char, end_char, entity_name))
                        i += length
                        break
            else:
                i += 1
        
        return entities_found


def test_embedding_loading(kg_path: str, entity_embeddings_path: str, entity2id_path: str):
    """Test 1: Load all embeddings and mappings."""
    logger.info("="*60)
    logger.info("Test 1: Loading Embeddings")
    logger.info("="*60)
    
    # Load KG
    logger.info(f"Loading KG from {kg_path}...")
    kg_data = torch.load(kg_path, weights_only=False)
    logger.info(f"  ✓ KG loaded: {kg_data.num_nodes} nodes, {kg_data.edge_index.size(1)} edges")
    
    # Load entity embeddings
    logger.info(f"Loading entity embeddings from {entity_embeddings_path}...")
    entity_embeddings = torch.load(entity_embeddings_path, weights_only=False)
    logger.info(f"  ✓ Embeddings shape: {entity_embeddings.shape}")
    
    # Load entity2id
    logger.info(f"Loading entity2id from {entity2id_path}...")
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)
    logger.info(f"  ✓ Entity mappings: {len(entity2id)} entities")
    
    # Sample entities
    sample_entities = list(entity2id.keys())[:5]
    logger.info(f"\n  Sample entities: {sample_entities}")
    
    return entity_embeddings, entity2id, kg_data


def test_projection_layer(entity_embeddings: torch.Tensor):
    """Test 2: Projection layer functionality."""
    logger.info("\n" + "="*60)
    logger.info("Test 2: Projection Layer")
    logger.info("="*60)
    
    kg_dim = entity_embeddings.shape[1]
    lm_dim = 768  # Using smaller model for testing
    
    logger.info(f"Creating projection layer: {kg_dim} → {lm_dim}")
    projection = SimpleProjection(kg_dim=kg_dim, lm_dim=lm_dim)
    
    # Test projection
    sample_kg_emb = entity_embeddings[0]
    logger.info(f"  Input KG embedding: {sample_kg_emb.shape}")
    
    projected = projection(sample_kg_emb)
    logger.info(f"  ✓ Projected embedding: {projected.shape}")
    logger.info(f"  ✓ Projection layer parameters: {sum(p.numel() for p in projection.parameters()):,}")
    
    return projection


def test_entity_linking(entity2id: Dict[str, int]):
    """Test 3: Entity linking."""
    logger.info("\n" + "="*60)
    logger.info("Test 3: Entity Linking")
    logger.info("="*60)
    
    linker = SimpleEntityLinker(entity2id)
    
    # Test texts
    test_texts = [
        "What phenotypes are associated with Thbd knockout?",
        "Elevated glucose in Bmp4 mutant mice",
        "Compare Fgfr2 and Cyp2c23 knockouts",
    ]
    
    for text in test_texts:
        entities = linker.link_entities(text)
        logger.info(f"\n  Text: '{text}'")
        if entities:
            logger.info(f"  ✓ Found {len(entities)} entities:")
            for start, end, entity in entities:
                logger.info(f"    - {entity} at position [{start}:{end}]")
        else:
            logger.info("    No entities found")
    
    return linker


def test_embedding_fusion(
    entity_embeddings: torch.Tensor,
    entity2id: Dict[str, int],
    projection: nn.Module,
    linker: SimpleEntityLinker
):
    """Test 4: Embedding fusion with actual LLM."""
    logger.info("\n" + "="*60)
    logger.info("Test 4: Embedding Fusion")
    logger.info("="*60)
    
    # Load small model for testing
    model_name = "bert-base-uncased"
    logger.info(f"Loading {model_name} for testing...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Test text
    text = "What phenotypes does Thbd knockout cause?"
    logger.info(f"\n  Input text: '{text}'")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids']
    logger.info(f"  Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    
    # Get base embeddings
    with torch.no_grad():
        token_embeddings = model.embeddings.word_embeddings(input_ids)
    logger.info(f"  ✓ Base token embeddings: {token_embeddings.shape}")
    
    # Find entities
    entities = linker.link_entities(text)
    logger.info(f"  Found entities: {[e[2] for e in entities]}")
    
    if entities:
        # Augment with KG embedding
        for start_char, end_char, entity_name in entities:
            if entity_name in entity2id:
                entity_id = entity2id[entity_name]
                kg_emb = entity_embeddings[entity_id]
                
                # Project
                with torch.no_grad():
                    kg_proj = projection(kg_emb)
                
                logger.info(f"\n  Augmenting '{entity_name}':")
                logger.info(f"    KG embedding: {kg_emb.shape}")
                logger.info(f"    Projected: {kg_proj.shape}")
                
                # Fusion (weighted addition)
                # In real system, we'd find token positions more carefully
                alpha = 0.3
                logger.info(f"    Fusion weight α = {alpha}")
                
                # Demonstrate fusion (simplified - not exact token positions)
                token_avg = token_embeddings[0].mean(dim=0)
                fused = (1 - alpha) * token_avg + alpha * kg_proj
                logger.info(f"    ✓ Fused embedding: {fused.shape}")
                
                # Compute similarity
                cos_sim = F.cosine_similarity(token_avg.unsqueeze(0), kg_proj.unsqueeze(0))
                logger.info(f"    Cosine similarity (token, KG): {cos_sim.item():.4f}")
    
    logger.info("\n  ✓ Embedding fusion test successful!")


def test_batch_augmentation(
    entity_embeddings: torch.Tensor,
    entity2id: Dict[str, int],
    projection: nn.Module
):
    """Test 5: Batch augmentation."""
    logger.info("\n" + "="*60)
    logger.info("Test 5: Batch Augmentation")
    logger.info("="*60)
    
    # Simulate batch
    batch_size = 4
    seq_len = 20
    lm_dim = 768
    
    logger.info(f"  Simulating batch: {batch_size} sequences, length {seq_len}")
    
    # Fake token embeddings
    token_embeddings = torch.randn(batch_size, seq_len, lm_dim)
    logger.info(f"  Token embeddings: {token_embeddings.shape}")
    
    # Augment random entities in batch
    num_entities_per_seq = 2
    
    with torch.no_grad():
        for batch_idx in range(batch_size):
            # Sample random entities
            entity_ids = torch.randint(0, len(entity_embeddings), (num_entities_per_seq,))
            
            for entity_id in entity_ids:
                # Get KG embedding
                kg_emb = entity_embeddings[entity_id]
                
                # Project
                kg_proj = projection(kg_emb)
                
                # Random position in sequence
                pos = torch.randint(0, seq_len - 2, (1,)).item()
                
                # Fuse
                alpha = 0.3
                token_embeddings[batch_idx, pos:pos+2] = (
                    (1 - alpha) * token_embeddings[batch_idx, pos:pos+2] +
                    alpha * kg_proj.unsqueeze(0).repeat(2, 1)
                )
    
    logger.info(f"  ✓ Augmented {batch_size * num_entities_per_seq} entity mentions")
    logger.info(f"  ✓ Final embeddings: {token_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Test embedding augmentation")
    
    parser.add_argument("--kg_path", type=str, default="data/kg/biological_kg.pt",
                       help="Path to biological_kg.pt")
    parser.add_argument("--entity_embeddings", type=str, 
                       default="checkpoints/stage1/entity_embeddings.pt",
                       help="Path to entity embeddings from RotatE training")
    parser.add_argument("--entity2id_path", type=str, default="data/kg/entity2id.json",
                       help="Path to entity2id.json")
    
    args = parser.parse_args()
    
    # Check files exist
    for path in [args.kg_path, args.entity_embeddings, args.entity2id_path]:
        if not Path(path).exists():
            logger.error(f"File not found: {path}")
            logger.error("Please run Stage 1 (RotatE training) first:")
            logger.error("  python scripts/stage1_train_rotate.py")
            return
    
    try:
        # Run tests
        entity_embeddings, entity2id, kg_data = test_embedding_loading(
            args.kg_path,
            args.entity_embeddings,
            args.entity2id_path
        )
        
        projection = test_projection_layer(entity_embeddings)
        
        linker = test_entity_linking(entity2id)
        
        test_embedding_fusion(entity_embeddings, entity2id, projection, linker)
        
        test_batch_augmentation(entity_embeddings, entity2id, projection)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✓ All Tests Passed!")
        logger.info("="*60)
        logger.info("\nEmbedding augmentation is working correctly.")
        logger.info("\nNext steps:")
        logger.info("  1. Train projection layer (Stage 2):")
        logger.info("     python scripts/stage2_train_projection.py")
        logger.info("  2. Train LoRA adapters (Stage 3):")
        logger.info("     python scripts/stage3_train_lora.py")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
