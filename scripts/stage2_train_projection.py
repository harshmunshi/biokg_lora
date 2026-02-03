#!/usr/bin/env python3
"""
Stage 2: Projection Layer Training

Train projection layer to align KG embeddings with LLM embedding space
using contrastive learning (InfoNCE loss).

Duration: ~2 hours on 1× GPU (A100 or RTX 4090)

Usage:
    python scripts/stage2_train_projection.py \
        --entity_embeddings checkpoints/stage1/entity_embeddings.pt \
        --entity2id_path data/kg/entity2id.json \
        --base_model "meta-llama/Llama-3-8B" \
        --output_dir checkpoints/stage2 \
        --num_epochs 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectionLayer(nn.Module):
    """
    Project KG embeddings to LLM embedding space.
    
    Uses multi-layer MLP with normalization for better alignment.
    """
    
    def __init__(self, kg_dim: int = 256, lm_dim: int = 4096, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, lm_dim),
            nn.LayerNorm(lm_dim),
        )
    
    def forward(self, kg_emb):
        """
        Args:
            kg_emb: (B, kg_dim) or (kg_dim,)
        Returns:
            projected: (B, lm_dim) or (lm_dim,)
        """
        return self.proj(kg_emb)


class EntityAlignmentDataset(Dataset):
    """
    Dataset for entity alignment: pairs of (entity_name, kg_embedding).
    """
    
    def __init__(
        self,
        entity_embeddings: torch.Tensor,
        entity2id: Dict[str, int],
        id2entity: Dict[int, str]
    ):
        self.entity_embeddings = entity_embeddings
        self.entity2id = entity2id
        self.id2entity = id2entity
        
        # Create list of valid entities (excluding special tokens)
        self.valid_entities = []
        for entity_id in range(len(entity_embeddings)):
            if entity_id in id2entity:
                entity_name = id2entity[entity_id]
                # Filter out very short or special entities
                if len(entity_name) > 1 and not entity_name.startswith("_"):
                    self.valid_entities.append((entity_id, entity_name))
        
        logger.info(f"  Valid entities for alignment: {len(self.valid_entities)}")
    
    def __len__(self):
        return len(self.valid_entities)
    
    def __getitem__(self, idx):
        entity_id, entity_name = self.valid_entities[idx]
        kg_embedding = self.entity_embeddings[entity_id]
        
        return {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'kg_embedding': kg_embedding
        }


def collate_fn(batch):
    """Collate batch of entity data."""
    entity_ids = [item['entity_id'] for item in batch]
    entity_names = [item['entity_name'] for item in batch]
    kg_embeddings = torch.stack([item['kg_embedding'] for item in batch])
    
    return {
        'entity_ids': entity_ids,
        'entity_names': entity_names,
        'kg_embeddings': kg_embeddings
    }


def info_nce_loss(
    kg_proj: torch.Tensor,
    lm_embs: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for alignment.
    
    Args:
        kg_proj: (B, D) projected KG embeddings
        lm_embs: (B, D) LM embeddings
        temperature: Temperature for softmax
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize
    kg_proj = F.normalize(kg_proj, dim=-1)
    lm_embs = F.normalize(lm_embs, dim=-1)
    
    # Compute similarity matrix: (B, B)
    # similarity[i, j] = cosine_sim(kg_proj[i], lm_embs[j])
    similarity = torch.matmul(kg_proj, lm_embs.T) / temperature
    
    # Labels: diagonal elements are positives (i-th KG should match i-th LM)
    batch_size = kg_proj.size(0)
    labels = torch.arange(batch_size, device=kg_proj.device)
    
    # Cross-entropy: maximize similarity for positives, minimize for negatives
    loss = F.cross_entropy(similarity, labels)
    
    return loss


def get_lm_embeddings(
    entity_names: List[str],
    tokenizer,
    model,
    device: str
) -> torch.Tensor:
    """
    Get LM embeddings for entity names.
    
    Args:
        entity_names: List of entity names
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: Device to use
    
    Returns:
        embeddings: (B, hidden_dim) mean-pooled token embeddings
    """
    # Tokenize
    inputs = tokenizer(
        entity_names,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=32
    ).to(device)
    
    # Get embeddings from input embedding layer
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(inputs['input_ids'])
        # Mean pool over tokens (excluding padding)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    
    return embeddings


def train_epoch(
    projection: ProjectionLayer,
    dataloader: DataLoader,
    tokenizer,
    lm_model,
    optimizer,
    device: str,
    temperature: float,
    epoch: int
) -> Tuple[float, float]:
    """Train projection layer for one epoch."""
    projection.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        entity_names = batch['entity_names']
        kg_embeddings = batch['kg_embeddings'].to(device)
        
        # Get LM embeddings
        lm_embeddings = get_lm_embeddings(entity_names, tokenizer, lm_model, device)
        
        # Project KG embeddings
        kg_projected = projection(kg_embeddings)
        
        # Compute InfoNCE loss
        loss = info_nce_loss(kg_projected, lm_embeddings, temperature=temperature)
        
        # Compute accuracy (correct matches)
        with torch.no_grad():
            kg_proj_norm = F.normalize(kg_projected, dim=-1)
            lm_emb_norm = F.normalize(lm_embeddings, dim=-1)
            similarity = torch.matmul(kg_proj_norm, lm_emb_norm.T)
            predictions = similarity.argmax(dim=1)
            correct = (predictions == torch.arange(len(entity_names), device=device)).float().mean()
            total_acc += correct.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projection.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct.item():.3f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc


def evaluate(
    projection: ProjectionLayer,
    dataloader: DataLoader,
    tokenizer,
    lm_model,
    device: str,
    temperature: float
) -> Tuple[float, float]:
    """Evaluate projection layer."""
    projection.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            entity_names = batch['entity_names']
            kg_embeddings = batch['kg_embeddings'].to(device)
            
            # Get LM embeddings
            lm_embeddings = get_lm_embeddings(entity_names, tokenizer, lm_model, device)
            
            # Project KG embeddings
            kg_projected = projection(kg_embeddings)
            
            # Compute loss
            loss = info_nce_loss(kg_projected, lm_embeddings, temperature=temperature)
            
            # Compute accuracy
            kg_proj_norm = F.normalize(kg_projected, dim=-1)
            lm_emb_norm = F.normalize(lm_embeddings, dim=-1)
            similarity = torch.matmul(kg_proj_norm, lm_emb_norm.T)
            predictions = similarity.argmax(dim=1)
            correct = (predictions == torch.arange(len(entity_names), device=device)).float().mean()
            
            total_loss += loss.item()
            total_acc += correct.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc


def visualize_alignment(
    projection: ProjectionLayer,
    entity_embeddings: torch.Tensor,
    entity2id: Dict[str, int],
    id2entity: Dict[int, str],
    tokenizer,
    lm_model,
    device: str,
    num_samples: int = 10
):
    """Visualize alignment quality for sample entities."""
    logger.info("\n" + "="*60)
    logger.info("Sample Entity Alignments")
    logger.info("="*60)
    
    # Sample random entities
    valid_ids = [i for i in range(len(entity_embeddings)) if i in id2entity]
    sample_ids = torch.randperm(len(valid_ids))[:num_samples].tolist()
    
    projection.eval()
    with torch.no_grad():
        for idx in sample_ids:
            entity_id = valid_ids[idx]
            entity_name = id2entity[entity_id]
            
            # Get KG embedding and project
            kg_emb = entity_embeddings[entity_id].to(device)
            kg_proj = projection(kg_emb)
            
            # Get LM embedding
            lm_emb = get_lm_embeddings([entity_name], tokenizer, lm_model, device)[0]
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(kg_proj, lm_emb, dim=0).item()
            
            logger.info(f"  {entity_name:20s} | Cosine Similarity: {cos_sim:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Projection Layer Training")
    
    # Input
    parser.add_argument("--entity_embeddings", type=str, required=True,
                       help="Path to entity embeddings from Stage 1")
    parser.add_argument("--entity2id_path", type=str, required=True,
                       help="Path to entity2id.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3-8B",
                       help="Base LLM model for alignment")
    
    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=1024,
                       help="Hidden dimension for projection layer")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07,
                       help="Temperature for InfoNCE loss")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="Train/val split ratio")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(project="biokg-lora-stage2", config=vars(args))
    
    # Load data
    logger.info("="*60)
    logger.info("Stage 2: Projection Layer Training")
    logger.info("="*60)
    
    logger.info("\nLoading entity embeddings...")
    entity_embeddings = torch.load(args.entity_embeddings, weights_only=False)
    logger.info(f"  Entity embeddings shape: {entity_embeddings.shape}")
    
    logger.info("Loading entity mappings...")
    with open(args.entity2id_path, 'r') as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    logger.info(f"  Entities: {len(entity2id)}")
    
    # Load LM model (for embeddings only)
    logger.info(f"\nLoading base model: {args.base_model}")
    logger.info("  (Loading for embeddings only, not generation)")
    
    try:
        # Try loading as causal LM first
        lm_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"  Could not load as CausalLM: {e}")
        logger.info("  Trying as base model...")
        # Fallback to base model
        lm_model = AutoModel.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
    
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    logger.info("\nCreating alignment dataset...")
    dataset = EntityAlignmentDataset(entity_embeddings, entity2id, id2entity)
    
    # Train/val split
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"  Train: {len(train_dataset)} entities")
    logger.info(f"  Val: {len(val_dataset)} entities")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create projection layer
    kg_dim = entity_embeddings.shape[1]
    lm_dim = lm_model.config.hidden_size
    
    logger.info(f"\nCreating projection layer: {kg_dim} → {lm_dim}")
    projection = ProjectionLayer(
        kg_dim=kg_dim,
        lm_dim=lm_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in projection.parameters())
    logger.info(f"  Trainable parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Training Started")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(
            projection, train_loader, tokenizer, lm_model,
            optimizer, device, args.temperature, epoch + 1
        )
        
        val_loss, val_acc = evaluate(
            projection, val_loader, tokenizer, lm_model,
            device, args.temperature
        )
        
        scheduler.step()
        
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.3f}")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  ✓ New best model! Saving...")
            
            # Save projection layer
            torch.save(
                projection.state_dict(),
                output_dir / "projection_layer.pt"
            )
            
            # Save config
            config = {
                "kg_dim": kg_dim,
                "lm_dim": lm_dim,
                "hidden_dim": args.hidden_dim,
                "temperature": args.temperature,
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
                "base_model": args.base_model
            }
            with open(output_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
    
    # Visualize alignment
    visualize_alignment(
        projection, entity_embeddings, entity2id, id2entity,
        tokenizer, lm_model, device
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"\nProjection layer ready for Stage 3 (LoRA fine-tuning)")
    logger.info("\nNext step:")
    logger.info(f"  python scripts/train_lora.py \\")
    logger.info(f"      --entity_embeddings {args.entity_embeddings} \\")
    logger.info(f"      --projection_layer {output_dir}/projection_layer.pt \\")
    logger.info(f"      --qa_dataset data/qa_dataset.json")


if __name__ == "__main__":
    main()
