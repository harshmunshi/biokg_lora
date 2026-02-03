#!/usr/bin/env python3
"""
Stage 1: Train RotatE embeddings for link prediction.

Duration: 2-3 days on 1 GPU (500 epochs with full KG)

Usage:
    python scripts/stage1_train_rotate.py \
        --kg_path data/kg/biological_kg.pt \
        --output_dir checkpoints/stage1 \
        --num_epochs 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from biokg_lora.data.dataset import KGDataset, collate_kg_batch
from biokg_lora.models.rotate import RotatE, rotate_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_link_prediction(model, dataloader, device, k_list=[1, 3, 10]):
    """
    Evaluate link prediction performance.
    
    Returns:
        mrr: Mean Reciprocal Rank
        hits: Dict of Hits@K
    """
    model.eval()
    ranks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            head, relation, tail = batch["head"], batch["relation"], batch["tail"]
            
            # Score all possible tails
            all_scores = []
            for i in range(len(head)):
                scores = model._score_all_tails(
                    head[i:i+1], relation[i:i+1], tail[i:i+1]
                )
                all_scores.append(scores)
            
            all_scores = torch.stack(all_scores)  # (B, num_entities)
            
            # Get ranks
            sorted_indices = torch.argsort(all_scores, dim=1, descending=False)
            
            for i in range(len(head)):
                true_tail = tail[i].item()
                # Find rank of true entity
                matches = (sorted_indices[i] == true_tail).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    rank = matches[0].item() + 1
                else:
                    # Edge case: true entity not found (shouldn't happen but handle gracefully)
                    rank = num_entities  # Worst possible rank
                ranks.append(rank)
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    # MRR
    mrr = (1.0 / ranks).mean().item()
    
    # Hits@K
    hits = {}
    for k in k_list:
        hits[k] = (ranks <= k).float().mean().item()
    
    model.train()
    return mrr, hits


def train(args):
    """Main training loop."""
    logger.info("="*60)
    logger.info("Stage 1: RotatE Embedding Training")
    logger.info("="*60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Enable performance optimizations for Ampere+ GPUs (A100, RTX 30/40 series)
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("✓ Enabled TF32 precision for faster training on Ampere+ GPUs")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = KGDataset(
        kg_path=args.kg_path,
        entity2id_path=args.entity2id_path,
        split="train",
        neg_sample_size=args.neg_sample_size,
    )
    
    val_dataset = KGDataset(
        kg_path=args.kg_path,
        entity2id_path=args.entity2id_path,
        split="val",
        neg_sample_size=args.neg_sample_size,
    )
    
    # DataLoader with performance optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_kg_batch,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if args.num_workers > 0 else False,  # Avoid worker respawning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_kg_batch,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("\nInitializing RotatE model...")
    model = RotatE(
        num_entities=train_dataset.num_entities,
        num_relations=train_dataset.num_relations,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Mixed precision training for faster GPU computation
    use_amp = torch.cuda.is_available() and args.use_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("✓ Enabled mixed precision training (AMP) for 1.3-1.8× speedup")
    
    # Training loop
    logger.info(f"\nTraining for {args.num_epochs} epochs...")
    
    best_mrr = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward with mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    positive_sample = (batch["head"], batch["relation"], batch["tail"])
                    loss, metrics = rotate_loss(
                        model,
                        positive_sample,
                        batch["negative_sample"],
                        mode=batch["mode"],
                        alpha=args.adversarial_temp,
                    )
                
                # Backward with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training (no AMP)
                positive_sample = (batch["head"], batch["relation"], batch["tail"])
                loss, metrics = rotate_loss(
                    model,
                    positive_sample,
                    batch["negative_sample"],
                    mode=batch["mode"],
                    alpha=args.adversarial_temp,
                )
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_every_n_epochs == 0:
            logger.info("Running validation...")
            mrr, hits = evaluate_link_prediction(
                model, val_loader, device, k_list=[1, 3, 10]
            )
            
            logger.info(f"Validation MRR: {mrr:.4f}")
            logger.info(f"Hits@1: {hits[1]:.4f}, Hits@3: {hits[3]:.4f}, Hits@10: {hits[10]:.4f}")
            
            # Save best model
            if mrr > best_mrr:
                best_mrr = mrr
                
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mrr": mrr,
                    "hits": hits,
                    "entity_embeddings": model.get_entity_embeddings(),
                    "relation_embeddings": model.get_relation_embeddings(),
                }
                
                torch.save(checkpoint, output_dir / "rotate_best.pt")
                logger.info(f"✓ Saved best model (MRR: {mrr:.4f})")
            
            scheduler.step(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, output_dir / f"rotate_epoch{epoch+1}.pt")
    
    # Save final embeddings
    logger.info("\nExtracting and saving embeddings...")
    
    checkpoint = torch.load(output_dir / "rotate_best.pt", weights_only=False)
    entity_embeddings = checkpoint["entity_embeddings"]
    relation_embeddings = checkpoint["relation_embeddings"]
    
    torch.save(entity_embeddings, output_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, output_dir / "relation_embeddings.pt")
    
    logger.info(f"✓ Entity embeddings: {entity_embeddings.shape}")
    logger.info(f"✓ Relation embeddings: {relation_embeddings.shape}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"Stage 1 Complete! Best MRR: {best_mrr:.4f}")
    logger.info("="*60)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("  - rotate_best.pt (best checkpoint)")
    logger.info("  - entity_embeddings.pt (for Stage 2)")
    logger.info("  - relation_embeddings.pt")
    
    logger.info("\n🚀 Next step: Stage 2 (Projection layer training)")
    logger.info(f"    python scripts/stage2_train_projection.py --kg_embeddings {output_dir}/entity_embeddings.pt")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: RotatE Training")
    
    parser.add_argument("--kg_path", type=str, required=True, help="Path to KG PyG Data file")
    parser.add_argument("--entity2id_path", type=str, required=True, help="Path to entity2id.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1", help="Output directory")
    
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--margin", type=float, default=9.0, help="Margin for loss")
    parser.add_argument("--neg_sample_size", type=int, default=128, help="Negative samples per positive")
    parser.add_argument("--adversarial_temp", type=float, default=1.0, help="Self-adversarial temperature")
    
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    
    parser.add_argument("--val_every_n_epochs", type=int, default=10, help="Validation frequency")
    parser.add_argument("--save_every_n_epochs", type=int, default=50, help="Checkpoint save frequency")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (8-16 recommended)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable mixed precision")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
