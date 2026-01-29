#!/usr/bin/env python3
"""
Quick Start Demo for BioKG-LoRA.

Runs a complete mini-pipeline with dummy data:
1. Creates dummy KG
2. Trains mini RotatE model
3. Visualizes KG
4. Tests model forward pass

Usage:
    python scripts/quickstart.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from biokg_lora.data.kg_builder import create_dummy_kg
from biokg_lora.data.dataset import KGDataset, collate_kg_batch
from biokg_lora.models.rotate import RotatE, rotate_loss
from biokg_lora.visualization.kg_viz import visualize_kg_interactive, visualize_subgraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("BioKG-LoRA Quick Start Demo")
    logger.info("="*60)
    
    # Create output directory
    output_dir = Path("outputs/quickstart")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create dummy KG
    logger.info("\n[1/5] Creating dummy knowledge graph...")
    kg_data, metadata = create_dummy_kg(
        num_genes=100,
        num_phenotypes=50,
        num_go_terms=200,
        num_pathways=20,
        num_tissues=10,
        seed=42,
    )
    
    # Save KG
    kg_dir = output_dir / "kg"
    kg_dir.mkdir(exist_ok=True)
    
    import json
    torch.save(kg_data, kg_dir / "biological_kg.pt")
    with open(kg_dir / "entity2id.json", "w") as f:
        json.dump(metadata["entity2id"], f, indent=2)
    with open(kg_dir / "id2entity.json", "w") as f:
        json.dump(metadata["id2entity"], f, indent=2)
    
    logger.info(f"✓ KG created: {kg_data.num_nodes} nodes, {kg_data.edge_index.size(1)} edges")
    
    # Step 2: Create dataset
    logger.info("\n[2/5] Creating training dataset...")
    dataset = KGDataset(
        kg_path=str(kg_dir / "biological_kg.pt"),
        entity2id_path=str(kg_dir / "entity2id.json"),
        split="train",
        neg_sample_size=32,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_kg_batch,
    )
    
    logger.info(f"✓ Dataset created: {len(dataset)} samples")
    
    # Step 3: Train mini RotatE model
    logger.info("\n[3/5] Training RotatE model (10 epochs)...")
    
    model = RotatE(
        num_entities=kg_data.num_nodes,
        num_relations=kg_data.edge_type.max().item() + 1,
        embedding_dim=128,  # Smaller for demo
        margin=9.0,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            positive_sample = (batch["head"], batch["relation"], batch["tail"])
            loss, metrics = rotate_loss(
                model,
                positive_sample,
                batch["negative_sample"],
                mode=batch["mode"],
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 2 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    logger.info(f"✓ Training complete!")
    
    # Save model
    model_path = output_dir / "rotate_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "entity_embeddings": model.get_entity_embeddings(),
        "relation_embeddings": model.get_relation_embeddings(),
    }, model_path)
    
    logger.info(f"✓ Model saved to {model_path}")
    
    # Step 4: Visualize KG
    logger.info("\n[4/5] Creating visualizations...")
    
    try:
        visualize_kg_interactive(
            kg_path=str(kg_dir / "biological_kg.pt"),
            entity2id_path=str(kg_dir / "entity2id.json"),
            output_html=str(output_dir / "kg_interactive.html"),
            max_nodes=100,
        )
        logger.info(f"✓ Interactive visualization: {output_dir / 'kg_interactive.html'}")
    except Exception as e:
        logger.warning(f"Could not create interactive viz: {e}")
    
    try:
        visualize_subgraph(
            kg_path=str(kg_dir / "biological_kg.pt"),
            entity2id_path=str(kg_dir / "entity2id.json"),
            center_entity="Gene0000",
            hops=2,
            output_path=str(output_dir / "subgraph.png"),
        )
        logger.info(f"✓ Subgraph visualization: {output_dir / 'subgraph.png'}")
    except Exception as e:
        logger.warning(f"Could not create subgraph viz: {e}")
    
    # Step 5: Test embedding quality
    logger.info("\n[5/5] Testing embedding quality...")
    
    embeddings = model.get_entity_embeddings().cpu()
    
    # Compute cosine similarity for a few entity pairs
    from torch.nn.functional import normalize
    embeddings_norm = normalize(embeddings, dim=1)
    
    # Random entity pairs
    entity1, entity2 = 0, 1
    entity3, entity4 = 0, 50
    
    sim_12 = (embeddings_norm[entity1] @ embeddings_norm[entity2]).item()
    sim_34 = (embeddings_norm[entity3] @ embeddings_norm[entity4]).item()
    
    logger.info(f"  Similarity (Entity 0, Entity 1): {sim_12:.4f}")
    logger.info(f"  Similarity (Entity 0, Entity 50): {sim_34:.4f}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Quick Start Complete! 🎉")
    logger.info("="*60)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("  - kg/biological_kg.pt")
    logger.info("  - kg/entity2id.json")
    logger.info("  - rotate_model.pt")
    logger.info("  - kg_interactive.html")
    logger.info("  - subgraph.png")
    
    logger.info("\n📚 Next Steps:")
    logger.info("  1. Explore visualizations in outputs/quickstart/")
    logger.info("  2. Run full pipeline with real data:")
    logger.info("     python scripts/stage0_build_kg.py")
    logger.info("     python scripts/stage1_train_rotate.py")
    logger.info("     python scripts/stage2_train_projection.py")
    logger.info("     python scripts/stage3_train_lora.py")
    logger.info("  3. Read docs/ for detailed guides")


if __name__ == "__main__":
    main()
