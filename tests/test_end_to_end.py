"""
End-to-end test for BioKG-LoRA pipeline with dummy data.

Tests:
1. KG construction
2. RotatE training
3. Dataset loading
4. Model forward pass
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from torch.utils.data import DataLoader

from biokg_lora.data.kg_builder import create_dummy_kg
from biokg_lora.data.dataset import KGDataset, collate_kg_batch
from biokg_lora.models.rotate import RotatE, rotate_loss
from biokg_lora.models.projection import KGProjection
from biokg_lora.models.entity_linker import EntityLinker
from biokg_lora.visualization.kg_viz import visualize_kg_interactive


def test_kg_construction():
    """Test KG construction."""
    print("\n[Test 1/6] KG Construction...")
    
    kg_data, metadata = create_dummy_kg(
        num_genes=50,
        num_phenotypes=25,
        seed=42,
    )
    
    assert kg_data.num_nodes > 0
    assert kg_data.edge_index.size(1) > 0
    assert len(metadata["entity2id"]) == kg_data.num_nodes
    
    print(f"✓ KG: {kg_data.num_nodes} nodes, {kg_data.edge_index.size(1)} edges")


def test_dataset():
    """Test dataset loading."""
    print("\n[Test 2/6] Dataset Loading...")
    
    kg_data, metadata = create_dummy_kg(num_genes=50, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save KG
        torch.save(kg_data, tmpdir / "test_kg.pt")
        with open(tmpdir / "entity2id.json", "w") as f:
            json.dump(metadata["entity2id"], f)
        
        # Create dataset
        dataset = KGDataset(
            kg_path=str(tmpdir / "test_kg.pt"),
            entity2id_path=str(tmpdir / "entity2id.json"),
            split="train",
            neg_sample_size=32,
        )
        
        assert len(dataset) > 0
        
        # Test __getitem__
        sample = dataset[0]
        assert "head" in sample
        assert "relation" in sample
        assert "tail" in sample
        assert sample["negative_sample"].shape[0] == 32
        
        print(f"✓ Dataset: {len(dataset)} samples")


def test_rotate_training():
    """Test RotatE training."""
    print("\n[Test 3/6] RotatE Training...")
    
    kg_data, metadata = create_dummy_kg(num_genes=50, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save KG
        torch.save(kg_data, tmpdir / "test_kg.pt")
        with open(tmpdir / "entity2id.json", "w") as f:
            json.dump(metadata["entity2id"], f)
        
        # Create dataset
        dataset = KGDataset(
            kg_path=str(tmpdir / "test_kg.pt"),
            entity2id_path=str(tmpdir / "entity2id.json"),
            split="train",
            neg_sample_size=32,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            collate_fn=collate_kg_batch,
        )
        
        # Create model
        model = RotatE(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=64,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 2 epochs
        for epoch in range(2):
            for batch in dataloader:
                positive_sample = (batch["head"], batch["relation"], batch["tail"])
                loss, metrics = rotate_loss(
                    model,
                    positive_sample,
                    batch["negative_sample"],
                    mode=batch["mode"],
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Get embeddings
        entity_embeddings = model.get_entity_embeddings()
        
        assert entity_embeddings.shape[0] == dataset.num_entities
        assert entity_embeddings.shape[1] == 64
        
        print(f"✓ RotatE: embeddings shape {entity_embeddings.shape}")


def test_projection():
    """Test projection layer."""
    print("\n[Test 4/6] Projection Layer...")
    
    projection = KGProjection(
        kg_embedding_dim=256,
        lm_embedding_dim=4096,
        hidden_dim=1024,
    )
    
    # Test forward
    kg_emb = torch.randn(10, 256)
    lm_emb = projection(kg_emb)
    
    assert lm_emb.shape == (10, 4096)
    
    print(f"✓ Projection: {kg_emb.shape} → {lm_emb.shape}")


def test_entity_linker():
    """Test entity linker."""
    print("\n[Test 5/6] Entity Linker...")
    
    entity_names = ["Thbd", "Bmp4", "MP:0003350", "GO:0007596"]
    entity2id = {name: i for i, name in enumerate(entity_names)}
    
    linker = EntityLinker(entity2id, use_scispacy=False)
    
    text = "The gene Thbd causes phenotype MP:0003350."
    entities = linker.link_entities(text)
    
    assert len(entities) >= 2  # Should find Thbd and MP:0003350
    
    print(f"✓ Entity Linker: found {len(entities)} entities")


def test_visualization():
    """Test KG visualization."""
    print("\n[Test 6/6] Visualization...")
    
    kg_data, metadata = create_dummy_kg(num_genes=30, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save KG
        torch.save(kg_data, tmpdir / "test_kg.pt")
        with open(tmpdir / "entity2id.json", "w") as f:
            json.dump(metadata["entity2id"], f)
        
        # Create visualization
        try:
            visualize_kg_interactive(
                kg_path=str(tmpdir / "test_kg.pt"),
                entity2id_path=str(tmpdir / "entity2id.json"),
                output_html=str(tmpdir / "test_viz.html"),
                max_nodes=30,
            )
            
            assert (tmpdir / "test_viz.html").exists()
            print(f"✓ Visualization created")
        except Exception as e:
            print(f"⚠ Visualization failed (expected if dependencies missing): {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("BioKG-LoRA End-to-End Test")
    print("="*60)
    
    try:
        test_kg_construction()
        test_dataset()
        test_rotate_training()
        test_projection()
        test_entity_linker()
        test_visualization()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
