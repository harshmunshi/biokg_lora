# Quick Start Guide

Get started with BioKG-LoRA in 5 minutes using dummy data.

---

## Prerequisites

- Python 3.10+
- PyTorch installed
- BioKG-LoRA installed (`pip install -e .`)

---

## 1. Run Quick Demo

```bash
python scripts/quickstart.py
```

This will:
- ✅ Create a dummy knowledge graph (380 entities, 1200+ triples)
- ✅ Train a mini RotatE model (10 epochs, ~2 minutes)
- ✅ Generate visualizations
- ✅ Test model forward pass

**Expected output:**

```
BioKG-LoRA Quick Start Demo
============================================================

[1/5] Creating dummy knowledge graph...
✓ KG created: 380 nodes, 1245 edges

[2/5] Creating training dataset...
✓ Dataset created: 996 samples

[3/5] Training RotatE model (10 epochs)...
  Epoch 2/10: Loss = 0.1234
  Epoch 4/10: Loss = 0.0987
  Epoch 6/10: Loss = 0.0756
  Epoch 8/10: Loss = 0.0645
  Epoch 10/10: Loss = 0.0578
✓ Training complete!

[4/5] Creating visualizations...
✓ Interactive visualization: outputs/quickstart/kg_interactive.html
✓ Subgraph visualization: outputs/quickstart/subgraph.png

[5/5] Testing embedding quality...
  Similarity (Entity 0, Entity 1): 0.2341
  Similarity (Entity 0, Entity 50): -0.1234

Quick Start Complete! 🎉
```

**Outputs saved to**: `outputs/quickstart/`

---

## 2. Explore Visualizations

### Interactive KG Visualization

Open in browser:

```bash
open outputs/quickstart/kg_interactive.html
```

Features:
- 🎨 Color-coded by entity type
- 🔍 Interactive zoom and pan
- 📝 Hover for entity names
- 🔗 Click to explore connections

### Subgraph Visualization

View PNG:

```bash
open outputs/quickstart/subgraph.png
```

Shows 2-hop neighborhood around a gene.

---

## 3. Build Custom KG

### Dummy KG with Custom Size

```bash
python scripts/stage0_build_kg.py \
    --mode dummy \
    --num_genes 500 \
    --num_phenotypes 250 \
    --visualize \
    --output_dir data/kg
```

Outputs:
- `data/kg/biological_kg.pt` - KG data
- `data/kg/entity2id.json` - Entity mappings
- `outputs/kg_viz/kg_interactive.html` - Visualization

---

## 4. Train RotatE Embeddings

### Mini Training (5 minutes)

```bash
python scripts/stage1_train_rotate.py \
    --kg_path data/kg/biological_kg.pt \
    --entity2id_path data/kg/entity2id.json \
    --num_epochs 20 \
    --batch_size 128 \
    --output_dir checkpoints/stage1_mini
```

### Full Training (2-3 days on GPU)

```bash
python scripts/stage1_train_rotate.py \
    --kg_path data/kg/biological_kg.pt \
    --entity2id_path data/kg/entity2id.json \
    --num_epochs 500 \
    --batch_size 1024 \
    --output_dir checkpoints/stage1
```

---

## 5. Test Components

### Test KG Construction

```python
from biokg_lora.data.kg_builder import create_dummy_kg

kg_data, metadata = create_dummy_kg(
    num_genes=100,
    num_phenotypes=50,
    seed=42
)

print(f"Nodes: {kg_data.num_nodes}")
print(f"Edges: {kg_data.edge_index.size(1)}")
```

### Test RotatE Model

```python
import torch
from biokg_lora.models.rotate import RotatE

model = RotatE(
    num_entities=1000,
    num_relations=10,
    embedding_dim=256
)

# Forward pass
head = torch.randint(0, 1000, (32,))
relation = torch.randint(0, 10, (32,))
tail = torch.randint(0, 1000, (32,))

scores = model(head, relation, tail)
print(f"Scores: {scores.shape}")
```

### Test Entity Linker

```python
from biokg_lora.models.entity_linker import EntityLinker

entity2id = {
    "Thbd": 0,
    "Bmp4": 1,
    "MP:0003350": 2,
}

linker = EntityLinker(entity2id)

text = "The gene Thbd causes phenotype MP:0003350."
entities = linker.link_entities(text)

for name, start, end, eid in entities:
    print(f"Found: {name} (ID: {eid})")
```

---

## 6. Visualize KG

### Interactive Visualization

```python
from biokg_lora.visualization.kg_viz import visualize_kg_interactive

visualize_kg_interactive(
    kg_path="data/kg/biological_kg.pt",
    entity2id_path="data/kg/entity2id.json",
    output_html="my_kg.html",
    max_nodes=500
)
```

### Subgraph Around Entity

```python
from biokg_lora.visualization.kg_viz import visualize_subgraph

visualize_subgraph(
    kg_path="data/kg/biological_kg.pt",
    entity2id_path="data/kg/entity2id.json",
    center_entity="Gene0000",
    hops=2,
    output_path="subgraph.png"
)
```

---

## 7. Run End-to-End Test

```bash
python tests/test_end_to_end.py
```

Tests:
- ✅ KG construction
- ✅ Dataset loading
- ✅ RotatE training
- ✅ Projection layer
- ✅ Entity linker
- ✅ Visualization

---

## 8. Next Steps

### With Dummy Data

✅ You can complete the entire pipeline with dummy data:

```bash
# Stage 0: Build KG
python scripts/stage0_build_kg.py --mode dummy

# Stage 1: Train RotatE (fast mini version)
python scripts/stage1_train_rotate.py \
    --kg_path data/kg/biological_kg.pt \
    --entity2id_path data/kg/entity2id.json \
    --num_epochs 20

# Output: checkpoints/stage1/entity_embeddings.pt
```

### With Real Data

📚 To use real biological data:

1. **Download data sources** - See `DATA_SOURCES.md`
2. **Build full KG** - Implement data parsers
3. **Train on real data** - Use provided scripts
4. **Evaluate** - See `TRAINING.md`

---

## Common Issues

### Q: "ImportError: No module named 'biokg_lora'"

**A**: Install package:
```bash
pip install -e .
```

### Q: "CUDA out of memory"

**A**: Reduce batch size:
```bash
python scripts/stage1_train_rotate.py --batch_size 128
```

### Q: "spaCy model not found"

**A**: Install spaCy models:
```bash
python -m spacy download en_core_web_sm
```

### Q: Slow training

**A**: Use GPU and increase workers:
```bash
python scripts/stage1_train_rotate.py --num_workers 8
```

---

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: GitHub Issues
- **Examples**: `scripts/` folder
- **Tests**: `tests/` folder

---

## Summary

**5-Minute Demo**:
```bash
python scripts/quickstart.py
```

**Custom KG + Training**:
```bash
python scripts/stage0_build_kg.py --mode dummy --num_genes 500
python scripts/stage1_train_rotate.py --kg_path data/kg/biological_kg.pt --entity2id_path data/kg/entity2id.json --num_epochs 20
```

**Test Everything**:
```bash
python tests/test_end_to_end.py
```

🚀 **You're ready to explore BioKG-LoRA!**
