# BioKG-LoRA: Standalone Implementation

## ⚠️ Important Clarification

**BioKG-LoRA is completely self-contained and independent.**

### ❌ Common Misconception

BioKG-LoRA does **NOT** reuse embeddings from GraphPath-VLM.

### ✅ Reality

BioKG-LoRA trains everything from scratch:

1. **Stage 0: Build KG** (1-2 days, CPU)
   - Download and integrate biological databases
   - Create unified knowledge graph
   - 87K entities, 1.5M triples

2. **Stage 1: Train RotatE Embeddings** (2-3 days, 1 GPU)
   - Train RotatE from scratch on the KG
   - Link prediction task
   - Output: 256-dim embeddings for all entities

3. **Stage 2: Train Projection Layer** (2 hours, 1 GPU)
   - Align KG embeddings with LLM embeddings
   - Contrastive learning

4. **Stage 3: LoRA Fine-tuning** (4-6 hours, 1 GPU)
   - Fine-tune LLM with KG augmentation
   - QA task with biological questions

**Total time: ~1 week from raw data to trained model**

---

## Why Two Separate Projects?

### BioKG-LoRA (This Project)
- **Purpose**: Text-based clinical reasoning
- **Input**: Questions about genes, phenotypes, clinical parameters
- **Output**: Natural language explanations
- **Modality**: Text → Text
- **Training**: KG construction → RotatE → Projection → LoRA
- **Timeline**: ~1 week

### GraphPath-VLM (Separate Project)
- **Purpose**: Vision-based phenotype discovery
- **Input**: Whole Slide Images (WSIs) of tissue
- **Output**: Phenotype predictions + visual attention
- **Modality**: Images + KG → Predictions
- **Training**: KG construction → RotatE → Vision encoder → Multimodal fusion
- **Timeline**: ~2-3 weeks

---

## Can They Share Embeddings?

**Yes, optionally!** If you want to:

1. **Train once, use twice**:
   - Build KG once (Stage 0)
   - Train RotatE once (Stage 1)
   - Use embeddings in both BioKG-LoRA AND GraphPath-VLM

2. **Or train separately**:
   - Each project can train its own embeddings
   - Allows project-specific KG customization
   - No dependencies between projects

**But the default implementation trains from scratch.**

---

## Current Implementation Status

### ✅ Implemented in BioKG-LoRA

- **Stage 0**: Complete KG construction pipeline
  - `scripts/stage0_build_kg.py`
  - `biokg_lora/data/kg_builder.py`
  - Dummy KG for testing

- **Stage 1**: Complete RotatE training
  - `scripts/stage1_train_rotate.py`
  - `biokg_lora/models/rotate.py`
  - Full training loop with validation

- **Stage 2**: Architecture ready (training script TODO)
  - `biokg_lora/models/projection.py`
  - Contrastive loss implemented

- **Stage 3**: Architecture ready (training script TODO)
  - `biokg_lora/models/biokg_lora.py`
  - Full model with LoRA

---

## Timeline Breakdown

### From Scratch (Recommended)

```
Day 1-2:   Build KG (Stage 0)
Day 3-5:   Train RotatE (Stage 1)
Day 6:     Train Projection (Stage 2)
Day 7:     LoRA Fine-tuning (Stage 3)

Total: 1 week
```

### With Existing Embeddings (Optional)

If you already have RotatE embeddings (e.g., from GraphPath-VLM):

```
Day 1:  Train Projection (Stage 2)
Day 2:  LoRA Fine-tuning (Stage 3)

Total: 2 days
```

**But you still need to implement Stage 2 and 3 training scripts!**

---

## What You Can Run Right Now

### With Dummy Data (5 minutes)

```bash
python scripts/quickstart.py
```

This runs:
- ✅ Stage 0: Creates dummy KG (380 entities)
- ✅ Stage 1: Trains mini RotatE (10 epochs)
- ✅ Visualization

### With Custom Dummy Data (30 minutes)

```bash
# Build KG
python scripts/stage0_build_kg.py --mode dummy --num_genes 500

# Train RotatE
python scripts/stage1_train_rotate.py \
    --kg_path data/kg/biological_kg.pt \
    --entity2id_path data/kg/entity2id.json \
    --num_epochs 20
```

### Full Pipeline with Real Data

1. **Download data** (See docs/DATA_SOURCES.md - TODO)
2. **Build KG** (Stage 0 - implemented)
3. **Train RotatE** (Stage 1 - implemented)
4. **Train Projection** (Stage 2 - TODO)
5. **LoRA Fine-tuning** (Stage 3 - TODO)

---

## Key Takeaways

1. ✅ **BioKG-LoRA is standalone** - trains everything from scratch
2. ✅ **No dependency on GraphPath-VLM** - can run independently
3. ✅ **Stages 0-1 are complete** - KG + RotatE ready to use
4. 🚧 **Stages 2-3 need training scripts** - architectures ready, scripts TODO
5. 🔄 **Can optionally share embeddings** - but not required

---

## Relationship to GraphPath-VLM

```
┌─────────────────────────────────────────────────────────────┐
│  Biological Knowledge Graph (Shared)                        │
│  - MGI, GO, KEGG, STRING, MPO, GTEx                         │
│  - 87K entities, 1.5M triples                               │
└─────────────────────────────────────────────────────────────┘
                    ↓                      ↓
         ┌──────────────────┐  ┌───────────────────────┐
         │  BioKG-LoRA      │  │  GraphPath-VLM        │
         │  (Text)          │  │  (Vision)             │
         ├──────────────────┤  ├───────────────────────┤
         │ • RotatE trained │  │ • RotatE trained      │
         │   from scratch   │  │   from scratch        │
         │ • Projection     │  │ • Vision encoder      │
         │ • LLM + LoRA     │  │ • Cross-attention     │
         │ • QA generation  │  │ • WSI processing      │
         └──────────────────┘  └───────────────────────┘
                    ↓                      ↓
         Clinical Reasoning      Phenotype Discovery
         from Text               from Images
```

**Independent projects, optional sharing**

---

## FAQ

**Q: Do I need GraphPath-VLM to run BioKG-LoRA?**  
A: No. BioKG-LoRA is completely standalone.

**Q: Can I use embeddings from GraphPath-VLM in BioKG-LoRA?**  
A: Yes, if you want. But BioKG-LoRA trains its own by default.

**Q: How long does it take to train from scratch?**  
A: ~1 week (mostly Stage 1 RotatE training: 2-3 days on GPU).

**Q: What's implemented right now?**  
A: Stages 0-1 (KG + RotatE) are fully implemented. Stages 2-3 need training scripts.

**Q: Can I run anything now?**  
A: Yes! `python scripts/quickstart.py` runs a complete demo with dummy data.

**Q: What do I need to implement next?**  
A: Training scripts for Stage 2 (projection) and Stage 3 (LoRA).

---

## Summary

✅ **Standalone**: Trains everything from scratch  
✅ **Self-contained**: No external dependencies  
✅ **Complete Stages 0-1**: KG construction + RotatE training  
🚧 **TODO Stages 2-3**: Projection + LoRA training scripts  
🔄 **Optional sharing**: Can reuse embeddings with GraphPath-VLM if desired  

**Total timeline from scratch: ~1 week**  
**Ready to run: Quickstart demo (5 minutes)**
