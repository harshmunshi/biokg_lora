# 🚨 IMPORTANT: BioKG-LoRA is Standalone

## Correction to Previous Documentation

**BioKG-LoRA does NOT reuse embeddings from GraphPath-VLM.**

### What Changed

**OLD** (Incorrect):
- ❌ "Reuse RotatE embeddings from GraphPath-VLM"
- ❌ "Already trained embeddings"
- ❌ "Fast turnaround (2-3 days)"

**NEW** (Correct):
- ✅ "Train RotatE embeddings from scratch"
- ✅ "Complete 4-stage pipeline"
- ✅ "Timeline: ~1 week from raw data to trained model"

---

## Complete Pipeline

### Stage 0: KG Construction ✅ IMPLEMENTED
- **Duration**: 1-2 days (CPU)
- **Script**: `scripts/stage0_build_kg.py`
- **What it does**: Downloads and integrates biological databases
- **Output**: `biological_kg.pt` (87K entities, 1.5M triples)

### Stage 1: RotatE Training ✅ IMPLEMENTED
- **Duration**: 2-3 days (1 GPU)
- **Script**: `scripts/stage1_train_rotate.py`
- **What it does**: Trains KG embeddings via link prediction
- **Output**: `entity_embeddings.pt` (87K × 256-dim)

### Stage 2: Projection Training 🚧 TODO
- **Duration**: 2 hours (1 GPU)
- **Script**: `scripts/stage2_train_projection.py` (NOT YET CREATED)
- **What it does**: Aligns KG embeddings with LLM space
- **Output**: `projection_weights.pt`

### Stage 3: LoRA Fine-tuning 🚧 TODO
- **Duration**: 4-6 hours (1 GPU)
- **Script**: `scripts/stage3_train_lora.py` (NOT YET CREATED)
- **What it does**: Fine-tunes LLM with KG augmentation
- **Output**: `lora_adapter.pt`

---

## Why the Confusion?

The research document originally mentioned reusing embeddings as a **potential optimization** if you're running both BioKG-LoRA and GraphPath-VLM projects.

However, the **actual implementation** trains everything from scratch to make it:
- ✅ Self-contained
- ✅ Reproducible
- ✅ Independent
- ✅ Easier to understand

---

## What Works Right Now

### ✅ You Can Run Today

1. **Quick demo** (5 minutes):
   ```bash
   python scripts/quickstart.py
   ```

2. **Custom KG + RotatE training** (30 minutes):
   ```bash
   python scripts/stage0_build_kg.py --mode dummy --num_genes 500
   python scripts/stage1_train_rotate.py --kg_path data/kg/biological_kg.pt --entity2id_path data/kg/entity2id.json --num_epochs 20
   ```

3. **Test pipeline**:
   ```bash
   python tests/test_end_to_end.py
   ```

### 🚧 Still Need Implementation

- Stage 2 training script
- Stage 3 training script
- Real data source parsers
- QA dataset generation
- Full evaluation suite

---

## Timeline

### From Scratch (Default)

```
╔═══════════════════════════════════════════════════════════╗
║  Complete BioKG-LoRA Pipeline Timeline                   ║
╠═══════════════════════════════════════════════════════════╣
║  Stage 0: Build KG             │ 1-2 days  │ CPU         ║
║  Stage 1: Train RotatE         │ 2-3 days  │ 1 GPU       ║
║  Stage 2: Train Projection     │ 2 hours   │ 1 GPU       ║
║  Stage 3: LoRA Fine-tuning     │ 4-6 hours │ 1 GPU       ║
╠═══════════════════════════════════════════════════════════╣
║  TOTAL                         │ ~1 week   │             ║
╚═══════════════════════════════════════════════════════════╝
```

### If Sharing with GraphPath-VLM (Optional)

If you've already built the KG and trained RotatE for GraphPath-VLM:

```
╔═══════════════════════════════════════════════════════════╗
║  Using Existing Embeddings (Optional)                    ║
╠═══════════════════════════════════════════════════════════╣
║  Stage 0: Build KG             │ SKIP      │ Already done║
║  Stage 1: Train RotatE         │ SKIP      │ Already done║
║  Stage 2: Train Projection     │ 2 hours   │ 1 GPU       ║
║  Stage 3: LoRA Fine-tuning     │ 4-6 hours │ 1 GPU       ║
╠═══════════════════════════════════════════════════════════╣
║  TOTAL                         │ ~1 day    │             ║
╚═══════════════════════════════════════════════════════════╝
```

**But the default implementation trains from scratch!**

---

## Key Points

1. ✅ **Standalone by default** - No dependency on GraphPath-VLM
2. ✅ **Trains from scratch** - Complete 4-stage pipeline
3. ✅ **Stages 0-1 implemented** - KG + RotatE ready
4. 🚧 **Stages 2-3 TODO** - Need training scripts
5. 🔄 **Can share optionally** - If running both projects

---

## Corrected Documentation

These files have been updated to reflect the standalone nature:
- ✅ `mkdocs/docs/research_biokg_lora.md` - Removed "reuse" language
- ✅ `biokg-lora/README.md` - Updated to 4-stage pipeline
- ✅ `biokg-lora/STANDALONE_CLARIFICATION.md` - New clarification doc
- ✅ `biokg-lora/README_CORRECTION.md` - This file

---

## Questions?

**Q: Why did the docs say "reuse"?**  
A: Early drafts mentioned it as an optimization. Implementation is standalone.

**Q: Can I still share embeddings?**  
A: Yes, if you want. But it's not required or assumed.

**Q: What's the recommended approach?**  
A: Train from scratch (default) for reproducibility.

**Q: How long from scratch?**  
A: ~1 week (mostly GPU time for Stage 1).

---

## Bottom Line

🎯 **BioKG-LoRA is a complete, self-contained project that trains all embeddings from scratch.**

No external dependencies. No assumptions. Just run the scripts in order.

✅ Stages 0-1: **READY TO USE**  
🚧 Stages 2-3: **NEED TRAINING SCRIPTS**
