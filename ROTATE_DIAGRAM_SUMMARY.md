# RotatE Training Pipeline Diagram - Summary

## 📊 Visual Overview Added to Documentation

A comprehensive ASCII diagram has been added to **Section 7.1.0** of `research_biokg_lora.md` showing the complete RotatE training pipeline.

---

## 🎯 What the Diagram Shows

### 1. INPUT: Knowledge Graph Triples

```
From Stage 0 → biological_kg.pt
├─ 87,452 entities
├─ 1,458,203 triples
└─ 15 relation types

Example triples:
  (Thbd, regulates, Coagulation_Cascade)
  (Bmp4, expressed_in, Kidney)
  (Fgfr2, causes, MP:0003350)  ← phenotype
```

**What goes in**: Raw KG triples from Stage 0 construction

---

### 2. TRAINING OBJECTIVE: Link Prediction

```
Task: Given (head, relation, ?), predict missing tail
      Given (?, relation, tail), predict missing head

Training Strategy:
├─ 1 positive triple:  (Thbd, regulates, Coagulation_Cascade) ✓
└─ 128 negative triples: (Thbd, regulates, Kidney) ✗
                         (Thbd, regulates, Bmp4) ✗
                         ...
```

**What it learns**: Which entity completions are valid vs invalid

---

### 3. MODEL ARCHITECTURE: RotatE (5 Steps)

#### Step 1: Entity Embedding Lookup
```
Entity Embedding Table (87,452 × 256)
  Thbd    → [0.12, -0.34, ..., 0.56]
  Kidney  → [0.67, -0.11, ..., 0.33]
  ...
```

#### Step 2: Represent as Complex Numbers
```
Split 256-dim → 128 complex numbers
  h = [h₀, h₁, ..., h₁₂₇]  where hᵢ = aᵢ + bᵢi
```

#### Step 3: Relation as Rotation
```
Relation Embedding Table (15 × 128 angles)
  regulates → [θ₀, θ₁, ..., θ₁₂₇]
  
Convert to unit circle: rᵢ = e^(iθᵢ) = cos(θᵢ) + i·sin(θᵢ)
```

#### Step 4: Rotate Head by Relation
```
Complex multiplication: h ∘ r
  h_rotated = [h₀·r₀, h₁·r₁, ..., h₁₂₇·r₁₂₇]
  
This rotates h in complex space!
```

#### Step 5: Compute Distance to Tail
```
Score = ||h ∘ r - t||

Low score  → h ∘ r ≈ t → Triple is TRUE  ✓
High score → h ∘ r ≠ t → Triple is FALSE ✗
```

---

### 4. TRAINING LOSS: Self-Adversarial Negative Sampling

```
For each positive triple (h, r, t):

1. Positive score: s⁺ = ||h ∘ r - t||

2. Generate 128 negative samples:
   - 64 by corrupting head: (h', r, t)
   - 64 by corrupting tail: (h, r, t')

3. Negative scores: s⁻ = [s₁⁻, s₂⁻, ..., s₁₂₈⁻]

4. Self-adversarial weighting (focus on hard negatives):
   wᵢ = softmax(-αsᵢ⁻)  ← higher if sᵢ⁻ is low (hard negative)

5. Margin-based loss:
   ℒ = -log σ(γ - s⁺) - Σᵢ wᵢ·log σ(sᵢ⁻ - γ)
   
   where γ = 9.0 (margin)
```

**Key Innovation**: Hard negatives (low score, look plausible) get more weight during training → forces model to learn fine-grained distinctions

---

### 5. OUTPUT: Trained Embeddings

```
After 500 epochs (~3 days on A100):

entity_embeddings.pt
├─ Shape: (87,452, 256)
├─ Size: 85 MB
└─ Properties:
   • Similar entities close in space
   • Semantic relationships preserved
   • Ready for downstream tasks

relation_embeddings.pt
├─ Shape: (15, 128)
└─ Size: 8 KB

Performance:
├─ MRR: 0.68
├─ Hits@1: 52%  (correct entity in top 1)
├─ Hits@3: 78%
└─ Hits@10: 89%
```

---

## 🧠 What These Embeddings Capture

### Entity Embeddings Encode:

```
✓ Gene function (kinase, transcription factor, ...)
✓ Tissue specificity (kidney-expressed, liver, ...)
✓ Pathway membership (coagulation, apoptosis, ...)
✓ Phenotype associations (renal, cardiac, skeletal)
✓ Protein interactions (hub genes vs peripheral)
✓ Evolutionary relationships (gene families)
```

**Example Similarities**:
```
Thbd  ↔ Proc   (both in coagulation cascade)
Bmp4  ↔ Bmp7   (same gene family)
Kidney ↔ Nephron (tissue hierarchy)
```

### Relation Embeddings Encode:

```
✓ Semantic relationship type
✓ Symmetric relations → rotation by π
✓ Antisymmetric relations → unique rotation
✓ Inverse relations (causes ↔ caused_by) → r vs -r
✓ Composition (multi-hop reasoning) → compose rotations
```

---

## 🔗 Connection to Stage 2

```
entity_embeddings.pt → Projection Layer → LLM token space
     (87K, 256)          (256 → 4096)      (87K, 4096)
```

The projection layer (Stage 2) maps these biological embeddings into the LLM's token space, allowing the LLM to "see" and reason about the knowledge graph!

---

## 📐 Why This Diagram Matters

### For Understanding:
- **Visual learners** can see the data flow
- **Step-by-step** breakdown of complex process
- **Concrete examples** at each stage

### For Implementation:
- Clear **input format** specification
- **Architecture details** for coding
- **Expected outputs** for validation

### For Research:
- **Training objective** clearly stated
- **Loss function** fully specified
- **Evaluation metrics** documented

---

## 📍 Location in Documentation

**File**: `/mkdocs/docs/research_biokg_lora.md`

**Section**: 7.1.0 "Training Pipeline Overview" (NEW)

**Line**: ~775 (right before the RotatE architecture code)

**Size**: ~200 lines of ASCII art + explanations

---

## 🎨 Diagram Features

### ✅ What's Included:

1. **Input specification** with example triples
2. **Training objective** (link prediction)
3. **5-step model architecture** with math
4. **Loss function** with formulas
5. **Output format** and metrics
6. **Semantic interpretation** of embeddings
7. **Connection to next stage**

### 📊 Visual Elements:

- **Boxes** for key components
- **Arrows** showing data flow
- **Examples** at each step
- **Math notation** for precision
- **Performance metrics**
- **Intuition** sections

---

## 🚀 Usage

### For Students/Researchers:
Read this diagram **before** diving into the code to understand:
- What RotatE does
- Why it works
- What to expect

### For Implementers:
Use this as a **specification** for:
- Input data format
- Model architecture
- Training loop
- Validation metrics

### For Reviewers:
Reference this to quickly understand:
- The approach
- The training objective
- Expected results

---

## ✨ Key Takeaways

### 1. Clear Data Flow
```
KG Triples → RotatE Model → Entity Embeddings → Stage 2
```

### 2. Training is Link Prediction
```
Learn embeddings such that:
  True triples score LOW  (close in space)
  False triples score HIGH (far in space)
```

### 3. RotatE's Innovation
```
Relations = Rotations in complex space
This allows modeling:
  • Symmetry (rotation by π)
  • Inversion (r vs -r)
  • Composition (multiply rotations)
```

### 4. Self-Adversarial Training
```
Hard negatives get more weight
→ Forces fine-grained learning
→ Better generalization
```

### 5. Biologically Meaningful
```
Embeddings capture:
  Gene function ✓
  Tissue specificity ✓
  Pathway relationships ✓
  Phenotype associations ✓
```

---

## 🎯 Diagram Completeness

| Aspect | Covered? | Detail Level |
|--------|----------|--------------|
| Input format | ✅ | High - with examples |
| Training objective | ✅ | High - with task definition |
| Model architecture | ✅ | Very high - 5 steps with math |
| Loss function | ✅ | High - formula + intuition |
| Training strategy | ✅ | High - negative sampling |
| Output format | ✅ | High - shapes + metrics |
| Semantic meaning | ✅ | High - what's captured |
| Next steps | ✅ | Medium - connection to Stage 2 |

**Overall**: Publication-quality diagram ready for papers/presentations! 🎉
