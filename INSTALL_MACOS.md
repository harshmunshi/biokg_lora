# Installation Guide for macOS

**BioKG-LoRA on macOS** - Special instructions for Apple Silicon and Intel Macs

---

## Quick Install (Recommended for macOS)

```bash
cd biokg-lora

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch first (CPU or MPS)
pip install torch torchvision torchaudio

# Install base package (without GPU-specific dependencies)
pip install -e .

# Install optional NLP features (if needed)
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Known Issues on macOS

### 1. ❌ `torch-scatter` and `torch-sparse` Build Failures

**Problem**: These packages require CUDA and don't work well on macOS.

**Solution**: They're optional! Most functionality works without them.

```bash
# Install without GPU-accelerated graph ops
pip install -e .

# If you really need them (advanced users):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 2. ❌ `bitsandbytes` Not Available

**Problem**: QLoRA (4-bit quantization) requires CUDA, not available on macOS.

**Solution**: Use regular LoRA or 8-bit quantization instead.

```bash
# Skip quantization extras on macOS
pip install -e .  # Works without bitsandbytes
```

### 3. ❌ NumPy Binary Incompatibility

**Error**: `ValueError: numpy.dtype size changed`

**Solution**: Reinstall spacy and numpy:

```bash
pip uninstall -y spacy thinc numpy
pip install "numpy>=1.24.0,<2.0.0"
pip install spacy
```

---

## Step-by-Step Installation

### Step 1: Check Python Version

```bash
python3 --version  # Should be 3.10 or 3.11
```

If you need Python 3.10:

```bash
# Using Homebrew
brew install python@3.10

# Or using pyenv
pyenv install 3.10.13
pyenv local 3.10.13
```

### Step 2: Create Virtual Environment

```bash
cd biokg-lora

# Create venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Verify
which python  # Should show .venv/bin/python
```

### Step 3: Install PyTorch

**For Apple Silicon (M1/M2/M3)**:

```bash
# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio
```

**For Intel Macs**:

```bash
# CPU-only PyTorch
pip install torch torchvision torchaudio
```

### Step 4: Install Core Dependencies

```bash
# Install base package
pip install -e .
```

This installs:
- ✅ PyTorch
- ✅ Transformers
- ✅ PEFT (LoRA)
- ✅ NetworkX
- ✅ Matplotlib, Seaborn, Plotly
- ✅ Pandas, NumPy, SciPy
- ❌ No torch-scatter (optional)
- ❌ No bitsandbytes (not available)

### Step 5: Install Optional Dependencies

**NLP Features** (for entity linking):

```bash
pip install spacy
python -m spacy download en_core_web_sm

# Optional: Biomedical NLP (800MB)
python -m spacy download en_core_sci_md
```

**Data Sources**:

```bash
pip install bioservices obonet pronto
```

**Development Tools**:

```bash
pip install black isort mypy pylint pytest pytest-cov
```

---

## Verify Installation

### Quick Test

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import biokg_lora; print('BioKG-LoRA: OK')"

# Run quick demo
python scripts/quickstart.py
```

### Expected Output

```
[1/5] Creating dummy knowledge graph...
✓ KG created: 380 nodes, 1245 edges

[2/5] Creating training dataset...
✓ Dataset created: 996 samples

[3/5] Training RotatE model (10 epochs)...
  Epoch 2/10: Loss = 0.1234
  ...

✅ Quick Start Complete!
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch torchvision torchaudio
```

### Issue: `ValueError: numpy.dtype size changed`

```bash
pip uninstall -y numpy spacy thinc
pip install "numpy>=1.24.0,<2.0.0"
pip install spacy
```

### Issue: `ImportError: cannot import name 'Data' from 'torch_geometric'`

```bash
pip uninstall -y torch-geometric
pip install torch-geometric
```

### Issue: Can't build `torch-scatter`

**Don't install it!** It's optional. Use CPU-only version:

```bash
# Skip GPU dependencies
pip install -e .  # Works without torch-scatter
```

### Issue: Slow training

macOS doesn't have CUDA, so:
- Use smaller models
- Use smaller batch sizes
- Train on CPU/MPS (slower than GPU)
- Or use cloud GPU (AWS, GCP, Lambda Labs)

---

## What Works on macOS

### ✅ Full Functionality

- Stage 0: KG Construction ✅
- Stage 1: RotatE Training ✅ (CPU/MPS)
- Visualization ✅
- Testing ✅
- Model forward pass ✅
- Entity linking ✅

### ⚠️ Limited Functionality

- Training speed: Slower than CUDA GPU
- Large models: May need smaller batches
- QLoRA: Not available (use regular LoRA)

### ❌ Not Available

- 4-bit quantization (bitsandbytes)
- CUDA-accelerated graph ops (torch-scatter)
- Multi-GPU training (DDP)

---

## Recommended Workflow for macOS

### Development & Testing (macOS)

```bash
# Develop code on macOS
python scripts/stage0_build_kg.py --mode dummy --num_genes 100
python scripts/stage1_train_rotate.py --num_epochs 5  # Quick test

# Run tests
python tests/test_end_to_end.py
```

### Full Training (Cloud GPU)

```bash
# For production training, use cloud GPU:
# - Lambda Labs: $0.50/hr (RTX 4090)
# - AWS EC2: p3.2xlarge
# - GCP: a2-highgpu-1g

# Or Colab/Kaggle notebooks
```

---

## Performance on macOS

### Apple Silicon (M1/M2/M3)

**MPS Backend** (Metal Performance Shaders):

```python
import torch

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

**Expected Speed**:
- M1/M2: ~2-3x slower than CUDA GPU
- M3: ~1.5-2x slower than CUDA GPU
- Still much faster than CPU

### Intel Macs

**CPU Only**:

```python
device = torch.device("cpu")
```

**Expected Speed**:
- ~5-10x slower than CUDA GPU
- Recommended: Use cloud GPU for training

---

## Alternative: Docker (If Needed)

If you have persistent issues, use Docker:

```bash
# Pull PyTorch image (x86_64 only)
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Run container
docker run -it -v $(pwd):/workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash

# Inside container
cd /workspace
pip install -e .
```

**Note**: Docker on Apple Silicon uses Rosetta 2 (slower)

---

## Summary

### ✅ Recommended macOS Setup

```bash
# 1. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch
pip install torch torchvision torchaudio

# 3. Install biokg-lora
pip install -e .

# 4. Install spacy (optional)
pip install spacy
python -m spacy download en_core_web_sm

# 5. Test
python scripts/quickstart.py
```

### ❌ Skip These on macOS

- `torch-scatter` (optional, GPU-only)
- `torch-sparse` (optional, GPU-only)
- `bitsandbytes` (not available)

### ⚙️ Use Cloud GPU For

- Stage 1: RotatE training (full 500 epochs)
- Stage 3: LoRA fine-tuning (with QLoRA)
- Large-scale experiments

---

## Getting Help

- **General Issues**: See `docs/INSTALL.md`
- **macOS Specific**: This file
- **GitHub Issues**: Report platform-specific bugs
- **Slack/Discord**: Community support

---

**Last Updated**: January 2026  
**Tested On**: macOS 14 Sonoma (M2, Intel)
