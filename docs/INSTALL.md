# Installation Guide

Complete installation instructions for BioKG-LoRA.

---

## Quick Install

```bash
# Clone repository
cd biokg-lora

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# Install package
uv pip install -e .

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_md  # Optional, for better biomedical NER
```

---

## Detailed Installation

### 1. System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows 10/11 with WSL2

**Hardware:**
- **CPU**: 8+ cores recommended
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100, RTX 4090, or better)
- **Storage**: 500GB+ free space

### 2. Python Setup

**Python Version**: 3.10 or 3.11

#### Option A: Using `uv` (Recommended)

`uv` is a fast Python package manager.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.10

# Activate
source .venv/bin/activate  # Linux/Mac
```

#### Option B: Using `conda`

```bash
# Create conda environment
conda create -n biokg-lora python=3.10
conda activate biokg-lora
```

#### Option C: Using `venv`

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Install PyTorch

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (testing)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Install PyTorch Geometric

PyTorch Geometric is used for KG operations.

```bash
# Install PyG
pip install torch-geometric

# Install dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Replace `cu118` with your CUDA version.

### 5. Install BioKG-LoRA

```bash
# Install in development mode
pip install -e .

# Or with uv
uv pip install -e .
```

This will install all dependencies from `pyproject.toml`.

### 6. Install Additional Tools

#### spaCy Models

```bash
# Basic English model
python -m spacy download en_core_web_sm

# Biomedical model (optional, 800MB)
python -m spacy download en_core_sci_md
```

#### Jupyter (Optional)

```bash
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=biokg-lora
```

### 7. Verify Installation

Run the quickstart demo:

```bash
python scripts/quickstart.py
```

Expected output:
```
[1/5] Creating dummy knowledge graph...
✓ KG created: 380 nodes, 1245 edges

[2/5] Creating training dataset...
✓ Dataset created: 996 samples

[3/5] Training RotatE model (10 epochs)...
  Epoch 2/10: Loss = 0.1234
  Epoch 4/10: Loss = 0.0987
  ...

✅ Quick Start Complete!
```

### 8. Test Installation

Run tests:

```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_end_to_end.py
```

---

## Troubleshooting

### Issue: `ImportError: cannot import name 'Data' from 'torch_geometric.data'`

**Solution**: Install PyTorch Geometric correctly

```bash
pip uninstall torch-geometric torch-scatter torch-sparse
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Issue: `CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```bash
   python scripts/stage1_train_rotate.py --batch_size 256
   ```

2. Use gradient accumulation:
   ```bash
   python scripts/stage3_train_lora.py --gradient_accumulation_steps 4
   ```

3. Use 8-bit or 4-bit quantization:
   ```bash
   python scripts/stage3_train_lora.py --load_in_8bit
   ```

### Issue: `spaCy model not found`

**Solution**: Install spaCy models

```bash
python -m spacy download en_core_web_sm
```

### Issue: `ModuleNotFoundError: No module named 'biokg_lora'`

**Solution**: Install package in development mode

```bash
pip install -e .
```

### Issue: Slow training

**Solutions**:
1. Use more workers:
   ```bash
   python scripts/stage1_train_rotate.py --num_workers 8
   ```

2. Use mixed precision:
   ```bash
   python scripts/stage1_train_rotate.py --fp16
   ```

3. Use multiple GPUs:
   ```bash
   torchrun --nproc_per_node=2 scripts/stage1_train_rotate.py
   ```

---

## Platform-Specific Notes

### Linux

Should work out of the box with NVIDIA drivers installed.

### macOS

- No CUDA support (CPU or MPS only)
- Install with CPU-only PyTorch:
  ```bash
  pip install torch torchvision torchaudio
  ```
- Use smaller models and batch sizes

### Windows

- Use WSL2 for best compatibility
- Native Windows: May have issues with some dependencies
- Recommended: Install under WSL2

---

## Optional Dependencies

### For Visualization

```bash
pip install pyvis plotly networkx
```

### For Development

```bash
pip install black isort mypy pylint pytest pytest-cov
```

### For Documentation

```bash
pip install mkdocs mkdocs-material
```

---

## Next Steps

After installation:

1. **Quick Start**: `python scripts/quickstart.py`
2. **Read QUICKSTART.md**: 5-minute demo guide
3. **Download Data**: See `DATA_SOURCES.md`
4. **Train Models**: See `TRAINING.md`

---

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See `docs/` folder
- **Tests**: Run `pytest tests/` to verify setup

---

## Uninstallation

```bash
# Remove virtual environment
rm -rf .venv

# Or with conda
conda env remove -n biokg-lora
```
