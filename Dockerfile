# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
# Extract dependencies from pyproject.toml and install with pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.1.0 \
    transformers>=4.36.0 \
    peft>=0.7.0 \
    accelerate>=0.25.0 \
    torch-geometric>=2.4.0 \
    networkx>=3.2 \
    pandas>=2.1.0 \
    "numpy>=1.24.0,<2.0.0" \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    plotly>=5.18.0 \
    pyvis>=0.3.2 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    rich>=13.7.0 \
    tensorboard>=2.15.0 \
    wandb>=0.16.0

# Install GPU-specific packages for graph operations
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy application code
COPY biokg_lora/ ./biokg_lora/
COPY scripts/ ./scripts/

# Create directories for data and checkpoints with proper permissions
RUN mkdir -p /workspace/data /workspace/checkpoints && \
    chmod -R 777 /workspace/checkpoints

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create a non-root user but allow write access
RUN useradd -m -u 1000 -s /bin/bash trainer && \
    chown -R trainer:trainer /workspace

# Switch to non-root user
USER trainer

# Default command
CMD ["python", "scripts/stage1_train_rotate.py", \
     "--kg_path", "data/kg/biological_kg.pt", \
     "--entity2id_path", "data/kg/entity2id.json"]
