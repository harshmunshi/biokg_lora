#!/bin/bash
# Helper script to train RotatE embeddings using Docker with GPU support

set -e

echo "🐳 BioKG-LoRA Docker Training Script"
echo "===================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not available. Please install nvidia-docker2."
    echo "   See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✅ Docker and NVIDIA runtime detected"
echo ""

# Check if data exists
if [ ! -f "data/kg/biological_kg.pt" ]; then
    echo "❌ Data files not found in data/kg/"
    echo "   Please ensure data files are present before training."
    echo "   Run 'git lfs pull' if using LFS, or generate data with stage0."
    exit 1
fi

if [ ! -f "data/kg/entity2id.json" ]; then
    echo "❌ entity2id.json not found in data/kg/"
    echo "   Required files: biological_kg.pt, entity2id.json"
    exit 1
fi

echo "✅ Data files found:"
ls -lh data/kg/biological_kg.pt data/kg/entity2id.json | awk '{print "   - " $9 ": " $5}'
echo ""

# Create necessary directories
mkdir -p checkpoints logs

echo "📦 Building Docker image..."
docker-compose build

echo ""
echo "🚀 Starting training container..."
echo "   - GPU: Enabled (check nvidia-smi in container)"
echo "   - Data: ./data (read-only)"
echo "   - Checkpoints: ./checkpoints (read-write)"
echo "   - Logs: ./logs (read-write)"
echo ""
echo "📊 Monitor training:"
echo "   - Logs: docker-compose logs -f"
echo "   - TensorBoard: tensorboard --logdir logs/"
echo "   - GPU usage: watch -n 1 nvidia-smi"
echo ""

# Run training
docker-compose up

echo ""
echo "✅ Training complete! Checkpoints saved to ./checkpoints/stage1/"
