# Docker GPU Training Guide

Train RotatE embeddings using Docker with GPU support - no `uv` or `conda` required!

## Quick Start

### 1. Clone and Transfer Data

```bash
git clone https://github.com/harshmunshi/biokg_lora.git
cd biokg_lora

# Transfer your data files to the data/ directory
# (rsync, scp, or other method - data not in git)
```

### 2. Run Training

```bash
# Easy method - use helper script
./train_docker.sh

# Manual method
docker-compose up
```

## Prerequisites

- Docker installed
- NVIDIA Docker Runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support
- Data files in `data/` directory

## What's Included

- **Dockerfile**: PyTorch 2.1.2 + CUDA 12.1
- **docker-compose.yml**: GPU configuration + volume mounts
- **train_docker.sh**: Automated training script

## Configuration

Edit `docker-compose.yml` to customize training:

```yaml
command: >
  python scripts/stage1_train_rotate.py
  --kg_path data/kg/biological_kg.pt
  --entity2id_path data/kg/entity2id.json
  --num_epochs 500              # ← Change here
  --batch_size 1024             # ← Change here
  --learning_rate 0.0001
```

## File Permissions

Checkpoints are automatically writable by your host user (UID 1000).

## Monitoring

```bash
# View logs
docker-compose logs -f

# Check GPU
nvidia-smi

# TensorBoard
tensorboard --logdir logs/
```

## Troubleshooting

### GPU Not Detected

Install nvidia-docker2:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Out of Memory

Reduce batch size in `docker-compose.yml`:
```yaml
--batch_size 512
```

## Output

After training:
```
checkpoints/stage1/
├── rotate_model.pt
├── entity_embeddings.pt
└── training_metrics.json
```

For full documentation, see the README.
