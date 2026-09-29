# A100 Environment Setup & Execution Guide

This document contains the final environment setup and execution instructions for running the real pilot experiment (and subsequent matrix) on the A100 machine.

## 1. Requirements

- **Python Version:** 3.9 - 3.11 (3.10 recommended)
- **CUDA/PyTorch:** CUDA 11.8 or 12.1 is highly recommended for PyTorch 2.3.0

## 2. Installation Commands

Create a clean virtual environment and install the exact pinned dependencies, along with the GPU-enabled PyTorch binaries:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (adjust `--index-url` if you use CUDA 11.8)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining project dependencies
pip install -r requirements.txt
```

## 3. Training Command (Pilot Experiment)

Once the environment is configured and datasets are accessible, start the ResNet-18 Phase K Pilot experiment:

```bash
# Add the project root to PYTHONPATH so `src` module is properly resolved
export PYTHONPATH=$(pwd)

# Run the training script for the pilot experiment
python scripts/train.py \
    --model configs/models/resnet18.yaml \
    --recipe configs/recipes/recipe_a.yaml \
    --seed 42
```

## 4. Evaluation Command

After training completes and the best checkpoint is saved, evaluate the pilot model across the clean and CIFAR-100-C validation sets:

```bash
# Run the standalone evaluation script
python scripts/evaluate.py \
    --model configs/models/resnet18.yaml \
    --recipe configs/recipes/recipe_a.yaml \
    --seed 42
```
