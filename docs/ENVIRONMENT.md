# Environment and Hardware Configuration

This document outlines the strict environment requirements for reproducing the experiments in this repository.

### 1. Recommended Python Version
**Python 3.11** is the officially supported and recommended version for this project. 
* *Why not Python 3.14 (or bleeding edge)?* Pre-compiled `.whl` binaries for `torch==2.3.0` and `numpy==1.26.4` do not exist for highly experimental/pre-release Python versions, causing fallback to massive source-compilation that often fails.
* Research reproducibility strictly relies on pinned library versions. Please downgrade or use a virtual environment locked to Python 3.11.

### 2. CUDA Compatibility Expectations
The project pins PyTorch to `2.3.0`.
* **CUDA 11.8** and **CUDA 12.1** are natively supported by the standard PyTorch 2.3.0 wheels. 
* Ensure your system's NVCC and display drivers meet the minimum versions required for these CUDA toolkits.
* If you run into NCCL issues during multi-GPU training, confirm you are using the cu118/cu121 wheels from the official PyTorch index.

### 3. Local Laptop Development Environment
For writing code, debugging dataloaders, and running sanity checks, local execution (even CPU-only) is fully supported.
* **Setup:** Create a local Python 3.11 `.venv`.
* **Action:** `pip install -r requirements.txt`
* *Note:* Model training on a laptop CPU or low-end GPU is strictly for logic-checking (1 epoch max). Do not attempt to run the full 100-epoch hyperparameter sweeps locally.

### 4. A100 Cluster Environment Recommendation
The final 12 rigorous experiments (Phase K - M) must be executed on an A100 (or equivalent high-end V100/H100) cluster.
* **Environment:** Use a base Docker container like `nvcr.io/nvidia/pytorch:24.03-py3` (which contains PyTorch 2.3) or create a fresh Conda environment mapped to Python 3.11.
* **Data Transfer:** Ensure the CIFAR-100 and CIFAR-100-C datasets are staged on fast NVMe/local node storage, rather than accessed over a slow network filesystem, to prevent I/O bottlenecks.
* **Execution:** Submit jobs using the robust configuration system (`scripts/train.py --model configs/models/resnet18.yaml ...`).
