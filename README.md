# Architecture or Training Recipe?
*A Controlled Study of Robustness and Calibration in CNNs and Vision Transformers Under Distribution Shift*

This repository contains the source code, configurations, and reproducible experiments for an empirical study comparing standard CNNs (ResNet-18) against Vision Transformers (ViT-Tiny) on the CIFAR-100-C distribution shift benchmark.

## Repository Organization (Zero-Chaos Standard)
- `src/`: Core training, dataloading, and evaluation logic.
- `configs/`: YAML or programmatic configurations for all runs.
- `scripts/`: Top-level executable scripts (`train.py`, `evaluate.py`).
- `results/`: Contains `raw/`, `processed/`, and `tables/` for experiment outputs.
- `checkpoints/`: Model state dictionaries.
- `experiments/`: Tracked metadata or logs for individual runs.
- `notebooks/`: Exploratory analysis and visualization ONLY.
- `figures/`: Generated plots for the final report.
- `report/`, `presentation/`, `demo/`: Final deliverables.
- `docs/`, `archive/`: Documentation and retired assets.

Please refer to `PROJECT_SPEC.md` and `EXPERIMENT_PLAN.md` for methodology details.
