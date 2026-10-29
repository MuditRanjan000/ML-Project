import os
import sys
import json
import copy
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.utils.seed import set_seed
from src.data.transforms import get_transforms
from src.models.resnet import get_resnet18
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Diagnostic requires RTX 3050 CUDA environment.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Running GPU Resumption Diagnostic on: {gpu_name}")

    diag_root = Path("results/diagnostics/resume_tests")
    diag_root.mkdir(parents=True, exist_ok=True)

    # Load and merge base, resnet18, and recipe_a configs
    base_config = load_config("configs/base.yaml")
    model_config = load_config("configs/models/resnet18.yaml")
    recipe_config = load_config("configs/recipes/recipe_a.yaml")

    cfg = merge_configs(base_config, model_config)
    cfg = merge_configs(cfg, recipe_config)

    # Small diagnostic subset configuration (2 epochs, batch size 32)
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 32
    cfg["experiment"]["seed"] = 42
    cfg["experiment"]["model_name"] = "resnet18"
    cfg["experiment"]["recipe"] = "recipe_a"

    # Use actual CIFAR-100 data with production transforms
    train_transform = get_transforms(cfg, is_training=True)
    eval_transform = get_transforms(cfg, is_training=False)

    data_dir = cfg.get("data", {}).get("data_dir", "data/cifar100")
    raw_train = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=train_transform)
    raw_val = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=eval_transform)

    from src.data.split_manager import get_or_create_split
    train_indices, val_indices, split_hash = get_or_create_split(seed=42)

    # Take a small deterministic slice: 256 training samples (8 batches), 64 val samples (2 batches)
    train_subset_indices = train_indices[:256]
    val_subset_indices = val_indices[:64]

    train_ds = Subset(raw_train, train_subset_indices)
    val_ds = Subset(raw_val, val_subset_indices)

    # Note: Using num_workers=0 for deterministic single-process GPU benchmarking
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # -------------------------------------------------------------
    # 1. RUN A: UNINTERRUPTED 2-EPOCH RUN
    # -------------------------------------------------------------
    print("\n--- Starting Run A (Uninterrupted 2 Epochs) ---")
    run_a_dir = diag_root / "gpu_run_a"
    run_a_dir.mkdir(parents=True, exist_ok=True)
    cfg_a = copy.deepcopy(cfg)
    cfg_a["experiment"]["output_dir"] = str(run_a_dir)
    cfg_a["experiment"]["run_id"] = "resnet18_recipe_a_diag_run_a"

    set_seed(42)
    torch.cuda.reset_peak_memory_stats()

    model_a = get_resnet18(cfg_a).to(device)
    opt_a = get_optimizer(model_a, cfg_a)
    sched_a = get_scheduler(opt_a, cfg_a)
    crit_a = nn.CrossEntropyLoss()

    trainer_a = Trainer(
        model=model_a,
        optimizer=opt_a,
        scheduler=sched_a,
        criterion=crit_a,
        config=cfg_a,
        device=device,
        checkpoint_dir=str(run_a_dir / "checkpoints")
    )

    trainer_a.fit(train_loader, val_loader)

    peak_vram_alloc_a = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_reserved_a = torch.cuda.max_memory_reserved() / (1024 ** 2)

    weights_a = {k: v.clone() for k, v in model_a.state_dict().items()}
    opt_state_a = copy.deepcopy(opt_a.state_dict())
    lr_a = sched_a.get_last_lr()[0]
    best_epoch_a = trainer_a.best_epoch
    best_acc_a = trainer_a.best_metric

    # -------------------------------------------------------------
    # 2. RUN B: INTERRUPTED AFTER EPOCH 1, THEN RESUMED FOR EPOCH 2
    # -------------------------------------------------------------
    print("\n--- Starting Run B (Interrupted after Epoch 1, Resumed for Epoch 2) ---")
    run_b_dir = diag_root / "gpu_run_b"
    run_b_dir.mkdir(parents=True, exist_ok=True)
    cfg_b = copy.deepcopy(cfg)
    cfg_b["experiment"]["output_dir"] = str(run_b_dir)
    cfg_b["experiment"]["run_id"] = "resnet18_recipe_a_diag_run_b"

    set_seed(42)
    torch.cuda.reset_peak_memory_stats()

    # Phase 1: Epoch 1 only
    model_b = get_resnet18(cfg_b).to(device)
    opt_b = get_optimizer(model_b, cfg_b)
    sched_b = get_scheduler(opt_b, cfg_b)
    crit_b = nn.CrossEntropyLoss()

    trainer_b1 = Trainer(
        model=model_b,
        optimizer=opt_b,
        scheduler=sched_b,
        criterion=crit_b,
        config=cfg_b,
        device=device,
        checkpoint_dir=str(run_b_dir / "checkpoints")
    )

    trainer_b1.fit(train_loader, val_loader, stop_at_epoch=1)

    # Phase 2: Tear down and instantiate fresh model & optimizer
    del model_b, opt_b, sched_b, trainer_b1
    torch.cuda.empty_cache()

    fresh_model_b = get_resnet18(cfg_b).to(device)
    fresh_opt_b = get_optimizer(fresh_model_b, cfg_b)
    fresh_sched_b = get_scheduler(fresh_opt_b, cfg_b)

    trainer_b2 = Trainer(
        model=fresh_model_b,
        optimizer=fresh_opt_b,
        scheduler=fresh_sched_b,
        criterion=crit_b,
        config=cfg_b,
        device=device,
        checkpoint_dir=str(run_b_dir / "checkpoints")
    )

    latest_ckpt_b = run_b_dir / "checkpoints" / "latest_model.pt"
    print(f"Resuming from: {latest_ckpt_b}")
    already_complete = trainer_b2.resume_from_checkpoint(latest_ckpt_b)
    assert not already_complete, "Run B should not be complete after epoch 1"
    assert trainer_b2.start_epoch == 2, f"Expected start_epoch=2, got {trainer_b2.start_epoch}"

    # Train epoch 2
    trainer_b2.fit(train_loader, val_loader)

    peak_vram_alloc_b = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_reserved_b = torch.cuda.max_memory_reserved() / (1024 ** 2)

    weights_b = {k: v.clone() for k, v in fresh_model_b.state_dict().items()}
    opt_state_b = copy.deepcopy(fresh_opt_b.state_dict())
    lr_b = fresh_sched_b.get_last_lr()[0]
    best_epoch_b = trainer_b2.best_epoch
    best_acc_b = trainer_b2.best_metric

    # -------------------------------------------------------------
    # 3. COMPARISON AND REPORT GENERATION
    # -------------------------------------------------------------
    print("\n--- Diagnostic Comparison: Uninterrupted vs. Resumed ---")

    # Compare weights
    max_weight_diff = 0.0
    total_params = 0
    for k in weights_a:
        diff = (weights_a[k] - weights_b[k]).abs().max().item()
        if diff > max_weight_diff:
            max_weight_diff = diff
        total_params += weights_a[k].numel()

    # Compare optimizer state
    max_momentum_diff = 0.0
    for p_id in opt_state_a.get("state", {}):
        if "momentum_buffer" in opt_state_a["state"][p_id]:
            buf_a = opt_state_a["state"][p_id]["momentum_buffer"]
            buf_b = opt_state_b["state"][p_id]["momentum_buffer"]
            diff = (buf_a - buf_b).abs().max().item()
            if diff > max_momentum_diff:
                max_momentum_diff = diff

    report = {
        "gpu_device": gpu_name,
        "cuda_version": torch.version.cuda,
        "model": "ResNet-18 (CIFAR modified)",
        "recipe": "recipe_a",
        "dataset_subset": "CIFAR-100 (256 train, 64 val)",
        "epochs_tested": 2,
        "run_a_uninterrupted": {
            "best_epoch": best_epoch_a,
            "best_val_accuracy": best_acc_a,
            "final_lr": lr_a,
            "peak_vram_allocated_mb": round(peak_vram_alloc_a, 2),
            "peak_vram_reserved_mb": round(peak_vram_reserved_a, 2)
        },
        "run_b_interrupted_resumed": {
            "resumed_at_epoch": 2,
            "best_epoch": best_epoch_b,
            "best_val_accuracy": best_acc_b,
            "final_lr": lr_b,
            "peak_vram_allocated_mb": round(peak_vram_alloc_b, 2),
            "peak_vram_reserved_mb": round(peak_vram_reserved_b, 2)
        },
        "comparison": {
            "max_parameter_abs_diff": float(max_weight_diff),
            "max_momentum_buffer_abs_diff": float(max_momentum_diff),
            "learning_rate_match": (lr_a == lr_b),
            "best_epoch_match": (best_epoch_a == best_epoch_b),
            "best_val_accuracy_match": (best_acc_a == best_acc_b),
            "vram_allocated_delta_mb": round(abs(peak_vram_alloc_a - peak_vram_alloc_b), 2)
        }
    }

    summary_path = diag_root / "gpu_diagnostic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=2))
    print(f"\nDiagnostic summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
