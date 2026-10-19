"""CPU-only smoke diagnostic, deliberately separate from pilot and official outputs."""

from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.split_manager import get_or_create_split
from src.data.transforms import get_transforms
from src.evaluation.contract import PredictionStore, RunMetadata, calibration, load_records
from src.evaluation.evaluator import Evaluator
from src.models.resnet import get_resnet18
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs
from src.utils.seed import set_seed


CONFIG_PATH = "configs/diagnostics/cpu_resnet18_recipe_a_seed42.yaml"


def main() -> None:
    config = merge_configs(load_config("configs/base.yaml"), load_config("configs/models/resnet18.yaml"))
    config = merge_configs(config, load_config("configs/recipes/recipe_a.yaml"))
    config = merge_configs(config, load_config(CONFIG_PATH))
    diagnostic = config["diagnostic"]
    run_id = diagnostic["run_id"]
    output_root = Path(config["experiment"]["output_dir"])
    run_root = output_root / run_id
    device = torch.device("cpu")
    set_seed(config["experiment"]["seed"])

    train_transform = get_transforms(config, is_training=True)
    evaluation_transform = get_transforms(config, is_training=False)
    data_root = config["data"]["data_dir"]
    train_data = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
    validation_data = datasets.CIFAR100(data_root, train=True, download=False, transform=evaluation_transform)
    clean_test_data = datasets.CIFAR100(data_root, train=False, download=True, transform=evaluation_transform)
    train_indices, validation_indices, split_hash = get_or_create_split()
    train_subset = Subset(train_data, train_indices[:diagnostic["train_subset_size"]])
    validation_subset = Subset(validation_data, validation_indices[:diagnostic["validation_subset_size"]])
    clean_subset = Subset(clean_test_data, list(range(diagnostic["clean_test_subset_size"])))
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": 0, "pin_memory": False}
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_subset, shuffle=False, **loader_args)
    clean_loader = DataLoader(clean_subset, shuffle=False, **loader_args)

    model = get_resnet18(config)
    optimizer = get_optimizer(model, config)
    trainer = Trainer(model, optimizer, get_scheduler(optimizer, config), nn.CrossEntropyLoss(), config, device,
                      str(run_root / "checkpoints"))
    trainer.fit(train_loader, validation_loader)

    checkpoint = run_root / "checkpoints" / "best_model.pt"
    checkpoint_metadata = run_root / "checkpoints" / "best_metadata.json"
    if not checkpoint.is_file() or not checkpoint_metadata.is_file():
        raise RuntimeError("Diagnostic training did not write validation-selected best checkpoint metadata.")
    best = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state"])
    metadata = RunMetadata(run_id, "resnet18", "recipe_a", 42, best["checkpoint_identifier"],
                           int(best["best_epoch"]), Evaluator.validation_accuracy_fraction(float(best["best_validation_accuracy"])),
                           split_hash, hashlib.sha256(repr(evaluation_transform).encode()).hexdigest())
    predictions_root = Path(config["evaluation"]["results_root"]) / "predictions"
    model.eval()
    with torch.no_grad(), PredictionStore.create(predictions_root, metadata, record_type="clean_cifar100",
                                                 expected_samples=len(clean_subset), num_classes=100,
                                                 temperature=1.0) as store:
        offset = 0
        for inputs, labels in clean_loader:
            logits = model(inputs).cpu().numpy()
            targets = labels.numpy()
            store.append(np.arange(offset, offset + len(targets)), logits, targets)
            offset += len(targets)
        record_directory = store.finalize()
    _, records = load_records(record_directory)
    raw_ece, prediction, _, _ = calibration(records["logits"], records["labels"], 1.0)
    summary = {
        "diagnostic_only": True,
        "run_id": run_id,
        "split_hash": split_hash,
        "train_subset_size": len(train_subset),
        "validation_subset_size": len(validation_subset),
        "clean_test_subset_size": len(clean_subset),
        "checkpoint": str(checkpoint),
        "best_epoch": best["best_epoch"],
        "best_validation_accuracy_percent": best["best_validation_accuracy"],
        "clean_accuracy_fraction": float((prediction == records["labels"]).mean()),
        "clean_ece_raw_fraction": raw_ece,
        "prediction_records": str(record_directory),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
