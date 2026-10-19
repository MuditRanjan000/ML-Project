import torch
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any

from src.calibration.temperature_scaling import fit_temperature_scaling
from src.data.cifar100 import get_cifar100_dataloaders
from src.data.transforms import get_transforms
from src.data.cifar100_c import get_cifar100c_dataloader
from src.evaluation.contract import CORRUPTIONS, SEVERITIES, PredictionStore, ResultAccumulator, RunMetadata, load_records, write_tables

class Evaluator:
    """
    Orchestrates the entire evaluation pipeline (Clean, Corrupted, Calibration).
    """
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.results_dir = Path(self.config.get("evaluation", {}).get("results_root", "results"))
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "runs").mkdir(exist_ok=True)
        (self.results_dir / "corruption_cells").mkdir(exist_ok=True)
        (self.results_dir / "reliability_bins").mkdir(exist_ok=True)
        
    def run_full_evaluation(self, run_id: str):
        """Backward-compatible entry point for the canonical contract evaluator."""
        return self.run_contract_evaluation(run_id)

    def run_contract_evaluation(self, run_id: str):
        """Run the frozen evaluator with disk-backed records and canonical tables."""
        ckpt_dir = Path(self.config.get("experiment", {}).get("output_dir", "results")) / run_id / "checkpoints"
        best_path, metadata_path = ckpt_dir / "best_model.pt", ckpt_dir / "best_metadata.json"
        if not best_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("best_model.pt and best_metadata.json are both required.")
        state = torch.load(best_path, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        best = json.loads(metadata_path.read_text(encoding="utf-8"))
        best_percent = float(best["best_validation_accuracy"])
        if not 0.0 <= best_percent <= 100.0:
            raise ValueError("Trainer best_validation_accuracy must be a percentage in [0, 100].")
        eval_transform = get_transforms(self.config, is_training=False)
        _, validation_loader, test_loader = get_cifar100_dataloaders(self.config, None, eval_transform)
        from src.data.split_manager import get_or_create_split
        _, _, split_hash = get_or_create_split()
        import hashlib
        metadata = RunMetadata(
            run_id=run_id,
            architecture=self.config.get("experiment", {}).get("model_name", "unknown"),
            recipe=self.config.get("experiment", {}).get("recipe", "unknown"),
            training_seed=int(self.config.get("experiment", {}).get("seed", 42)),
            checkpoint_id=best["checkpoint_identifier"],
            selected_epoch=int(best["best_epoch"]),
            best_validation_accuracy_fraction=self.validation_accuracy_fraction(best_percent),
            split_hash=split_hash,
            evaluation_preprocessing_config_id=hashlib.sha256(repr(eval_transform).encode()).hexdigest(),
        )
        temperature = fit_temperature_scaling(self.model, validation_loader, self.device)
        raw_root = self.results_dir / "predictions"
        accumulator = ResultAccumulator(metadata, temperature)
        self.model.eval()
        with torch.no_grad():
            with PredictionStore.create(raw_root, metadata, record_type="clean_cifar100", expected_samples=10_000, num_classes=100, temperature=temperature) as clean_store:
                offset = 0
                for inputs, labels in test_loader:
                    logits = self.model(inputs.to(self.device)).cpu().numpy()
                    targets = labels.numpy()
                    clean_store.append(np.arange(offset, offset + targets.size), logits, targets)
                    offset += targets.size
                clean_directory = clean_store.finalize()
            _, clean = load_records(clean_directory)
            accumulator.add_clean(clean["logits"], clean["labels"])
            del clean
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    with PredictionStore.create(raw_root, metadata, record_type="cifar100c", corruption=corruption, severity=severity, expected_samples=10_000, num_classes=100, temperature=temperature) as store:
                        offset = 0
                        for inputs, labels in get_cifar100c_dataloader(self.config, corruption, severity, eval_transform):
                            logits = self.model(inputs.to(self.device)).cpu().numpy()
                            targets = labels.numpy()
                            store.append(np.arange(offset, offset + targets.size), logits, targets)
                            offset += targets.size
                        cell_directory = store.finalize()
                    _, cell = load_records(cell_directory)
                    accumulator.add_cell(corruption, severity, cell["logits"], cell["labels"])
                    del cell
        tables = accumulator.finalize()
        write_tables(self.results_dir, metadata, tables)
        return tables

    @staticmethod
    def validation_accuracy_fraction(best_validation_accuracy_percent: float) -> float:
        """Convert trainer's percentage field to a fraction exactly once."""
        if not 0.0 <= best_validation_accuracy_percent <= 100.0:
            raise ValueError("Trainer best_validation_accuracy must be a percentage in [0, 100].")
        return best_validation_accuracy_percent / 100.0
