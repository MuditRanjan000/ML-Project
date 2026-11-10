import torch
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any

from src.calibration.temperature_scaling import fit_temperature_scaling
from src.data.cifar100 import get_cifar100_dataloaders
from src.data.transforms import (
    get_transforms,
    get_evaluation_preprocessing_id,
    get_evaluation_preprocessing_descriptor_sha256,
    validate_evaluation_preprocessing,
    FROZEN_EVALUATION_PREPROCESSING_ID,
    EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256
)
from src.data.cifar100_c import get_cifar100c_dataloader
from src.evaluation.contract import CORRUPTIONS, SEVERITIES, PredictionStore, ResultAccumulator, RunMetadata, load_records, write_tables
from src.training.checkpointing import CheckpointManager

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

        # Contract evaluation strictly fails closed if authoritative latest_model.pt is missing.
        # Standalone best_model.pt + best_metadata.json are not accepted because isolated artifacts
        # cannot independently verify historical training trajectory or validation metric authenticity.
        latest_path = ckpt_dir / "latest_model.pt"
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"Evaluator rejected checkpoint directory '{ckpt_dir}': missing authoritative 'latest_model.pt'. "
                f"Contract evaluation strictly fails closed without 'latest_model.pt' because standalone best "
                f"artifacts cannot independently prove historical training trajectory or validation metric authenticity."
            )

        # 1. Reconcile stale or missing best checkpoint artifacts from authoritative latest checkpoint
        CheckpointManager.reconcile_checkpoints(ckpt_dir, self.config)

        best_path, metadata_path = ckpt_dir / "best_model.pt", ckpt_dir / "best_metadata.json"
        if not best_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("best_model.pt and best_metadata.json are both required.")
        state = torch.load(best_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint '{best_path}' payload is invalid (not a dictionary).")
        best = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(best, dict):
            raise ValueError(f"Metadata sidecar '{metadata_path}' is invalid (not a dictionary).")

        # Validate evaluation preprocessing identity against active frozen configuration
        validate_evaluation_preprocessing(
            config=self.config,
            evaluation_preprocessing_id=state.get("evaluation_preprocessing_id"),
            evaluation_preprocessing_descriptor_sha256=state.get("evaluation_preprocessing_descriptor_sha256")
        )
        validate_evaluation_preprocessing(
            config=self.config,
            evaluation_preprocessing_id=best.get("evaluation_preprocessing_id"),
            evaluation_preprocessing_descriptor_sha256=best.get("evaluation_preprocessing_descriptor_sha256")
        )

        # Validate epoch consistency
        state_epoch = state.get("best_epoch", state.get("epoch"))
        meta_epoch = best.get("best_epoch")
        if state_epoch is None or meta_epoch is None or int(state_epoch) != int(meta_epoch):
            raise ValueError(
                f"Evaluator rejected checkpoint: best_model.pt epoch ({state_epoch}) "
                f"does not match best_metadata.json best_epoch ({meta_epoch})."
            )
        if not isinstance(state_epoch, int) or state_epoch < 1:
            raise ValueError(f"Evaluator rejected checkpoint: invalid best_epoch ({state_epoch}).")
        max_epochs = self.config.get("training", {}).get("epochs", 100)
        if state_epoch > max_epochs:
            raise ValueError(f"Evaluator rejected checkpoint: epoch ({state_epoch}) exceeds configured max epochs ({max_epochs}).")

        # Validate validation metric consistency
        state_metric = state.get("best_metric")
        meta_metric = best.get("best_validation_accuracy")
        if state_metric is None or meta_metric is None or abs(float(state_metric) - float(meta_metric)) > 1e-6:
            raise ValueError(
                f"Evaluator rejected checkpoint: best_model.pt metric ({state_metric}) "
                f"does not match best_metadata.json best_validation_accuracy ({meta_metric})."
            )
        best_percent = float(meta_metric)
        if not 0.0 <= best_percent <= 100.0:
            raise ValueError("Trainer best_validation_accuracy must be a percentage in [0, 100].")

        # Validate split identity
        from src.data.split_manager import get_or_create_split, FROZEN_SPLIT_HASH
        _, _, expected_split_hash = get_or_create_split()
        state_split = state.get("split_hash") or (state.get("run_identity", {}).get("split_hash") if isinstance(state.get("run_identity"), dict) else None)
        meta_split = best.get("split_hash")
        if state_split is not None and state_split != expected_split_hash:
            raise ValueError(f"Evaluator rejected checkpoint: best_model.pt split_hash ('{state_split}') != expected ('{expected_split_hash}').")
        if meta_split is not None and meta_split != expected_split_hash:
            raise ValueError(f"Evaluator rejected checkpoint: best_metadata.json split_hash ('{meta_split}') != expected ('{expected_split_hash}').")

        # Validate run identity and config hash
        expected_arch = self.config.get("experiment", {}).get("model_name") or self.config.get("model", {}).get("architecture")
        expected_recipe = self.config.get("experiment", {}).get("recipe")
        expected_seed = self.config.get("experiment", {}).get("seed")
        from src.training.trainer import compute_config_hash
        expected_config_hash = compute_config_hash(self.config, expected_split_hash)

        if isinstance(state.get("run_identity"), dict):
            s_rid = state["run_identity"]
            if expected_arch and s_rid.get("architecture") and s_rid.get("architecture") != expected_arch:
                raise ValueError(f"Evaluator rejected checkpoint: architecture mismatch (checkpoint has '{s_rid.get('architecture')}', expected '{expected_arch}').")
            if expected_recipe and s_rid.get("recipe") and s_rid.get("recipe") != expected_recipe:
                raise ValueError(f"Evaluator rejected checkpoint: recipe mismatch (checkpoint has '{s_rid.get('recipe')}', expected '{expected_recipe}').")
            if expected_seed is not None and s_rid.get("seed") is not None and s_rid.get("seed") != expected_seed:
                raise ValueError(f"Evaluator rejected checkpoint: seed mismatch (checkpoint has '{s_rid.get('seed')}', expected '{expected_seed}').")
            if s_rid.get("config_hash") and s_rid.get("config_hash") != expected_config_hash:
                raise ValueError(f"Evaluator rejected checkpoint: config_hash mismatch (checkpoint run_identity has '{s_rid.get('config_hash')}', expected '{expected_config_hash}').")

        if state.get("config_hash") and state.get("config_hash") != expected_config_hash:
            raise ValueError(f"Evaluator rejected checkpoint: config_hash mismatch (checkpoint has '{state.get('config_hash')}', expected '{expected_config_hash}').")
        if best.get("config_hash") and best.get("config_hash") != expected_config_hash:
            raise ValueError(f"Evaluator rejected checkpoint: config_hash mismatch (metadata has '{best.get('config_hash')}', expected '{expected_config_hash}').")
        if expected_arch and best.get("model_name") and best.get("model_name") != expected_arch:
            raise ValueError(f"Evaluator rejected checkpoint: architecture mismatch (metadata has '{best.get('model_name')}', expected '{expected_arch}').")
        if expected_recipe and best.get("recipe") and best.get("recipe") != expected_recipe:
            raise ValueError(f"Evaluator rejected checkpoint: recipe mismatch (metadata has '{best.get('recipe')}', expected '{expected_recipe}').")
        if expected_seed is not None and best.get("seed") is not None and best.get("seed") != expected_seed:
            raise ValueError(f"Evaluator rejected checkpoint: seed mismatch (metadata has '{best.get('seed')}', expected '{expected_seed}').")

        # Validate model state payload
        if "model_state" not in state or not isinstance(state["model_state"], dict) or len(state["model_state"]) == 0:
            raise ValueError("Evaluator rejected checkpoint: model_state is missing or empty.")

        self.model.load_state_dict(state["model_state"])
        eval_transform = get_transforms(self.config, is_training=False)
        _, validation_loader, test_loader = get_cifar100_dataloaders(self.config, None, eval_transform)
        from src.data.split_manager import get_or_create_split
        _, _, split_hash = get_or_create_split()

        metadata = RunMetadata(
            run_id=run_id,
            architecture=self.config.get("experiment", {}).get("model_name", "unknown"),
            recipe=self.config.get("experiment", {}).get("recipe", "unknown"),
            training_seed=int(self.config.get("experiment", {}).get("seed", 42)),
            checkpoint_id=best["checkpoint_identifier"],
            selected_epoch=int(best["best_epoch"]),
            best_validation_accuracy_fraction=self.validation_accuracy_fraction(best_percent),
            split_hash=split_hash,
            evaluation_preprocessing_config_id=FROZEN_EVALUATION_PREPROCESSING_ID,
            evaluation_preprocessing_descriptor_sha256=EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
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
