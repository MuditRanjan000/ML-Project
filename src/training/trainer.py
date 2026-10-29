import os
import json
import hashlib
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.training.checkpointing import CheckpointManager
from src.training.amp import AMPManager

def compute_config_hash(config: Dict[str, Any], split_hash: Optional[str] = None) -> str:
    """
    Computes a deterministic canonical SHA-256 hash of all experimental settings.
    Normalizes non-experimental fields (such as local filesystem paths or hardware worker counts).
    """
    normalized = {
        "model": config.get("model", {}),
        "training": {
            "epochs": config.get("training", {}).get("epochs", 100),
            "batch_size": config.get("training", {}).get("batch_size", 128),
            "use_amp": config.get("training", {}).get("use_amp", True),
        },
        "optimizer": config.get("optimizer", {}),
        "scheduler": config.get("scheduler", {}),
        "augmentation": config.get("augmentation", {}),
        "data": {
            "dataset": config.get("data", {}).get("dataset", "CIFAR-100"),
            "image_size": config.get("data", {}).get("image_size", 32),
            "num_classes": config.get("data", {}).get("num_classes", 100),
            "num_workers": int(config.get("data", {}).get("num_workers", 4)),
            "persistent_workers": bool(config.get("data", {}).get("persistent_workers", False)),
            "split_hash": split_hash or ""
        },
        "experiment": {
            "seed": config.get("experiment", {}).get("seed", 42),
            "model_name": config.get("model", {}).get(
                "architecture",
                config.get("experiment", {}).get("model_name", "unknown")
            ),
            "recipe": config.get("experiment", {}).get("recipe", "unknown")
        }
    }
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

class Trainer:
    """
    Functional Training Engine with reliable interruption-safe resumption.
    Handles training loops, validation, metric tracking, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Union[str, Path]
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.config = config
        self.device = device

        # Logging and Checkpointing
        self.logger = logging.getLogger("Trainer")
        self.ckpt_manager = CheckpointManager(checkpoint_dir)

        # Resumption & Progression State
        self.start_epoch = 1
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self._saved_rng_state = None

        # AMP
        amp_enabled = config.get("training", {}).get("use_amp", True)
        self.amp = AMPManager(enabled=amp_enabled)

    def _get_seed(self) -> int:
        return int(self.config.get("experiment", {}).get("seed", 42))

    def _get_run_identity(self) -> Dict[str, Any]:
        """Constructs canonical run identity descriptor for validation."""
        from src.data.split_manager import FROZEN_SPLIT_HASH

        model_name = self.config.get("model", {}).get(
            "architecture",
            self.config.get("experiment", {}).get("model_name", "unknown")
        )
        recipe = self.config.get("experiment", {}).get("recipe", "unknown")
        seed = self._get_seed()
        epochs = int(self.config.get("training", {}).get("epochs", 100))
        batch_size = int(self.config.get("training", {}).get("batch_size", 128))
        data_cfg = self.config.get("data", {})
        num_workers = int(data_cfg.get("num_workers", 4))
        persistent_workers = bool(data_cfg.get("persistent_workers", False))
        run_id = self.config.get("experiment", {}).get("run_id") or f"{model_name}_{recipe}_seed{seed}"

        split_hash = self.config.get("data", {}).get("split_hash")
        if not split_hash:
            try:
                from src.data.split_manager import get_or_create_split
                _, _, split_hash = get_or_create_split(seed=seed)
            except Exception:
                split_hash = FROZEN_SPLIT_HASH

        config_hash = compute_config_hash(self.config, split_hash)

        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "architecture": model_name,
            "recipe": recipe,
            "seed": seed,
            "total_epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "split_hash": split_hash,
            "config_hash": config_hash,
            "optimizer": {
                "name": str(self.config.get("optimizer", {}).get("name", "sgd")).lower(),
                "lr": float(self.config.get("optimizer", {}).get("lr", 0.1)),
                "momentum": float(self.config.get("optimizer", {}).get("momentum", 0.9)),
                "weight_decay": float(self.config.get("optimizer", {}).get("weight_decay", 5e-4))
            },
            "scheduler": {
                "name": str(self.config.get("scheduler", {}).get("name", "cosine")).lower(),
                "min_lr": float(self.config.get("scheduler", {}).get("min_lr", 1e-5)),
                "warmup_epochs": int(self.config.get("scheduler", {}).get("warmup_epochs", 0)),
                "step_size": int(self.config.get("scheduler", {}).get("step_size", 30)),
                "gamma": float(self.config.get("scheduler", {}).get("gamma", 0.1))
            },
            "augmentation": {
                "type": str(self.config.get("augmentation", {}).get("type", "standard")),
                "mixup_alpha": float(self.config.get("augmentation", {}).get("mixup_alpha", 0.0)),
                "cutmix_alpha": float(self.config.get("augmentation", {}).get("cutmix_alpha", 0.0)),
                "rand_augment": str(self.config.get("augmentation", {}).get("rand_augment", "")),
                "label_smoothing": float(self.config.get("augmentation", {}).get("label_smoothing", 0.0))
            },
            "model_config": self.config.get("model", {})
        }

    def _capture_rng_state(self, dataloader=None) -> Dict[str, Any]:
        """Captures Python, NumPy, PyTorch CPU, CUDA, and DataLoader generator RNG states."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
            "dataloader_generator": None
        }
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                state["torch_cuda"] = [torch.cuda.get_rng_state()]

        if dataloader is not None and hasattr(dataloader, "generator") and dataloader.generator is not None:
            try:
                state["dataloader_generator"] = dataloader.generator.get_state()
            except Exception:
                pass
        return state

    def _restore_rng_state(self, rng_state: Dict[str, Any], dataloader=None) -> None:
        """Restores all RNG states exactly to preserve deterministic trajectories."""
        if not rng_state:
            return
        if "python" in rng_state and rng_state["python"] is not None:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state and rng_state["numpy"] is not None:
            np.random.set_state(rng_state["numpy"])
        if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
            cpu_rng = rng_state["torch_cpu"]
            if isinstance(cpu_rng, torch.Tensor):
                cpu_rng = cpu_rng.cpu().byte()
            torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None:
            cuda_states = rng_state["torch_cuda"]
            if isinstance(cuda_states, list):
                cuda_states = [s.cpu().byte() if isinstance(s, torch.Tensor) else s for s in cuda_states]
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                elif len(cuda_states) > 0:
                    torch.cuda.set_rng_state(cuda_states[0])
            elif isinstance(cuda_states, torch.Tensor):
                torch.cuda.set_rng_state(cuda_states.cpu().byte())

        if dataloader is not None and hasattr(dataloader, "generator") and dataloader.generator is not None:
            if rng_state.get("dataloader_generator") is not None:
                gen_state = rng_state["dataloader_generator"]
                if isinstance(gen_state, torch.Tensor):
                    gen_state = gen_state.cpu().byte()
                dataloader.generator.set_state(gen_state)

    def _validate_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_path: Path) -> None:
        """Validates checkpoint payload and asserts strict run identity compatibility."""
        if checkpoint_path.name == "best_model.pt" or checkpoint.get("checkpoint_type") == "best":
            raise ValueError(
                f"Cannot resume from '{checkpoint_path.name}'. Resumption must use 'latest_model.pt' to preserve exact training trajectory."
            )

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Corrupt checkpoint at {checkpoint_path}: payload is not a dictionary.")

        if "model_state" not in checkpoint:
            raise ValueError(f"Invalid checkpoint at {checkpoint_path}: missing required key 'model_state'.")

        if "epoch" not in checkpoint:
            raise ValueError(f"Invalid checkpoint at {checkpoint_path}: missing required key 'epoch'.")

        schema_version = checkpoint.get("schema_version")
        if schema_version != "1.0":
            raise ValueError(
                f"Incompatible checkpoint schema version: found '{schema_version}', expected '1.0'."
            )

        saved_id = checkpoint.get("run_identity", {})
        current_id = self._get_run_identity()

        # 1. Base identity checks
        checks = [
            ("architecture", "Architecture mismatch"),
            ("recipe", "Recipe mismatch"),
            ("seed", "Seed mismatch"),
            ("total_epochs", "Configured total epochs mismatch"),
            ("batch_size", "Batch size mismatch"),
            ("num_workers", "DataLoader worker count mismatch"),
            ("persistent_workers", "DataLoader persistent_workers mismatch"),
            ("split_hash", "Dataset split hash mismatch"),
            ("run_id", "Run ID mismatch"),
        ]

        for key, err_msg in checks:
            saved_val = saved_id.get(key)
            curr_val = current_id.get(key)
            if saved_val is not None and curr_val is not None and saved_val != curr_val:
                raise ValueError(
                    f"{err_msg}: checkpoint has {key}='{saved_val}', but current config has {key}='{curr_val}'."
                )

        # 2. Detailed Optimizer Hyperparameter Checks
        saved_opt = saved_id.get("optimizer", {})
        curr_opt = current_id.get("optimizer", {})
        for opt_key in ["name", "lr", "momentum", "weight_decay"]:
            s_val = saved_opt.get(opt_key)
            c_val = curr_opt.get(opt_key)
            if s_val is not None and c_val is not None and s_val != c_val:
                raise ValueError(
                    f"Optimizer hyperparameter mismatch for '{opt_key}': checkpoint has {opt_key}='{s_val}', but current config has {opt_key}='{c_val}'."
                )

        # 3. Detailed Scheduler Parameter Checks
        saved_sched = saved_id.get("scheduler", {})
        curr_sched = current_id.get("scheduler", {})
        for sched_key in ["name", "min_lr", "warmup_epochs", "step_size", "gamma"]:
            s_val = saved_sched.get(sched_key)
            c_val = curr_sched.get(sched_key)
            if s_val is not None and c_val is not None and s_val != c_val:
                raise ValueError(
                    f"Scheduler parameter mismatch for '{sched_key}': checkpoint has {sched_key}='{s_val}', but current config has {sched_key}='{c_val}'."
                )

        # 4. Detailed Augmentation Setting Checks
        saved_aug = saved_id.get("augmentation", {})
        curr_aug = current_id.get("augmentation", {})
        for aug_key in ["type", "mixup_alpha", "cutmix_alpha", "rand_augment", "label_smoothing"]:
            s_val = saved_aug.get(aug_key)
            c_val = curr_aug.get(aug_key)
            if s_val is not None and c_val is not None and s_val != c_val:
                raise ValueError(
                    f"Augmentation setting mismatch for '{aug_key}': checkpoint has {aug_key}='{s_val}', but current config has {aug_key}='{c_val}'."
                )

        # 5. Canonical Config Hash Check (covers all nested model parameters & settings)
        saved_hash = saved_id.get("config_hash") or checkpoint.get("config_hash")
        curr_hash = current_id.get("config_hash")
        if saved_hash is not None and curr_hash is not None and saved_hash != curr_hash:
            raise ValueError(
                f"Experimental configuration mismatch: config hash differs (checkpoint: {saved_hash[:12]}..., current: {curr_hash[:12]}...). Configuration settings differ from saved run."
            )

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Loads checkpoint state and prepares trainer to resume from the next epoch.
        Returns True if training is already completed, False if resumption succeeded.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}. Resumption cannot start without a valid checkpoint."
            )

        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint at {path}: {e}") from e

        self._validate_checkpoint(checkpoint, path)

        completed_epoch = int(checkpoint["epoch"])
        total_epochs = int(self.config.get("training", {}).get("epochs", 100))

        self.logger.info(
            f"Loaded checkpoint from {path} (completed epoch {completed_epoch}/{total_epochs})"
        )

        # 1. Model state
        self.model.load_state_dict(checkpoint["model_state"])

        # 2. Optimizer state
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # 3. Scheduler state
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # 4. AMP Scaler state
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
            self.amp.load_state_dict(checkpoint["scaler_state"])

        # 5. Progress and Model Selection
        self.best_metric = float(checkpoint.get("best_metric", -float('inf')))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.ckpt_manager.set_best(self.best_epoch, self.best_metric)

        # 6. Saved RNG state to be applied right at epoch boundary
        self._saved_rng_state = checkpoint.get("rng_state")

        # 7. Start epoch
        self.start_epoch = completed_epoch + 1

        # Check if already complete
        if completed_epoch >= total_epochs:
            self.logger.info(
                f"Run has already completed all {total_epochs} configured epochs. No further training needed."
            )
            return True

        self.logger.info(
            f"Resuming training from epoch {self.start_epoch} of {total_epochs} (best val acc: {self.best_metric:.2f}% at epoch {self.best_epoch})"
        )
        return False

    def train_epoch(self, dataloader) -> float:
        """
        Train for a single epoch.
        """
        self.model.train()
        total_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            with self.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            scaled_loss = self.amp.scale_loss(loss)
            scaled_loss.backward()
            self.amp.step(self.optimizer)

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model on the given dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with self.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / max(total, 1)
        avg_loss = total_loss / max(len(dataloader), 1)

        return {"accuracy": accuracy, "loss": avg_loss}

    def fit(self, train_loader, val_loader, stop_at_epoch: Optional[int] = None):
        """
        Full training loop with resumption support.
        Optionally halts at `stop_at_epoch` to simulate an interruption.
        """
        epochs = self.config.get('training', {}).get('epochs', 100)
        target_epochs = min(epochs, stop_at_epoch) if stop_at_epoch is not None else epochs

        if self.start_epoch > target_epochs:
            self.logger.info(
                f"Training already complete ({self.start_epoch - 1}/{epochs} epochs). Nothing to train."
            )
            return {"accuracy": self.best_metric, "status": "completed"}

        # Restore RNG state right before entering the epoch loop
        if self._saved_rng_state is not None:
            self.logger.info("Restoring RNG states for exact epoch-boundary continuation...")
            self._restore_rng_state(self._saved_rng_state, train_loader)
            self._saved_rng_state = None

        self.logger.info(f"Starting training from epoch {self.start_epoch} to {target_epochs} (total configured: {epochs})...")

        for epoch in range(self.start_epoch, target_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_acc = metrics["accuracy"]
            metrics["train_loss"] = train_loss

            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch}/{epochs}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {metrics['loss']:.4f} - "
                f"Val Acc: {val_acc:.2f}%"
            )

            # Update best tracking before building checkpoint payload
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch

            # Save checkpoint
            run_id_dict = self._get_run_identity()
            rng_state = self._capture_rng_state(train_loader)
            state_dict = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'scaler_state': self.amp.state_dict(),
                'metrics': metrics,
                'best_epoch': self.best_epoch,
                'best_metric': self.best_metric,
                'rng_state': rng_state,
                'run_identity': run_id_dict,
                'config_hash': run_id_dict.get("config_hash"),
                'schema_version': "1.0",
                'checkpoint_type': "latest"
            }
            is_final_epoch = (epoch >= epochs)
            extra_metadata = {
                "model_name": self.config.get("experiment", {}).get("model_name", "unknown"),
                "recipe": self.config.get("experiment", {}).get("recipe", "unknown"),
                "seed": self.config.get("experiment", {}).get("seed", 42),
                "split_hash": run_id_dict["split_hash"],
                "config_hash": run_id_dict.get("config_hash"),
                "latest_training_state": "completed" if is_final_epoch else "running"
            }
            self.ckpt_manager.save_checkpoint(
                state_dict, epoch, metric=val_acc, extra_metadata=extra_metadata
            )
