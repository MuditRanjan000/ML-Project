import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch

class CheckpointManager:
    """
    Manages saving and loading model checkpoints with atomic interruption-safe writes.
    Saves best model, latest model, and experiment metadata.
    """
    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.logger = logging.getLogger("CheckpointManager")

    def set_best(self, epoch: int, metric: float) -> None:
        """Restores best checkpoint tracking state upon resumption."""
        self.best_epoch = epoch
        self.best_metric = metric

    def _atomic_torch_save(self, obj: Any, target_path: Path) -> None:
        """
        Safely serializes a PyTorch object to a temporary file before atomically
        replacing the target. Ensures an interrupted write cannot corrupt the target.
        """
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)

    def _atomic_json_save(self, data: Dict[str, Any], target_path: Path) -> None:
        """
        Safely writes JSON metadata to a temporary file, flushes to disk, and
        atomically replaces the destination.
        """
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric: float,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save the model checkpoint (latest and best) with atomic writes.
        Returns True if a new best checkpoint was saved, False otherwise.
        """
        is_new_best = metric > self.best_metric
        if is_new_best:
            self.logger.info(f"New best model found! Metric improved from {self.best_metric:.4f} to {metric:.4f}")
            self.best_metric = metric
            self.best_epoch = epoch

        # 1. Save latest model atomically
        latest_state = dict(state_dict)
        latest_state.setdefault("checkpoint_type", "latest")
        latest_state.setdefault("epoch", epoch)
        latest_state.setdefault("best_metric", self.best_metric)
        latest_state.setdefault("best_epoch", self.best_epoch)

        latest_filename = self.checkpoint_dir / "latest_model.pt"
        self._atomic_torch_save(latest_state, latest_filename)

        # 2. Save latest metadata sidecar atomically
        latest_metadata = {
            "latest_epoch": epoch,
            "latest_training_state": "running",
            "metric": metric,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_metric,
            "checkpoint_identifier": "latest_model.pt"
        }
        if extra_metadata:
            latest_metadata.update(extra_metadata)

        self._atomic_json_save(latest_metadata, self.checkpoint_dir / "latest_metadata.json")

        # 3. If new best, save best model and metadata atomically
        if is_new_best:
            best_state = dict(state_dict)
            best_state["checkpoint_type"] = "best"
            best_state.setdefault("epoch", epoch)
            best_state.setdefault("best_metric", self.best_metric)
            best_state.setdefault("best_epoch", self.best_epoch)

            best_filename = self.checkpoint_dir / "best_model.pt"
            self._atomic_torch_save(best_state, best_filename)

            best_metadata = {
                "best_epoch": epoch,
                "best_validation_accuracy": metric,
                "checkpoint_identifier": "best_model.pt"
            }
            if extra_metadata:
                best_metadata.update(extra_metadata)

            self._atomic_json_save(best_metadata, self.checkpoint_dir / "best_metadata.json")

        return is_new_best

    def load_checkpoint(self, filepath: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
        """
        Load a model checkpoint safely.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location)
