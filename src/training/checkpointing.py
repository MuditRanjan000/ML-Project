import torch
from pathlib import Path
from typing import Dict, Any
import json
import logging

class CheckpointManager:
    """
    Manages saving and loading model checkpoints.
    Saves best model, latest model, and experiment metadata.
    """
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf')
        self.logger = logging.getLogger("CheckpointManager")
        
    def save_checkpoint(self, state_dict: Dict[str, Any], epoch: int, metric: float, extra_metadata: Dict[str, Any] = None):
        """
        Save the model checkpoint (latest and best).
        """
        # Save latest
        latest_filename = self.checkpoint_dir / "latest_model.pt"
        torch.save(state_dict, latest_filename)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "metric": metric,
            "best_metric": max(self.best_metric, metric)
        }
        if extra_metadata:
            metadata.update(extra_metadata)
            
        with open(self.checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Save best
        if metric > self.best_metric:
            self.logger.info(f"New best model found! Metric improved from {self.best_metric:.4f} to {metric:.4f}")
            self.best_metric = metric
            best_filename = self.checkpoint_dir / "best_model.pt"
            torch.save(state_dict, best_filename)
            
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        return torch.load(filepath, map_location="cpu")
