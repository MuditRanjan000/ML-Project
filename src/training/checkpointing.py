import torch
from pathlib import Path
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Manages saving and loading model checkpoints.
    """
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf')
        
    def save_checkpoint(self, state_dict: Dict[str, Any], epoch: int, metric: float, is_best: bool = False):
        """
        Save the model checkpoint.
        """
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state_dict, filename)
        
        if is_best or metric > self.best_metric:
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
