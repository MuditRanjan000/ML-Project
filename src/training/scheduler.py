import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from typing import Dict, Any

def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> lr_scheduler.LRScheduler:
    """
    Builds the learning rate scheduler from the recipe configuration.
    """
    sched_config = config.get("scheduler", {})
    name = sched_config.get("name", "cosine").lower()
    
    # We retrieve epochs from training config for T_max
    epochs = config.get("training", {}).get("epochs", 100)
    min_lr = float(sched_config.get("min_lr", 1e-5))
    
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=min_lr
        )
    elif name == "step":
        step_size = sched_config.get("step_size", 30)
        gamma = sched_config.get("gamma", 0.1)
        return lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
