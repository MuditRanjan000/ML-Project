import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Builds the optimizer from the recipe configuration.
    """
    opt_config = config.get("optimizer", {})
    name = opt_config.get("name", "sgd").lower()
    lr = float(opt_config.get("lr", 0.1))
    weight_decay = float(opt_config.get("weight_decay", 5e-4))
    
    if name == "sgd":
        momentum = float(opt_config.get("momentum", 0.9))
        return optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    elif name == "adamw":
        return optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
