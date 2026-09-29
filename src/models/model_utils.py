import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Prints a basic summary of the model parameter counts.
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    print(f"--- {model_name} Summary ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable:        {total_params - trainable_params:,}")
    print("-" * 25)
