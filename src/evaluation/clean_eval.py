import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.evaluation.metrics import calculate_top1_metrics
from src.calibration.ece import compute_ece

def evaluate_clean(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    temperature: float = 1.0,
    num_ece_bins: int = 15
) -> Dict[str, float]:
    """
    Evaluates clean top-1 accuracy, error, and ECE.
    """
    model.eval()
    correct = 0
    total = 0
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
    metrics = calculate_top1_metrics(correct, total)
    
    # Compute ECE
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    ece_val = compute_ece(logits, targets, temperature=temperature, n_bins=num_ece_bins)
    
    return {
        "clean_top1_acc": metrics["accuracy"],
        "clean_top1_error": metrics["error"],
        "ece": ece_val,
        "logits": logits,
        "targets": targets
    }
