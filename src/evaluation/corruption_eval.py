import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.data.cifar100_c import get_cifar100c_dataloader
from src.data.transforms import get_transforms
from src.evaluation.metrics import calculate_top1_metrics

def evaluate_corruptions(
    model: nn.Module, 
    config: Dict[str, Any], 
    device: torch.device, 
    run_id: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates the model across 15 corruptions and 5 severities.
    Generates the corruption_cells table.
    """
    eval_config = config.get("evaluation", {})
    # Standard corruptions per Hendrycks & Dietterich
    default_corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]
    corruptions = eval_config.get("corruptions", default_corruptions)
    severities = eval_config.get("severities", [1, 2, 3, 4, 5])
    
    eval_transform = get_transforms(config, is_training=False)
    
    cells = []
    predictions_dict = {}
    model.eval()
    
    with torch.no_grad():
        for corruption in corruptions:
            # Grouping into families
            family = "noise" if "noise" in corruption else \
                     "blur" if "blur" in corruption else \
                     "weather" if corruption in ["snow", "frost", "fog", "brightness"] else "digital"
                     
            for severity in severities:
                loader = get_cifar100c_dataloader(config, corruption, severity, eval_transform)
                
                correct = 0
                total = 0
                
                all_logits = []
                all_targets = []
                all_preds = []
                all_confs = []
                
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    probs = torch.softmax(outputs, dim=1)
                    confs, predicted = probs.max(1)
                    
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    if eval_config.get("save_predictions", False):
                        all_logits.append(outputs.cpu())
                        all_targets.append(targets.cpu())
                        all_preds.append(predicted.cpu())
                        all_confs.append(confs.cpu())
                        
                metrics = calculate_top1_metrics(correct, total)
                
                if eval_config.get("save_predictions", False):
                    predictions_dict[f"{corruption}_{severity}"] = {
                        "corruption": corruption,
                        "severity": severity,
                        "corruption_index": torch.arange(total),
                        "logits": torch.cat(all_logits, dim=0),
                        "targets": torch.cat(all_targets, dim=0),
                        "predicted_class": torch.cat(all_preds, dim=0),
                        "confidence": torch.cat(all_confs, dim=0)
                    }
                
                cells.append({
                    "run_id": run_id,
                    "corruption": corruption,
                    "family": family,
                    "severity": severity,
                    "n": total,
                    "correct": correct,
                    "incorrect": total - correct,
                    "accuracy_fraction": metrics["accuracy"],
                    "error_fraction": metrics["error"]
                })
                
    return pd.DataFrame(cells), predictions_dict
