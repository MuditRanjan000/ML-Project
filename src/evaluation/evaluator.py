import torch
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any

from src.evaluation.clean_eval import evaluate_clean
from src.evaluation.corruption_eval import evaluate_corruptions
from src.evaluation.metrics import mean_corrupted_error, absolute_error_increase
from src.calibration.temperature_scaling import fit_temperature_scaling
from src.data.cifar100 import get_cifar100_dataloaders
from src.data.transforms import get_transforms

class Evaluator:
    """
    Orchestrates the entire evaluation pipeline (Clean, Corrupted, Calibration).
    """
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "runs").mkdir(exist_ok=True)
        (self.results_dir / "corruption_cells").mkdir(exist_ok=True)
        (self.results_dir / "reliability_bins").mkdir(exist_ok=True)
        
    def run_full_evaluation(self, run_id: str):
        """
        Executes calibration, clean evaluation, and robustness evaluation.
        Loads validation-selected best checkpoint.
        """
        # Load best model
        best_model_path = Path(self.config.get("experiment", {}).get("output_dir", "results")) / run_id / "checkpoints" / "best_model.pt"
        if best_model_path.exists():
            state = torch.load(best_model_path, map_location="cpu")
            self.model.load_state_dict(state["model_state"])
        
        eval_transform = get_transforms(self.config, is_training=False)
        _, val_loader, test_loader = get_cifar100_dataloaders(self.config, None, eval_transform)
        
        # 1. Fit Temperature on Clean Validation Set ONLY
        opt_temp = fit_temperature_scaling(self.model, val_loader, self.device)
        
        # 2. Evaluate Clean Test Set
        num_ece_bins = self.config.get("evaluation", {}).get("ece_bins", 15)
        clean_res_raw = evaluate_clean(self.model, test_loader, self.device, temperature=1.0, num_ece_bins=num_ece_bins)
        clean_res_scaled = evaluate_clean(self.model, test_loader, self.device, temperature=opt_temp, num_ece_bins=num_ece_bins)
        
        # 3. Evaluate CIFAR-100-C
        df_corrupt, predictions = evaluate_corruptions(self.model, self.config, self.device, run_id)
        
        # 4. Calculate Aggregate Metrics
        mce = mean_corrupted_error(df_corrupt["error_fraction"].tolist())
        aei = absolute_error_increase(mce, clean_res_raw["clean_top1_error"])
        
        # 5. Build Summary
        summary = {
            "run_id": run_id,
            "metadata": {
                "model_name": self.config.get("experiment", {}).get("model_name", "unknown"),
                "recipe": self.config.get("experiment", {}).get("recipe", "unknown"),
                "seed": self.config.get("experiment", {}).get("seed", 42)
            },
            "metrics": {
                "Clean Top-1 Acc": clean_res_raw["clean_top1_acc"],
                "Mean Corrupted Error": mce,
                "Absolute Error Increase (AEI)": aei,
                "Raw ECE": clean_res_raw["ece"],
                "Temp-Scaled ECE": clean_res_scaled["ece"],
                "Optimal Temperature": opt_temp
            }
        }
        
        # 6. Save Outputs
        # Save JSON
        with open(self.results_dir / "runs" / f"{run_id}.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        # Save CSV summary
        df_summary = pd.json_normalize(summary)
        df_summary.to_csv(self.results_dir / "runs" / f"{run_id}.csv", index=False)
        
        # Save Corruption Cells
        df_corrupt.to_csv(self.results_dir / "corruption_cells" / f"{run_id}.csv", index=False)
        
        # Save Predictions if enabled
        if self.config.get("evaluation", {}).get("save_predictions", False):
            pred_dir = self.results_dir / "predictions" / run_id
            pred_dir.mkdir(parents=True, exist_ok=True)
            
            # Save clean predictions
            torch.save({
                "logits": clean_res_raw["logits"],
                "targets": clean_res_raw["targets"]
            }, pred_dir / "clean_predictions.pt")
            
            # Save corruption predictions
            for corr_sev, preds in predictions.items():
                torch.save(preds, pred_dir / f"{corr_sev}_predictions.pt")
        
        return summary
