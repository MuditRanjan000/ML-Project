import argparse
import sys
from pathlib import Path
import torch
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.evaluation.evaluator import Evaluator
from src.models.resnet import get_resnet18
from src.models.vit import get_vit_tiny

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Models")
    parser.add_argument("--model", type=str, required=True, help="Path to model config")
    parser.add_argument("--recipe", type=str, required=True, help="Path to recipe config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load and merge configurations
    base_config = load_config("configs/base.yaml")
    model_config = load_config(args.model)
    recipe_config = load_config(args.recipe)
    
    config = merge_configs(base_config, model_config)
    config = merge_configs(config, recipe_config)
    
    config["experiment"]["seed"] = args.seed
    model_name = config.get("model", {}).get("architecture", "unknown")
    config["experiment"]["model_name"] = model_name
    recipe_name = Path(args.recipe).stem
    config["experiment"]["recipe"] = recipe_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{model_name}_{recipe_name}_seed{args.seed}"
    
    print(f"Starting evaluation for {run_id} on {device}")
    
    if "resnet" in model_name:
        model = get_resnet18(config)
    elif "vit" in model_name:
        model = get_vit_tiny(config)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
        
    evaluator = Evaluator(model, config, device)
    summary = evaluator.run_full_evaluation(run_id)
    
    print("\nEvaluation Summary:")
    print(json.dumps(summary["metrics"], indent=4))

if __name__ == "__main__":
    main()
