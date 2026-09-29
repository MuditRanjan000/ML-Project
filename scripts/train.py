import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path so src module can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, merge_configs
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.cifar100 import get_cifar100_dataloaders
from src.data.transforms import get_transforms
from src.models.resnet import get_resnet18
from src.models.vit import get_vit_tiny
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Models")
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
    
    # Update seed and metadata
    config["experiment"]["seed"] = args.seed
    config["experiment"]["model_name"] = config.get("model", {}).get("architecture", "unknown")
    config["experiment"]["recipe"] = Path(args.recipe).stem
    set_seed(args.seed)
    
    # Setup Device & Logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = config.get("model", {}).get("architecture", "unknown_model")
    recipe_name = Path(args.recipe).stem
    exp_name = f"{model_name}_{recipe_name}_seed{args.seed}"
    output_dir = Path(config.get("experiment", {}).get("output_dir", "results")) / exp_name
    
    logger = setup_logger("train", str(output_dir))
    logger.info(f"Starting experiment: {exp_name} on {device}")
    
    # Data Setup
    logger.info("Setting up data loaders...")
    train_transform = get_transforms(config, is_training=True)
    eval_transform = get_transforms(config, is_training=False)
    
    # For CI and basic tests, we allow bypassing actual dataloader creation if 
    # we aren't running the full script, but normally this builds real dataloaders
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        config, train_transform, eval_transform
    )
    
    # Model Setup
    logger.info("Initializing model...")
    if "resnet" in model_name:
        model = get_resnet18(config)
    elif "vit" in model_name:
        model = get_vit_tiny(config)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
        
    model = model.to(device)
    
    # Training Components
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints")
    )
    
    # Start Training
    trainer.fit(train_loader, val_loader)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
