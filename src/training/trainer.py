import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

from src.training.checkpointing import CheckpointManager
from src.training.amp import AMPManager

class Trainer:
    """
    Functional Training Engine.
    Handles training loops, validation, metric tracking, and checkpointing.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        config: Dict[str, Any], 
        device: torch.device,
        checkpoint_dir: str
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.config = config
        self.device = device
        
        # Logging and Checkpointing
        self.logger = logging.getLogger("Trainer")
        self.ckpt_manager = CheckpointManager(checkpoint_dir)
        
        # AMP
        amp_enabled = config.get("training", {}).get("use_amp", True)
        self.amp = AMPManager(enabled=amp_enabled)
        
    def train_epoch(self, dataloader) -> float:
        """
        Train for a single epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with self.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            scaled_loss = self.amp.scale_loss(loss)
            scaled_loss.backward()
            self.amp.step(self.optimizer)
            
            total_loss += loss.item()
            
        return total_loss / max(len(dataloader), 1)
        
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model on the given dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with self.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = 100. * correct / max(total, 1)
        avg_loss = total_loss / max(len(dataloader), 1)
        
        return {"accuracy": accuracy, "loss": avg_loss}
        
    def fit(self, train_loader, val_loader):
        """
        Full training loop.
        """
        epochs = self.config.get('training', {}).get('epochs', 100)
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_acc = metrics["accuracy"]
            
            self.scheduler.step()
            
            self.logger.info(
                f"Epoch [{epoch}/{epochs}] - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {metrics['loss']:.4f} - "
                f"Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            state_dict = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'metrics': metrics
            }
            extra_metadata = {
                "model_name": self.config.get("experiment", {}).get("model_name", "unknown"),
                "recipe": self.config.get("experiment", {}).get("recipe", "unknown"),
                "seed": self.config.get("experiment", {}).get("seed", 42)
            }
            self.ckpt_manager.save_checkpoint(
                state_dict, epoch, metric=val_acc, extra_metadata=extra_metadata
            )
