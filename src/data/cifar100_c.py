import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import os

class CIFAR100C(Dataset):
    """
    CIFAR-100-C Dataset.
    Loads corrupted numpy arrays for specific corruptions and severities.
    """
    def __init__(
        self, 
        root: str, 
        corruption: str, 
        severity: int, 
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Validate severity
        if not (1 <= severity <= 5):
            raise ValueError("Severity must be between 1 and 5.")
            
        data_path = self.root / f"{corruption}.npy"
        label_path = self.root / "labels.npy"
        
        if not data_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"CIFAR-100-C files not found at {self.root}. "
                "Ensure dataset is downloaded and extracted."
            )
            
        # Each corruption file contains 50,000 images (10,000 per severity, severities 1 to 5)
        # 10,000 images per severity slice
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        # Load data (images are NHWC uint8)
        full_data = np.load(str(data_path))
        self.data = full_data[start_idx:end_idx]
        
        # Load labels (same labels for all severities)
        full_labels = np.load(str(label_path))
        self.labels = full_labels[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        
        if self.transform is not None:
            # transform typically expects PIL Image or uint8 NHWC array
            img = self.transform(img)
            
        return img, int(label)

def get_cifar100c_dataloader(
    config: Dict[str, Any],
    corruption: str,
    severity: int,
    transform: Optional[Callable] = None
) -> DataLoader:
    """
    Creates a DataLoader for a specific CIFAR-100-C corruption and severity.
    """
    data_config = config.get("data", {})
    # By default look for CIFAR-100-C next to CIFAR-100
    data_dir = Path(data_config.get("data_dir", "data/cifar100"))
    cifar100c_dir = data_dir.parent / "CIFAR-100-C"
    
    batch_size = config.get("training", {}).get("batch_size", 128)
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)
    
    dataset = CIFAR100C(
        root=str(cifar100c_dir),
        corruption=corruption,
        severity=severity,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return loader
