import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

def get_cifar100_dataloaders(
    config: Dict[str, Any],
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test dataloaders for CIFAR-100.
    """
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("data_dir", "data/cifar100"))
    batch_size = config.get("training", {}).get("batch_size", 128)
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)
    seed = config.get("experiment", {}).get("seed", 42)
    val_split = data_config.get("val_split", 0.1) # 10% of 50k = 5k val, 45k train
    
    # Download and load base datasets
    train_full = datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )
    
    # Deterministic validation split
    total_train = len(train_full)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    # Use PyTorch generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size], generator=generator
    )
    
    # The validation set should ideally use the eval_transform. 
    # Since random_split wraps the dataset, we must manually override its transform or 
    # instantiate it twice. For clean implementation, we instantiate the train dataset 
    # again with eval_transform for validation purposes.
    val_dataset_clean = datasets.CIFAR100(
        root=str(data_dir), train=True, download=False, transform=eval_transform
    )
    # Apply the same indices to the clean dataset
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_dataset.indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
