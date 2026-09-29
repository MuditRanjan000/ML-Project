from torchvision import transforms
from typing import Callable, Dict, Any

def get_transforms(config: Dict[str, Any], is_training: bool = False) -> Callable:
    """
    Builds torchvision transforms based on configuration.
    Separates purely deterministic resizing/normalization from augmentations.
    """
    data_config = config.get("data", {})
    aug_config = config.get("augmentation", {})
    
    img_size = data_config.get("image_size", 32)
    # Standard CIFAR-100 normalization values
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    if is_training:
        aug_type = aug_config.get("type", "standard")
        
        if aug_type == "standard":
            # Recipe A: Conventional Baseline
            return transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif aug_type == "modern":
            # Recipe B: Modern Composite 
            # (Note: complex augments like mixup/cutmix happen at batch level in trainer,
            # rand_augment happens here)
            return transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                # torchvision v2 transforms can support RandAugment, but for classic 
                # baseline we use standard RandAugment if installed, else fallback.
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    else:
        # Evaluation transforms (Clean and CIFAR-100-C)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
