import json
import hashlib
from torchvision import transforms
from typing import Callable, Dict, Any, Optional

def get_evaluation_preprocessing_descriptor(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns the canonical evaluation preprocessing configuration descriptor.
    Binds dataset, image dimensions, normalization constants, and transforms order.
    """
    if config is None:
        data_config = {}
    else:
        data_config = config.get("data", {})

    img_size = int(data_config.get("image_size", 32))
    dataset = str(data_config.get("dataset", "CIFAR-100"))
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    return {
        "dataset": dataset,
        "image_size": img_size,
        "normalization": {
            "mean": mean,
            "std": std
        },
        "transforms": [
            {"type": "ToTensor"},
            {"type": "Normalize", "mean": mean, "std": std}
        ]
    }

def get_evaluation_preprocessing_id(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Computes a deterministic canonical SHA-256 hash of the evaluation preprocessing descriptor.
    """
    desc = get_evaluation_preprocessing_descriptor(config)
    encoded = json.dumps(desc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

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
