import json
import hashlib
from torchvision import transforms
from typing import Callable, Dict, Any, Optional

FROZEN_EVALUATION_PREPROCESSING_ID = "5256477cf00b8bcaec855fb96067bcbe6e718235b083b92261dd407b501f989e"
EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256 = "dc9cb66d30e90972331e408340d0f00d8006e829665c79ddc254c7bb853f0862"

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

    # Extract normalization constants (default to frozen CIFAR-100 constants)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    norm_cfg = data_config.get("normalization", {})
    if isinstance(norm_cfg, dict):
        if "mean" in norm_cfg:
            mean = list(norm_cfg["mean"])
        if "std" in norm_cfg:
            std = list(norm_cfg["std"])
    if "mean" in data_config:
        mean = list(data_config["mean"])
    if "std" in data_config:
        std = list(data_config["std"])

    # Extract transform ordering (default to ToTensor -> Normalize)
    transforms_order = data_config.get("eval_transforms_order", ["ToTensor", "Normalize"])
    transforms_list = []
    for t in transforms_order:
        if t == "ToTensor":
            transforms_list.append({"type": "ToTensor"})
        elif t == "Normalize":
            transforms_list.append({"mean": mean, "std": std, "type": "Normalize"})
        else:
            transforms_list.append({"type": str(t)})

    return {
        "dataset": dataset,
        "image_size": img_size,
        "normalization": {
            "mean": mean,
            "std": std
        },
        "transforms": transforms_list
    }

def get_evaluation_preprocessing_descriptor_sha256(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Computes a deterministic canonical SHA-256 fingerprint of the evaluation preprocessing descriptor.
    """
    desc = get_evaluation_preprocessing_descriptor(config)
    encoded = json.dumps(desc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def get_evaluation_preprocessing_id(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Returns the project's canonical frozen approved evaluation preprocessing protocol identifier.
    Must always remain the frozen approved constant: '5256477cf00b8bcaec855fb96067bcbe6e718235b083b92261dd407b501f989e'.
    """
    return FROZEN_EVALUATION_PREPROCESSING_ID

def validate_evaluation_preprocessing(
    config: Optional[Dict[str, Any]] = None,
    evaluation_preprocessing_id: Optional[str] = None,
    evaluation_preprocessing_descriptor_sha256: Optional[str] = None
) -> None:
    """
    Validates evaluation preprocessing against the frozen protocol identifier and canonical descriptor fingerprint.

    Checks:
    1. Reconstructed active descriptor SHA-256 matches EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256.
    2. Checkpoint/metadata descriptor SHA-256 (if provided) matches EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256.
    3. Checkpoint/metadata evaluation_preprocessing_id (if provided) matches FROZEN_EVALUATION_PREPROCESSING_ID.

    Raises ValueError on any mismatch.
    """
    active_sha256 = get_evaluation_preprocessing_descriptor_sha256(config)
    if active_sha256 != EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256:
        active_desc = get_evaluation_preprocessing_descriptor(config)
        raise ValueError(
            f"Evaluation preprocessing configuration mismatch: active descriptor SHA-256 is '{active_sha256}', "
            f"expected '{EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256}'. "
            f"Active descriptor: {json.dumps(active_desc, sort_keys=True)}"
        )

    if evaluation_preprocessing_descriptor_sha256 is not None:
        if evaluation_preprocessing_descriptor_sha256 != EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256:
            raise ValueError(
                f"Evaluation preprocessing descriptor SHA-256 mismatch: "
                f"expected '{EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256}', "
                f"got '{evaluation_preprocessing_descriptor_sha256}'."
            )

    if evaluation_preprocessing_id is not None:
        if evaluation_preprocessing_id != FROZEN_EVALUATION_PREPROCESSING_ID:
            raise ValueError(
                f"Evaluation preprocessing protocol ID mismatch: "
                f"expected '{FROZEN_EVALUATION_PREPROCESSING_ID}', "
                f"got '{evaluation_preprocessing_id}'."
            )

def get_transforms(config: Dict[str, Any], is_training: bool = False) -> Callable:
    """
    Builds torchvision transforms based on configuration.
    Separates purely deterministic resizing/normalization from augmentations.
    """
    data_config = config.get("data", {})
    aug_config = config.get("augmentation", {})

    img_size = data_config.get("image_size", 32)

    # Read normalization constants
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    norm_cfg = data_config.get("normalization", {})
    if isinstance(norm_cfg, dict):
        if "mean" in norm_cfg:
            mean = tuple(norm_cfg["mean"])
        if "std" in norm_cfg:
            std = tuple(norm_cfg["std"])
    if "mean" in data_config:
        mean = tuple(data_config["mean"])
    if "std" in data_config:
        std = tuple(data_config["std"])
    
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
            return transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    else:
        # Evaluation transforms (Clean and CIFAR-100-C)
        eval_order = data_config.get("eval_transforms_order", ["ToTensor", "Normalize"])
        ops = []
        for t in eval_order:
            if t == "ToTensor":
                ops.append(transforms.ToTensor())
            elif t == "Normalize":
                ops.append(transforms.Normalize(mean, std))
            else:
                raise ValueError(f"Unknown evaluation transform type: {t}")
        return transforms.Compose(ops)
