import torch.nn as nn
import timm
from typing import Dict, Any

def get_vit_tiny(config: Dict[str, Any]) -> nn.Module:
    """
    Builds a ViT-Tiny architecture for CIFAR-100 using timm.
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    
    num_classes = data_config.get("num_classes", 100)
    pretrained = model_config.get("pretrained", False)
    patch_size = model_config.get("patch_size", 4)
    img_size = data_config.get("image_size", 32)
    
    # Use timm to build the vision transformer.
    # We use a tiny variant. For 32x32 images, patch_size=4 is standard, yielding 8x8=64 patches.
    # Alternatively we can use timm's built-in vit_tiny_patch16_224 and override the img_size 
    # and patch_size.
    model = timm.create_model(
        'vit_tiny_patch16_224',  # Base blueprint
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,       # Override image size for CIFAR
        patch_size=patch_size    # Override patch size for CIFAR
    )
    
    return model
