import torch.nn as nn
from torchvision.models import resnet18
from typing import Dict, Any

def get_resnet18(config: Dict[str, Any]) -> nn.Module:
    """
    Builds a ResNet-18 architecture modified for CIFAR-100.
    
    CIFAR images are 32x32. The standard ResNet from torchvision is designed for 
    224x224 ImageNet images, starting with a 7x7 conv and a MaxPool, which aggressively 
    downsamples the spatial dimensions to 8x8 right away.
    For CIFAR, we replace the first conv with a 3x3 conv (stride 1) and remove the MaxPool.
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    
    num_classes = data_config.get("num_classes", 100)
    pretrained = model_config.get("pretrained", False)
    modify_stem = model_config.get("modify_stem", True)
    
    # We never use ImageNet pretrained weights for the baseline research 
    # unless explicitly allowed. The user specified pretrained=False for Phase H.
    model = resnet18(weights=None)
    
    if modify_stem:
        # Replace the 7x7 conv with a 3x3 conv
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the maxpool (replace with Identity)
        model.maxpool = nn.Identity()
        
    # Modify the fully connected layer for CIFAR-100 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
