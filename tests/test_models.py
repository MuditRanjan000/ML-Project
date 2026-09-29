import unittest
import torch
from src.models.resnet import get_resnet18
from src.models.vit import get_vit_tiny
from src.models.model_utils import count_parameters, count_trainable_parameters

class TestModels(unittest.TestCase):
    def setUp(self):
        # Base CIFAR-100 configuration setup
        self.config = {
            "data": {
                "num_classes": 100,
                "image_size": 32
            },
            "model": {
                "pretrained": False,
                "modify_stem": True, # For ResNet
                "patch_size": 4      # For ViT
            }
        }
        # Dummy batch of 8 CIFAR images (32x32)
        self.dummy_input = torch.randn(8, 3, 32, 32)

    def test_resnet18(self):
        model = get_resnet18(self.config)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(self.dummy_input)
            
        # Output shape should be [batch_size, num_classes]
        self.assertEqual(output.shape, (8, 100))
        
        # Verify utility functions run without error
        total_params = count_parameters(model)
        trainable = count_trainable_parameters(model)
        self.assertTrue(total_params > 10_000_000) # ResNet18 is ~11M params
        self.assertEqual(total_params, trainable)

    def test_vit_tiny(self):
        model = get_vit_tiny(self.config)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(self.dummy_input)
            
        # Output shape should be [batch_size, num_classes]
        self.assertEqual(output.shape, (8, 100))
        
        # Verify parameter count logic
        total_params = count_parameters(model)
        trainable = count_trainable_parameters(model)
        self.assertTrue(total_params > 1_000_000) # ViT Tiny is ~5M params
        self.assertEqual(total_params, trainable)

if __name__ == "__main__":
    unittest.main()
