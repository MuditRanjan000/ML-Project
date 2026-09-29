import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import from our src.data modules
from src.data.cifar100 import get_cifar100_dataloaders
from src.data.cifar100_c import get_cifar100c_dataloader, CIFAR100C
from src.data.transforms import get_transforms

class DummyCIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        # Create 100 dummy images of size 32x32x3 (HWC, uint8)
        self.data = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
        self.targets = np.random.randint(0, 100, (100,)).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "experiment": {"seed": 42},
            "training": {"batch_size": 16},
            "data": {
                "data_dir": "data_test/cifar100",
                "val_split": 0.2,
                "num_workers": 0,
                "pin_memory": False
            },
            "augmentation": {"type": "standard"}
        }

    @patch("src.data.cifar100.datasets.CIFAR100", DummyCIFAR100)
    def test_cifar100_loaders(self):
        train_transform = get_transforms(self.config, is_training=True)
        eval_transform = get_transforms(self.config, is_training=False)

        train_loader, val_loader, test_loader = get_cifar100_dataloaders(
            self.config, train_transform, eval_transform
        )

        # 100 images * 0.2 val_split = 20 val, 80 train
        # Batch size 16 -> train should have 5 batches, val should have 2 batches (16+4)
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(val_loader.dataset), 20)
        self.assertEqual(len(test_loader.dataset), 100) # Full dummy test set

        # Check batch dimensions and labels
        for batch_imgs, batch_labels in train_loader:
            self.assertEqual(batch_imgs.shape, (16, 3, 32, 32)) # (B, C, H, W)
            self.assertEqual(batch_labels.shape, (16,))
            break # Only check first batch

    @patch("src.data.cifar100_c.np.load")
    @patch("src.data.cifar100_c.Path.exists")
    def test_cifar100c_loader(self, mock_exists, mock_np_load):
        # Mock Path.exists to always return True so it passes the file check
        mock_exists.return_value = True
        
        # Mock numpy arrays for corrupted data
        # Severity 1 means it takes index 0 to 10000. 
        # We need to mock a large enough array for indices up to 10000 to be valid.
        mock_data = np.random.randint(0, 255, (10005, 32, 32, 3), dtype=np.uint8)
        mock_labels = np.random.randint(0, 100, (10005,), dtype=np.int64)
        
        def side_effect_load(path):
            if "labels.npy" in str(path):
                return mock_labels
            return mock_data
            
        mock_np_load.side_effect = side_effect_load
        
        eval_transform = get_transforms(self.config, is_training=False)
        loader = get_cifar100c_dataloader(
            self.config, corruption="gaussian_noise", severity=1, transform=eval_transform
        )
        
        self.assertEqual(len(loader.dataset), 10000)
        
        for batch_imgs, batch_labels in loader:
            self.assertEqual(batch_imgs.shape, (16, 3, 32, 32))
            self.assertEqual(batch_labels.shape, (16,))
            break

if __name__ == "__main__":
    unittest.main()
