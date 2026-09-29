import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import tempfile
import shutil
from pathlib import Path
import json

from src.evaluation.metrics import calculate_top1_metrics, mean_corrupted_error, absolute_error_increase
from src.calibration.ece import compute_ece
from src.calibration.temperature_scaling import fit_temperature_scaling
from src.evaluation.evaluator import Evaluator

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Outputs fixed logits for deterministic testing
        self.fixed_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        
    def forward(self, x):
        return self.fixed_logits

class TestEvaluationStack(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()
        self.config = {
            "experiment": {"model_name": "dummy", "recipe": "test", "seed": 42},
            "evaluation": {"corruptions": ["gaussian_noise"], "severities": [1], "ece_bins": 15}
        }
        
    def test_metrics(self):
        metrics = calculate_top1_metrics(correct_predictions=80, total_samples=100)
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["error"], 0.2)
        
        mce = mean_corrupted_error([0.5, 0.7])
        self.assertAlmostEqual(mce, 0.6)
        
        aei = absolute_error_increase(0.6, 0.2)
        self.assertAlmostEqual(aei, 0.4)

    def test_ece_calculation(self):
        # 4 samples. 
        # Logits: high confidence for class 0, high conf class 1, etc.
        logits = torch.tensor([
            [10.0, 0.0], # Conf ~1.0, Pred 0
            [0.0, 10.0], # Conf ~1.0, Pred 1
            [0.0, 10.0], # Conf ~1.0, Pred 1
            [10.0, 0.0], # Conf ~1.0, Pred 0
        ])
        
        # 100% accurate targets
        targets = torch.tensor([0, 1, 1, 0])
        ece_perfect = compute_ece(logits, targets)
        self.assertAlmostEqual(ece_perfect, 0.0, places=4)
        
        # 50% accurate targets
        targets_half = torch.tensor([0, 1, 0, 1])
        ece_half = compute_ece(logits, targets_half)
        # Confidence is ~1.0, Accuracy is 0.5. |1.0 - 0.5| = 0.5
        self.assertAlmostEqual(ece_half, 0.5, places=4)
        
    def test_temperature_scaling(self):
        # Create dummy dataloader
        inputs = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([0, 1])
        loader = [(inputs, targets)]
        
        temp = fit_temperature_scaling(self.model, loader, self.device)
        self.assertTrue(isinstance(temp, float))
        self.assertTrue(temp > 0) # Temp should be positive scalar

class TestSplitManager(unittest.TestCase):
    def test_split_persistence(self):
        from src.data.split_manager import get_or_create_split
        import shutil
        
        test_dir = "tests/test_splits"
        Path(test_dir).mkdir(exist_ok=True)
        
        t1, v1, h1 = get_or_create_split(split_dir=test_dir, seed=123)
        self.assertEqual(len(t1), 45000)
        self.assertEqual(len(v1), 5000)
        
        # Call again, should load the same
        t2, v2, h2 = get_or_create_split(split_dir=test_dir, seed=123)
        self.assertEqual(h1, h2)
        self.assertEqual(t1[:10], t2[:10])
        
        shutil.rmtree(test_dir)

    def test_split_hash_reproducibility(self):
        from src.data.split_manager import get_or_create_split
        import shutil
        test_dir = "tests/test_splits_hash"
        Path(test_dir).mkdir(exist_ok=True)
        
        # Test exact deterministic hash calculation
        t1, v1, h1 = get_or_create_split(split_dir=test_dir, seed=42)
        # Expected hash based on the deterministic sorted-keys utf-8 JSON dump
        # We can just verify it is identical across runs on the same numpy seed
        shutil.rmtree(test_dir)
        Path(test_dir).mkdir(exist_ok=True)
        t2, v2, h2 = get_or_create_split(split_dir=test_dir, seed=42)
        
        self.assertEqual(h1, h2)
        
        shutil.rmtree(test_dir)

class TestPredictionSchema(unittest.TestCase):
    def test_clean_predictions(self):
        from src.evaluation.clean_eval import evaluate_clean
        from torch.utils.data import TensorDataset, DataLoader
        
        model = DummyModel()
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 1, 0])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        
        res = evaluate_clean(model, loader, torch.device("cpu"), temperature=1.0)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertIn("image_index", res)
        self.assertEqual(res["predicted_class"].shape[0], 4)
        self.assertEqual(res["confidence"].shape[0], 4)
        self.assertEqual(res["image_index"].shape[0], 4)

if __name__ == "__main__":
    unittest.main()
