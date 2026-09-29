import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import tempfile
import shutil
from pathlib import Path

from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

class TestTrainingEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "training": {"epochs": 2, "batch_size": 4, "use_amp": False},
            "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
            "scheduler": {"name": "cosine", "min_lr": 1e-5}
        }
        self.model = DummyModel()
        self.device = torch.device("cpu")
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_optimizer_factory(self):
        opt = get_optimizer(self.model, self.config)
        self.assertIsInstance(opt, torch.optim.SGD)
        self.assertEqual(opt.defaults["lr"], 0.1)

        adamw_config = {"optimizer": {"name": "adamw", "lr": 0.001}}
        opt = get_optimizer(self.model, adamw_config)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_scheduler_factory(self):
        opt = get_optimizer(self.model, self.config)
        sched = get_scheduler(opt, self.config)
        self.assertIsInstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_trainer_forward_backward_step(self):
        opt = get_optimizer(self.model, self.config)
        sched = get_scheduler(opt, self.config)
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=self.model,
            optimizer=opt,
            scheduler=sched,
            criterion=criterion,
            config=self.config,
            device=self.device,
            checkpoint_dir=self.test_dir
        )
        
        # Create a dummy dataloader (list of tuples)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))
        dummy_loader = [(inputs, targets)]
        
        # Capture initial weights to verify optimization step
        initial_weight = self.model.fc.weight.clone()
        
        # Run one epoch
        loss = trainer.train_epoch(dummy_loader)
        
        # Verify loss is returned and weights updated (backward + optimizer step)
        self.assertIsInstance(loss, float)
        self.assertFalse(torch.equal(initial_weight, self.model.fc.weight))
        
        # Test evaluation
        metrics = trainer.evaluate(dummy_loader)
        self.assertIn("accuracy", metrics)
        self.assertIn("loss", metrics)
        
        # Test checkpoint creation (call save manually)
        trainer.ckpt_manager.save_checkpoint(
            {"model_state": self.model.state_dict()}, epoch=1, metric=metrics["accuracy"]
        )
        
        self.assertTrue((Path(self.test_dir) / "latest_model.pt").exists())
        self.assertTrue((Path(self.test_dir) / "best_model.pt").exists())
        self.assertTrue((Path(self.test_dir) / "best_metadata.json").exists())
        self.assertTrue((Path(self.test_dir) / "latest_metadata.json").exists())
        
        # Verify best metadata contents
        import json
        with open(Path(self.test_dir) / "best_metadata.json") as f:
            best_meta = json.load(f)
        self.assertEqual(best_meta["best_epoch"], 1)
        self.assertEqual(best_meta["best_validation_accuracy"], metrics["accuracy"])
        self.assertEqual(best_meta["checkpoint_identifier"], "best_model.pt")
        
        # Verify latest metadata contents
        with open(Path(self.test_dir) / "latest_metadata.json") as f:
            latest_meta = json.load(f)
        self.assertEqual(latest_meta["latest_epoch"], 1)
        self.assertIn("latest_training_state", latest_meta)

if __name__ == "__main__":
    unittest.main()
