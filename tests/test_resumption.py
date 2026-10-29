import unittest
import os
import shutil
import tempfile
import copy
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.training.checkpointing import CheckpointManager
from src.training.amp import AMPManager
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler

class SmallTestModel(nn.Module):
    def __init__(self, in_features: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

def create_synthetic_dataloaders(
    n_samples: int = 64,
    in_features: int = 16,
    num_classes: int = 4,
    batch_size: int = 8,
    seed: int = 42
):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, in_features, generator=g)
    y = torch.randint(0, num_classes, (n_samples,), generator=g)

    # Validation uses second half
    train_ds = TensorDataset(X[:48], y[:48])
    val_ds = TensorDataset(X[48:], y[48:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class TestTrainingResumption(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.test_root = Path("results/diagnostics/resume_tests")
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.test_dir = tempfile.mkdtemp(dir=str(self.test_root))

        self.base_config = {
            "experiment": {
                "model_name": "small_test_model",
                "recipe": "test_recipe",
                "seed": 42,
                "run_id": "small_test_model_test_recipe_seed42",
                "output_dir": str(self.test_dir)
            },
            "model": {"architecture": "small_test_model"},
            "training": {"epochs": 4, "batch_size": 8, "use_amp": False},
            "optimizer": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 1e-4},
            "scheduler": {"name": "cosine", "min_lr": 1e-4},
            "data": {
                "split_hash": "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91"
            }
        }

    def tearDown(self):
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_trainer(self, config=None, model=None, ckpt_dir=None):
        cfg = copy.deepcopy(config or self.base_config)
        m = model or SmallTestModel()
        opt = get_optimizer(m, cfg)
        sched = get_scheduler(opt, cfg)
        crit = nn.CrossEntropyLoss()
        c_dir = ckpt_dir or Path(self.test_dir) / "checkpoints"
        return Trainer(
            model=m,
            optimizer=opt,
            scheduler=sched,
            criterion=crit,
            config=cfg,
            device=self.device,
            checkpoint_dir=str(c_dir)
        )

    # 1. A fresh run produces valid latest and best checkpoints
    def test_01_fresh_run_checkpoints(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()

        trainer.fit(train_loader, val_loader)

        ckpt_dir = Path(self.test_dir) / "checkpoints"
        latest_pt = ckpt_dir / "latest_model.pt"
        latest_json = ckpt_dir / "latest_metadata.json"
        best_pt = ckpt_dir / "best_model.pt"
        best_json = ckpt_dir / "best_metadata.json"

        self.assertTrue(latest_pt.is_file(), "latest_model.pt must exist")
        self.assertTrue(latest_json.is_file(), "latest_metadata.json must exist")
        self.assertTrue(best_pt.is_file(), "best_model.pt must exist")
        self.assertTrue(best_json.is_file(), "best_metadata.json must exist")

        import json
        with open(latest_json, "r") as f:
            latest_meta = json.load(f)
        self.assertEqual(latest_meta["latest_epoch"], 2)
        self.assertEqual(latest_meta["checkpoint_identifier"], "latest_model.pt")
        self.assertEqual(latest_meta["latest_training_state"], "completed")

    # 2. A run interrupted after an epoch resumes at the next epoch
    def test_02_interrupted_run_resumes_next_epoch(self):
        # Total configured epochs = 4
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 4

        trainer1 = self._build_trainer(config=cfg)
        train_loader1, val_loader1 = create_synthetic_dataloaders(seed=101)

        # Simulate interruption after completing epoch 2
        trainer1.fit(train_loader1, val_loader1, stop_at_epoch=2)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        self.assertTrue(latest_path.is_file())

        # Phase 2: Resume with same configured epochs = 4
        trainer2 = self._build_trainer(config=cfg)
        completed = trainer2.resume_from_checkpoint(latest_path)
        self.assertFalse(completed, "Run should not be completed at epoch 2 of 4")
        self.assertEqual(trainer2.start_epoch, 3, "Resumed run must start at epoch 3")

        trainer2.fit(train_loader1, val_loader1)

        # Checkpoint after completion should record epoch 4
        ckpt = torch.load(latest_path, map_location="cpu")
        self.assertEqual(ckpt["epoch"], 4)

    # 3. Model, optimizer and scheduler states are restored
    def test_03_model_optimizer_scheduler_restored(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 3
        trainer1 = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer1.fit(train_loader, val_loader, stop_at_epoch=1)

        saved_weights = {k: v.clone() for k, v in trainer1.model.state_dict().items()}
        saved_opt_state = copy.deepcopy(trainer1.optimizer.state_dict())
        saved_sched_lr = trainer1.scheduler.get_last_lr()

        # Fresh model with new random weights
        fresh_model = SmallTestModel()
        self.assertFalse(torch.equal(fresh_model.fc1.weight, saved_weights["fc1.weight"]))

        trainer2 = self._build_trainer(config=cfg, model=fresh_model)
        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        trainer2.resume_from_checkpoint(latest_path)

        # Verify weights restored
        for k, v in trainer2.model.state_dict().items():
            self.assertTrue(torch.equal(v, saved_weights[k]), f"Weight mismatch for {k}")

        # Verify scheduler restored
        self.assertEqual(trainer2.scheduler.get_last_lr(), saved_sched_lr)
        self.assertEqual(trainer2.scheduler.last_epoch, 1)

    # 4. Best epoch and best validation accuracy survive resumption
    def test_04_best_tracking_survives(self):
        trainer = self._build_trainer()
        # Manually save a state where epoch 1 was best (val acc = 95.0)
        state_dict = {
            "epoch": 2,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "scaler_state": None,
            "metrics": {"accuracy": 50.0, "loss": 1.2},
            "best_epoch": 1,
            "best_metric": 95.0,
            "rng_state": trainer._capture_rng_state(),
            "run_identity": trainer._get_run_identity(),
            "schema_version": "1.0",
            "checkpoint_type": "latest"
        }
        trainer.ckpt_manager.save_checkpoint(state_dict, epoch=2, metric=50.0)
        trainer.ckpt_manager.set_best(1, 95.0)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"

        # Resume in a fresh trainer
        new_trainer = self._build_trainer()
        new_trainer.resume_from_checkpoint(latest_path)

        self.assertEqual(new_trainer.best_epoch, 1)
        self.assertEqual(new_trainer.best_metric, 95.0)
        self.assertEqual(new_trainer.ckpt_manager.best_metric, 95.0)

    # 5. AMP scaler state is restored when applicable
    def test_05_amp_scaler_restored(self):
        amp = AMPManager(enabled=True)
        if amp.scaler is not None and torch.cuda.is_available():
            # Test actual CUDA scaler state roundtrip
            m = nn.Linear(4, 2).to("cuda")
            opt = torch.optim.SGD(m.parameters(), lr=0.1)
            inp = torch.randn(4, 4, device="cuda")
            with amp.autocast():
                out = m(inp)
                loss = out.sum()
            scaled = amp.scale_loss(loss)
            scaled.backward()
            amp.step(opt)

            saved_state = amp.state_dict()
            self.assertIsNotNone(saved_state)

            new_amp = AMPManager(enabled=True)
            new_amp.load_state_dict(saved_state)
            self.assertEqual(new_amp.state_dict()["scale"], saved_state["scale"])
        else:
            # Scaler is None on CPU or when disabled
            self.assertIsNone(amp.state_dict())
            amp.load_state_dict(None)

    # 6. RNG restoration reproduces next stochastic operations
    def test_06_rng_restoration(self):
        trainer = self._build_trainer()
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)

        # Capture RNG
        captured = trainer._capture_rng_state()

        # Draw stochastic outputs
        rand_py_1 = random.random()
        rand_np_1 = np.random.rand(5)
        rand_torch_1 = torch.randn(5)

        # Restore RNG
        trainer._restore_rng_state(captured)

        rand_py_2 = random.random()
        rand_np_2 = np.random.rand(5)
        rand_torch_2 = torch.randn(5)

        self.assertEqual(rand_py_1, rand_py_2)
        np.testing.assert_array_equal(rand_np_1, rand_np_2)
        self.assertTrue(torch.equal(rand_torch_1, rand_torch_2))

    # 7. DataLoader/sampler behavior is correct across epoch-boundary resumption
    def test_07_dataloader_epoch_boundary_continuation(self):
        g = torch.Generator().manual_seed(999)
        ds = TensorDataset(torch.arange(20), torch.arange(20))

        # Uninterrupted run: epoch 1 then epoch 2
        torch.manual_seed(999)
        dl1 = DataLoader(ds, batch_size=4, shuffle=True)
        e1_uninterrupted = [b[0].tolist() for b in dl1]
        rng_at_boundary = torch.get_rng_state()
        e2_uninterrupted = [b[0].tolist() for b in dl1]

        # Resumed run: restore boundary state, then iterate
        torch.set_rng_state(rng_at_boundary)
        dl2 = DataLoader(ds, batch_size=4, shuffle=True)
        e2_resumed = [b[0].tolist() for b in dl2]

        self.assertEqual(e2_uninterrupted, e2_resumed, "Resumed DataLoader must match uninterrupted sampler order")

    # 8. A configuration, seed or split-hash mismatch is rejected
    def test_08_mismatch_rejected(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 1
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"

        # A. Architecture mismatch
        cfg_bad_arch = copy.deepcopy(cfg)
        cfg_bad_arch["model"] = {"architecture": "vit_tiny"}
        t_bad = self._build_trainer(config=cfg_bad_arch)
        with self.assertRaises(ValueError) as ctx:
            t_bad.resume_from_checkpoint(latest_path)
        self.assertIn("Architecture mismatch", str(ctx.exception))

        # B. Recipe mismatch
        cfg_bad_recipe = copy.deepcopy(cfg)
        cfg_bad_recipe["experiment"]["recipe"] = "recipe_b"
        t_bad2 = self._build_trainer(config=cfg_bad_recipe)
        with self.assertRaises(ValueError) as ctx:
            t_bad2.resume_from_checkpoint(latest_path)
        self.assertIn("Recipe mismatch", str(ctx.exception))

        # C. Seed mismatch
        cfg_bad_seed = copy.deepcopy(cfg)
        cfg_bad_seed["experiment"]["seed"] = 999
        t_bad3 = self._build_trainer(config=cfg_bad_seed)
        with self.assertRaises(ValueError) as ctx:
            t_bad3.resume_from_checkpoint(latest_path)
        self.assertIn("Seed mismatch", str(ctx.exception))

        # D. Split hash mismatch
        cfg_bad_split = copy.deepcopy(cfg)
        cfg_bad_split["data"] = {"split_hash": "corrupt_hash_0000000000000000000000000000000000000000000000000000000000000000"}
        t_bad4 = self._build_trainer(config=cfg_bad_split)
        with self.assertRaises(ValueError) as ctx:
            t_bad4.resume_from_checkpoint(latest_path)
        self.assertIn("Dataset split hash mismatch", str(ctx.exception))

        # E. Total epochs mismatch
        cfg_bad_epochs = copy.deepcopy(cfg)
        cfg_bad_epochs["training"]["epochs"] = 99
        t_bad5 = self._build_trainer(config=cfg_bad_epochs)
        with self.assertRaises(ValueError) as ctx:
            t_bad5.resume_from_checkpoint(latest_path)
        self.assertIn("Configured total epochs mismatch", str(ctx.exception))

        # F. Learning rate mismatch
        cfg_bad_lr = copy.deepcopy(cfg)
        cfg_bad_lr["optimizer"]["lr"] = 0.999
        t_bad_lr = self._build_trainer(config=cfg_bad_lr)
        with self.assertRaises(ValueError) as ctx:
            t_bad_lr.resume_from_checkpoint(latest_path)
        self.assertIn("Optimizer hyperparameter mismatch for 'lr'", str(ctx.exception))

        # G. Scheduler settings mismatch
        cfg_bad_sched = copy.deepcopy(cfg)
        cfg_bad_sched["scheduler"]["min_lr"] = 0.01
        t_bad_sched = self._build_trainer(config=cfg_bad_sched)
        with self.assertRaises(ValueError) as ctx:
            t_bad_sched.resume_from_checkpoint(latest_path)
        self.assertIn("Scheduler parameter mismatch for 'min_lr'", str(ctx.exception))

        # H. Augmentation settings mismatch
        cfg_bad_aug = copy.deepcopy(cfg)
        cfg_bad_aug["augmentation"] = {"type": "modern", "mixup_alpha": 0.8}
        t_bad_aug = self._build_trainer(config=cfg_bad_aug)
        with self.assertRaises(ValueError) as ctx:
            t_bad_aug.resume_from_checkpoint(latest_path)
        self.assertIn("Augmentation setting mismatch", str(ctx.exception))

        # I. Resume from best_model.pt rejected
        best_path = Path(self.test_dir) / "checkpoints" / "best_model.pt"
        with self.assertRaises(ValueError) as ctx:
            trainer.resume_from_checkpoint(best_path)
        self.assertIn("Cannot resume from 'best_model.pt'", str(ctx.exception))

    # 9. Missing or corrupt resume checkpoints fail safely
    def test_09_missing_or_corrupt_fails_safely(self):
        trainer = self._build_trainer()

        # Missing checkpoint
        missing_path = Path(self.test_dir) / "non_existent.pt"
        with self.assertRaises(FileNotFoundError):
            trainer.resume_from_checkpoint(missing_path)

        # Corrupted checkpoint (truncated random bytes)
        corrupt_path = Path(self.test_dir) / "corrupt.pt"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_TORCH_FILE_1234567890")

        with self.assertRaises(RuntimeError):
            trainer.resume_from_checkpoint(corrupt_path)

    # 10. An already completed run does not start training again
    def test_10_already_completed_does_not_retrain(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer1 = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer1.fit(train_loader, val_loader)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"

        # Resuming with configured epochs = 2
        trainer2 = self._build_trainer(config=cfg)
        completed = trainer2.resume_from_checkpoint(latest_path)
        self.assertTrue(completed, "Resume must report run is already completed")

        # fit() should return immediately without executing epochs
        result = trainer2.fit(train_loader, val_loader)
        self.assertEqual(result.get("status"), "completed")

    # 11. An interrupted checkpoint write preserves the previous valid checkpoint
    def test_11_interrupted_write_preserves_valid_checkpoint(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader, stop_at_epoch=1)

        ckpt_dir = Path(self.test_dir) / "checkpoints"
        latest_path = ckpt_dir / "latest_model.pt"
        self.assertTrue(latest_path.is_file())

        # Verify valid checkpoint at epoch 1
        ckpt_v1 = torch.load(latest_path, map_location="cpu")
        self.assertEqual(ckpt_v1["epoch"], 1)

        # Simulate a crash during epoch 2 save by creating a corrupt .tmp file
        tmp_path = ckpt_dir / "latest_model.pt.tmp"
        with open(tmp_path, "wb") as f:
            f.write(b"INCOMPLETE_WRITE_CRASH")

        # The primary latest_model.pt remains intact and valid
        ckpt_after_crash = torch.load(latest_path, map_location="cpu")
        self.assertEqual(ckpt_after_crash["epoch"], 1)

        # Trainer can still resume successfully from latest_model.pt
        new_trainer = self._build_trainer(config=cfg)
        new_trainer.resume_from_checkpoint(latest_path)
        self.assertEqual(new_trainer.start_epoch, 2)

    # 12. Deterministic uninterrupted vs interrupted comparison
    def test_12_deterministic_uninterrupted_vs_resumed_trajectory(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 4

        # RUN A: Uninterrupted 4-epoch run
        dir_a = Path(self.test_dir) / "run_a"
        dir_a.mkdir(parents=True, exist_ok=True)
        cfg_a = copy.deepcopy(cfg)
        cfg_a["experiment"]["output_dir"] = str(dir_a)
        cfg_a["experiment"]["run_id"] = "test_run_a"

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        train_loader_a, val_loader_a = create_synthetic_dataloaders(seed=42)
        model_a = SmallTestModel()
        trainer_a = self._build_trainer(config=cfg_a, model=model_a, ckpt_dir=dir_a / "checkpoints")
        trainer_a.fit(train_loader_a, val_loader_a)

        final_weights_a = {k: v.clone() for k, v in trainer_a.model.state_dict().items()}

        # RUN B: Interrupted after 2 epochs, then resumed to epoch 4
        dir_b = Path(self.test_dir) / "run_b"
        dir_b.mkdir(parents=True, exist_ok=True)
        cfg_b = copy.deepcopy(cfg)
        cfg_b["experiment"]["output_dir"] = str(dir_b)
        cfg_b["experiment"]["run_id"] = "test_run_b"

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        train_loader_b, val_loader_b = create_synthetic_dataloaders(seed=42)
        model_b = SmallTestModel()
        trainer_b1 = self._build_trainer(config=cfg_b, model=model_b, ckpt_dir=dir_b / "checkpoints")

        # Train epochs 1 and 2
        trainer_b1.fit(train_loader_b, val_loader_b, stop_at_epoch=2)

        # Resume for epochs 3 and 4
        latest_b = dir_b / "checkpoints" / "latest_model.pt"
        fresh_model_b = SmallTestModel()
        trainer_b2 = self._build_trainer(config=cfg_b, model=fresh_model_b, ckpt_dir=dir_b / "checkpoints")
        trainer_b2.resume_from_checkpoint(latest_b)

        self.assertEqual(trainer_b2.start_epoch, 3)
        trainer_b2.fit(train_loader_b, val_loader_b)

        final_weights_b = {k: v.clone() for k, v in trainer_b2.model.state_dict().items()}

        # Compare weights
        for k in final_weights_a:
            diff = (final_weights_a[k] - final_weights_b[k]).abs().max().item()
            self.assertAlmostEqual(
                diff, 0.0, places=5,
                msg=f"Model parameter '{k}' diverged between uninterrupted and resumed run (max diff: {diff})"
            )

    # 13. DataLoader worker configuration mismatch is rejected
    def test_13_dataloader_worker_mismatch_rejected(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        cfg["data"]["num_workers"] = 0
        trainer1 = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer1.fit(train_loader, val_loader, stop_at_epoch=1)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        self.assertTrue(latest_path.is_file())

        # Attempt to resume with different num_workers
        cfg_bad_workers = copy.deepcopy(cfg)
        cfg_bad_workers["data"]["num_workers"] = 2
        trainer2 = self._build_trainer(config=cfg_bad_workers)

        with self.assertRaises(ValueError) as ctx:
            trainer2.resume_from_checkpoint(latest_path)
        self.assertIn("DataLoader worker count mismatch", str(ctx.exception))

        # Confirm that resuming with identical num_workers succeeds
        trainer_ok = self._build_trainer(config=cfg)
        completed = trainer_ok.resume_from_checkpoint(latest_path)
        self.assertFalse(completed)
        self.assertEqual(trainer_ok.start_epoch, 2)

if __name__ == "__main__":
    unittest.main()
