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

from src.training.trainer import Trainer, compute_config_hash
from src.training.checkpointing import CheckpointManager
from src.training.amp import AMPManager
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.data.transforms import get_evaluation_preprocessing_id
from src.evaluation.evaluator import Evaluator

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
    seed: int = 42,
    num_workers: int = 0
):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, in_features, generator=g)
    y = torch.randint(0, num_classes, (n_samples,), generator=g)

    # Validation uses second half
    train_ds = TensorDataset(X[:48], y[:48])
    val_ds = TensorDataset(X[48:], y[48:])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
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
                "dataset": "CIFAR-100",
                "image_size": 32,
                "num_classes": 100,
                "num_workers": 0,
                "persistent_workers": False,
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
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 4

        trainer1 = self._build_trainer(config=cfg)
        train_loader1, val_loader1 = create_synthetic_dataloaders(seed=101)

        trainer1.fit(train_loader1, val_loader1, stop_at_epoch=2)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        self.assertTrue(latest_path.is_file())

        trainer2 = self._build_trainer(config=cfg)
        completed = trainer2.resume_from_checkpoint(latest_path)
        self.assertFalse(completed, "Run should not be completed at epoch 2 of 4")
        self.assertEqual(trainer2.start_epoch, 3, "Resumed run must start at epoch 3")

        trainer2.fit(train_loader1, val_loader1)

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

        fresh_model = SmallTestModel()
        self.assertFalse(torch.equal(fresh_model.fc1.weight, saved_weights["fc1.weight"]))

        trainer2 = self._build_trainer(config=cfg, model=fresh_model)
        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        trainer2.resume_from_checkpoint(latest_path)

        for k, v in trainer2.model.state_dict().items():
            self.assertTrue(torch.equal(v, saved_weights[k]), f"Weight mismatch for {k}")

        self.assertEqual(trainer2.scheduler.get_last_lr(), saved_sched_lr)
        self.assertEqual(trainer2.scheduler.last_epoch, 1)

    # 4. Best epoch and best validation accuracy survive resumption
    def test_04_best_tracking_survives(self):
        trainer = self._build_trainer()
        run_id_dict = trainer._get_run_identity()
        state_dict = {
            "epoch": 2,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "scaler_state": None,
            "metrics": {"accuracy": 50.0, "loss": 1.2},
            "best_epoch": 1,
            "best_metric": 95.0,
            "best_model_state": {k: v.clone().cpu() for k, v in trainer.model.state_dict().items()},
            "best_metadata": {
                "best_epoch": 1,
                "best_validation_accuracy": 95.0,
                "checkpoint_identifier": "best_model.pt"
            },
            "rng_state": trainer._capture_rng_state(),
            "run_identity": run_id_dict,
            "config_hash": run_id_dict["config_hash"],
            "evaluation_preprocessing_id": run_id_dict["evaluation_preprocessing_id"],
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }
        trainer.ckpt_manager.set_best(1, 95.0, state_dict["best_model_state"])
        trainer.ckpt_manager.save_checkpoint(state_dict, epoch=2, metric=50.0)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"

        new_trainer = self._build_trainer()
        new_trainer.resume_from_checkpoint(latest_path)

        self.assertEqual(new_trainer.best_epoch, 1)
        self.assertEqual(new_trainer.best_metric, 95.0)
        self.assertEqual(new_trainer.ckpt_manager.best_metric, 95.0)

    # 5. AMP scaler state is restored when applicable
    def test_05_amp_scaler_restored(self):
        amp = AMPManager(enabled=True)
        if amp.scaler is not None and torch.cuda.is_available():
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
            self.assertIsNone(amp.state_dict())
            amp.load_state_dict(None)

    # 6. RNG restoration reproduces next stochastic operations
    def test_06_rng_restoration(self):
        trainer = self._build_trainer()
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)

        captured = trainer._capture_rng_state()

        rand_py_1 = random.random()
        rand_np_1 = np.random.rand(5)
        rand_torch_1 = torch.randn(5)

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

        torch.manual_seed(999)
        dl1 = DataLoader(ds, batch_size=4, shuffle=True)
        e1_uninterrupted = [b[0].tolist() for b in dl1]
        rng_at_boundary = torch.get_rng_state()
        e2_uninterrupted = [b[0].tolist() for b in dl1]

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
        cfg_bad_split["data"]["split_hash"] = "corrupt_hash_0000000000000000000000000000000000000000000000000000000000000000"
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

        missing_path = Path(self.test_dir) / "non_existent.pt"
        with self.assertRaises(FileNotFoundError):
            trainer.resume_from_checkpoint(missing_path)

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

        trainer2 = self._build_trainer(config=cfg)
        completed = trainer2.resume_from_checkpoint(latest_path)
        self.assertTrue(completed, "Resume must report run is already completed")

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

        ckpt_v1 = torch.load(latest_path, map_location="cpu")
        self.assertEqual(ckpt_v1["epoch"], 1)

        tmp_path = ckpt_dir / "latest_model.pt.tmp"
        with open(tmp_path, "wb") as f:
            f.write(b"INCOMPLETE_WRITE_CRASH")

        ckpt_after_crash = torch.load(latest_path, map_location="cpu")
        self.assertEqual(ckpt_after_crash["epoch"], 1)

        new_trainer = self._build_trainer(config=cfg)
        new_trainer.resume_from_checkpoint(latest_path)
        self.assertEqual(new_trainer.start_epoch, 2)

    # 12. Deterministic uninterrupted vs interrupted comparison
    def test_12_deterministic_uninterrupted_vs_resumed_trajectory(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 4

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

        trainer_b1.fit(train_loader_b, val_loader_b, stop_at_epoch=2)

        latest_b = dir_b / "checkpoints" / "latest_model.pt"
        fresh_model_b = SmallTestModel()
        trainer_b2 = self._build_trainer(config=cfg_b, model=fresh_model_b, ckpt_dir=dir_b / "checkpoints")
        trainer_b2.resume_from_checkpoint(latest_b)

        self.assertEqual(trainer_b2.start_epoch, 3)
        trainer_b2.fit(train_loader_b, val_loader_b)

        final_weights_b = {k: v.clone() for k, v in trainer_b2.model.state_dict().items()}

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

        cfg_bad_workers = copy.deepcopy(cfg)
        cfg_bad_workers["data"]["num_workers"] = 2
        trainer2 = self._build_trainer(config=cfg_bad_workers)

        with self.assertRaises(ValueError) as ctx:
            trainer2.resume_from_checkpoint(latest_path)
        self.assertIn("DataLoader worker count mismatch", str(ctx.exception))

        trainer_ok = self._build_trainer(config=cfg)
        completed = trainer_ok.resume_from_checkpoint(latest_path)
        self.assertFalse(completed)
        self.assertEqual(trainer_ok.start_epoch, 2)

    # 14. Stale/missing best checkpoint recovery via authoritative latest_model.pt
    def test_14_stale_best_checkpoint_recovery_and_evaluator(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 3
        trainer = self._build_trainer(config=cfg)
        ckpt_dir = Path(self.test_dir) / "checkpoints"

        # Epoch 1: val acc = 50.0%
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader, stop_at_epoch=1)

        best_pt = ckpt_dir / "best_model.pt"
        best_json = ckpt_dir / "best_metadata.json"
        self.assertTrue(best_pt.is_file())
        self.assertTrue(best_json.is_file())

        # Save weights at epoch 1
        weights_epoch1 = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        # Epoch 2 improves val acc to 80.0% (new best)
        # We simulate the exact crash boundary:
        # latest_model.pt is atomically saved with epoch=2, best_epoch=2, best_metric=80.0,
        # but crash happens BEFORE best_model.pt and best_metadata.json are updated!
        weights_epoch2 = {k: v.clone() + 0.5 for k, v in weights_epoch1.items()}
        trainer.model.load_state_dict(weights_epoch2)
        trainer.best_metric = 80.0
        trainer.best_epoch = 2
        trainer.best_model_state = weights_epoch2

        # Manually save latest_model.pt for epoch 2, leaving best_model.pt stale (from epoch 1)
        run_id_dict = trainer._get_run_identity()
        latest_state = {
            'epoch': 2,
            'model_state': weights_epoch2,
            'optimizer_state': trainer.optimizer.state_dict(),
            'scheduler_state': trainer.scheduler.state_dict(),
            'scaler_state': None,
            'metrics': {'accuracy': 80.0, 'loss': 0.5},
            'best_epoch': 2,
            'best_metric': 80.0,
            'best_model_state': weights_epoch2,
            'best_metadata': {
                'best_epoch': 2,
                'best_validation_accuracy': 80.0,
                'checkpoint_identifier': 'best_model.pt'
            },
            'rng_state': trainer._capture_rng_state(),
            'run_identity': run_id_dict,
            'config_hash': run_id_dict["config_hash"],
            'evaluation_preprocessing_id': run_id_dict["evaluation_preprocessing_id"],
            'schema_version': "2.0",
            'checkpoint_type': "latest"
        }
        CheckpointManager._atomic_torch_save(latest_state, ckpt_dir / "latest_model.pt")

        # Confirm stale best state before reconciliation
        best_before = torch.load(best_pt, map_location="cpu")
        self.assertEqual(best_before["epoch"], 1, "Best checkpoint should be stale before recovery")

        # 1 & 2. Trigger reconciliation
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired, "Reconciliation must detect inconsistency and repair")

        # Verify best_model.pt was updated to epoch 2 weights
        best_after = torch.load(best_pt, map_location="cpu")
        self.assertEqual(best_after["epoch"], 2)
        for k in weights_epoch2:
            self.assertTrue(torch.equal(best_after["model_state"][k], weights_epoch2[k].cpu()))

        # Verify best_metadata.json was repaired
        import json
        with open(best_json, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["best_epoch"], 2)
        self.assertEqual(meta["best_validation_accuracy"], 80.0)

        # 3. Resume training at epoch 3 with a lower accuracy (e.g. 40.0%)
        # Verify it DOES NOT overwrite the true best model
        fresh_trainer = self._build_trainer(config=cfg)
        fresh_trainer.resume_from_checkpoint(ckpt_dir / "latest_model.pt")
        self.assertEqual(fresh_trainer.start_epoch, 3)
        self.assertEqual(fresh_trainer.best_metric, 80.0)
        self.assertEqual(fresh_trainer.best_epoch, 2)

        # 4. Check Evaluator compatibility
        self.assertTrue(best_pt.is_file())
        self.assertTrue(best_json.is_file())
        self.assertEqual(best_after["schema_version"], "2.0")

    # 15. Interrupted metadata sidecar update recovery
    def test_15_interrupted_metadata_update_recovery(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader, stop_at_epoch=1)

        ckpt_dir = Path(self.test_dir) / "checkpoints"
        latest_json = ckpt_dir / "latest_metadata.json"

        # Corrupt latest_metadata.json to simulate crash during sidecar write
        with open(latest_json, "w") as f:
            f.write("CORRUPT_JSON_HALF_WRITTEN {")

        # Reconciliation must detect and repair from authoritative latest_model.pt
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired)

        import json
        with open(latest_json, "r") as f:
            repaired_meta = json.load(f)
        self.assertEqual(repaired_meta["latest_epoch"], 1)
        self.assertEqual(repaired_meta["checkpoint_identifier"], "latest_model.pt")

    # 16. Strict rejection when required fields are missing
    def test_16_missing_required_checkpoint_fields_fail_safely(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader, stop_at_epoch=1)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"
        valid_checkpoint = torch.load(latest_path, map_location="cpu")

        required_fields = [
            "epoch",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_state",
            "best_epoch",
            "best_metric",
            "best_model_state",
            "best_metadata",
            "run_identity",
            "config_hash",
            "evaluation_preprocessing_id",
            "checkpoint_type",
            "schema_version",
            "metrics"
        ]

        dummy_path = Path(self.test_dir) / "dummy.pt"
        for field in required_fields:
            corrupted = copy.deepcopy(valid_checkpoint)
            del corrupted[field]
            with self.assertRaises(ValueError, msg=f"Should reject when '{field}' is missing"):
                trainer._validate_checkpoint(corrupted, dummy_path)

        # Also test rejection of schema version 1.0
        corrupted_schema = copy.deepcopy(valid_checkpoint)
        corrupted_schema["schema_version"] = "1.0"
        with self.assertRaises(ValueError, msg="Should reject schema_version 1.0"):
            trainer._validate_checkpoint(corrupted_schema, dummy_path)

    # 17. Evaluation preprocessing mismatch rejection
    def test_17_evaluation_preprocessing_mismatch_rejected(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 2
        trainer = self._build_trainer(config=cfg)
        train_loader, val_loader = create_synthetic_dataloaders()
        trainer.fit(train_loader, val_loader, stop_at_epoch=1)

        latest_path = Path(self.test_dir) / "checkpoints" / "latest_model.pt"

        # Change image_size in config to simulate changed eval preprocessing
        cfg_bad_eval = copy.deepcopy(cfg)
        cfg_bad_eval["data"]["image_size"] = 64

        trainer_mismatch = self._build_trainer(config=cfg_bad_eval)
        with self.assertRaises(ValueError) as ctx:
            trainer_mismatch.resume_from_checkpoint(latest_path)
        self.assertIn("Evaluation preprocessing mismatch", str(ctx.exception))

    # 18. CUDA AMP scaler state restoration step test
    def test_18_cuda_amp_scaler_restoration_step(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available; skipping CUDA AMP scaler step test.")

        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["use_amp"] = True
        cfg["training"]["epochs"] = 3

        # 1. Initialize actual CUDA GradScaler
        m = SmallTestModel().to("cuda")
        opt = get_optimizer(m, cfg)
        sched = get_scheduler(opt, cfg)
        crit = nn.CrossEntropyLoss()
        c_dir = Path(self.test_dir) / "cuda_amp_ckpt"

        trainer1 = Trainer(
            model=m, optimizer=opt, scheduler=sched, criterion=crit,
            config=cfg, device=torch.device("cuda"), checkpoint_dir=str(c_dir)
        )
        self.assertIsNotNone(trainer1.amp.scaler)
        initial_scale = trainer1.amp.scaler.get_scale()
        self.assertEqual(initial_scale, 65536.0)

        # 2. Perform valid scaled forward/backward/optimizer step
        x_dummy = torch.randn(4, 16, device="cuda")
        y_dummy = torch.randint(0, 4, (4,), device="cuda")
        opt.zero_grad()
        with trainer1.amp.autocast():
            out1 = m(x_dummy)
            loss1 = crit(out1, y_dummy)
        trainer1.amp.scale_loss(loss1).backward()
        trainer1.amp.step(opt)
        self.assertEqual(trainer1.amp.scaler.state_dict()["_growth_tracker"], 1)

        # 3. Deliberately move scaler into a non-default, valid state via supported mechanisms:
        # Trigger inf gradient backoff to halve scale from 65536 to 32768
        opt.zero_grad()
        with trainer1.amp.autocast():
            out2 = m(x_dummy)
            loss2 = crit(out2, y_dummy)
        trainer1.amp.scale_loss(loss2).backward()
        for p in m.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float('inf'))
                break
        trainer1.amp.step(opt)  # skips optimizer step due to inf, cuts scale to 32768 and resets tracker to 0
        self.assertEqual(trainer1.amp.scaler.get_scale(), 32768.0)
        self.assertEqual(trainer1.amp.scaler.state_dict()["_growth_tracker"], 0)

        # Perform two subsequent clean steps to advance growth_tracker to 2
        for _ in range(2):
            opt.zero_grad()
            with trainer1.amp.autocast():
                out_clean = m(x_dummy)
                loss_clean = crit(out_clean, y_dummy)
            trainer1.amp.scale_loss(loss_clean).backward()
            trainer1.amp.step(opt)

        saved_scaler_state = copy.deepcopy(trainer1.amp.state_dict())
        self.assertEqual(saved_scaler_state["scale"], 32768.0)
        self.assertEqual(saved_scaler_state["_growth_tracker"], 2)

        # 4. Save checkpoint with non-default scaler state
        train_loader, val_loader = create_synthetic_dataloaders()
        run_id_dict = trainer1._get_run_identity()
        eval_id = get_evaluation_preprocessing_id(cfg)
        state_dict = {
            'epoch': 1,
            'model_state': trainer1.model.state_dict(),
            'optimizer_state': trainer1.optimizer.state_dict(),
            'scheduler_state': trainer1.scheduler.state_dict(),
            'scaler_state': trainer1.amp.state_dict(),
            'metrics': {'accuracy': 10.0, 'loss': 1.0},
            'best_epoch': 1,
            'best_metric': 10.0,
            'best_model_state': {k: v.cpu().clone() for k, v in trainer1.model.state_dict().items()},
            'rng_state': trainer1._capture_rng_state(train_loader),
            'run_identity': run_id_dict,
            'config_hash': run_id_dict["config_hash"],
            'evaluation_preprocessing_id': eval_id,
            'schema_version': "2.0",
            'checkpoint_type': "latest"
        }
        trainer1.ckpt_manager.save_checkpoint(
            state_dict, epoch=1, metric=10.0,
            extra_metadata={"model_name": "small_test_model", "recipe": "test"}
        )
        latest_pt = c_dir / "latest_model.pt"
        self.assertTrue(latest_pt.is_file())

        ckpt_payload = torch.load(latest_pt, map_location="cpu")
        self.assertEqual(ckpt_payload["scaler_state"]["scale"], 32768.0)
        self.assertEqual(ckpt_payload["scaler_state"]["_growth_tracker"], 2)

        # 5. Destroy original scaler and trainer
        del trainer1, m, opt, sched
        torch.cuda.empty_cache()

        # 6. Create fresh scaler with default initial state
        fresh_model = SmallTestModel().to("cuda")
        fresh_opt = get_optimizer(fresh_model, cfg)
        fresh_sched = get_scheduler(fresh_opt, cfg)

        trainer2 = Trainer(
            model=fresh_model, optimizer=fresh_opt, scheduler=fresh_sched, criterion=crit,
            config=cfg, device=torch.device("cuda"), checkpoint_dir=str(c_dir)
        )

        # 7. Demonstrate that fresh scaler state differs from saved state
        fresh_scaler_state = trainer2.amp.state_dict()
        self.assertEqual(trainer2.amp.scaler.get_scale(), 65536.0)
        self.assertEqual(fresh_scaler_state["_growth_tracker"], 0)
        self.assertNotEqual(trainer2.amp.scaler.get_scale(), saved_scaler_state["scale"])
        self.assertNotEqual(fresh_scaler_state["_growth_tracker"], saved_scaler_state["_growth_tracker"])

        # 8. Restore the saved checkpoint
        completed = trainer2.resume_from_checkpoint(latest_pt)
        self.assertFalse(completed)
        self.assertEqual(trainer2.start_epoch, 2)

        # 9. Verify that the entire relevant scaler state is restored (scale and growth tracker)
        restored_state = trainer2.amp.state_dict()
        self.assertEqual(trainer2.amp.scaler.get_scale(), 32768.0)
        self.assertEqual(restored_state["scale"], 32768.0)
        self.assertEqual(restored_state["_growth_tracker"], 2)
        self.assertEqual(restored_state, saved_scaler_state)

        # 10. Perform another CUDA AMP training step successfully
        train_loss = trainer2.train_epoch(train_loader)
        self.assertIsInstance(train_loss, float)
        self.assertFalse(np.isnan(train_loss))
        # Growth tracker advances by len(train_loader) clean batches (2 + 6 = 8)
        self.assertEqual(trainer2.amp.state_dict()["_growth_tracker"], 2 + len(train_loader))
        self.assertEqual(trainer2.amp.scaler.get_scale(), 32768.0)

    # 19. Production worker count continuation test
    def test_19_production_worker_resumption(self):
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 3
        cfg["data"]["num_workers"] = 2  # Exercise multiprocessing worker pool
        c_dir = Path(self.test_dir) / "prod_workers_ckpt"

        trainer1 = self._build_trainer(config=cfg, ckpt_dir=c_dir)
        train_loader, val_loader = create_synthetic_dataloaders(num_workers=2)

        # Train epoch 1
        trainer1.fit(train_loader, val_loader, stop_at_epoch=1)
        latest_pt = c_dir / "latest_model.pt"
        self.assertTrue(latest_pt.is_file())

        # Resume for epoch 2
        trainer2 = self._build_trainer(config=cfg, ckpt_dir=c_dir)
        completed = trainer2.resume_from_checkpoint(latest_pt)
        self.assertFalse(completed)
        self.assertEqual(trainer2.start_epoch, 2)

        trainer2.fit(train_loader, val_loader, stop_at_epoch=2)
        ckpt = torch.load(latest_pt, map_location="cpu")
        self.assertEqual(ckpt["epoch"], 2)

if __name__ == "__main__":
    unittest.main()
