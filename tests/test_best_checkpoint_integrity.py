import unittest
import copy
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from src.data.transforms import (
    get_evaluation_preprocessing_descriptor,
    get_evaluation_preprocessing_descriptor_sha256,
    get_evaluation_preprocessing_id,
    validate_evaluation_preprocessing,
    FROZEN_EVALUATION_PREPROCESSING_ID,
    EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
)
from src.data.split_manager import FROZEN_SPLIT_HASH
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer, compute_config_hash
from src.training.checkpointing import CheckpointManager
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler


class TinyDummyModel(nn.Module):
    def __init__(self, in_features: int = 16, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.fc(x)


def create_dummy_loaders(n_samples: int = 32, in_features: int = 16, num_classes: int = 100, batch_size: int = 8):
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n_samples, in_features, generator=g)
    y = torch.randint(0, num_classes, (n_samples,), generator=g)
    train_ds = TensorDataset(X[:24], y[:24])
    val_ds = TensorDataset(X[24:], y[24:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class TestBestCheckpointIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.base_config = {
            "model": {"name": "resnet18", "architecture": "resnet18", "num_classes": 100},
            "training": {
                "epochs": 2,
                "learning_rate": 0.05,
                "optimizer": "sgd",
                "use_amp": False,
            },
            "optimizer": {"name": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 1e-4},
            "scheduler": {"name": "cosine", "min_lr": 1e-4},
            "data": {
                "dataset": "CIFAR-100",
                "data_dir": str(Path(self.test_dir) / "data"),
                "image_size": 32,
                "num_classes": 100,
                "num_workers": 0,
                "persistent_workers": False,
                "split_hash": FROZEN_SPLIT_HASH
            },
            "augmentation": {"type": "standard"},
            "experiment": {
                "seed": 42,
                "model_name": "resnet18",
                "recipe": "recipe_a",
                "output_dir": self.test_dir,
                "split_hash": FROZEN_SPLIT_HASH
            },
            "evaluation": {
                "results_root": str(Path(self.test_dir) / "results")
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_trainer(self, config=None, model=None, ckpt_dir=None):
        cfg = copy.deepcopy(config or self.base_config)
        m = model or TinyDummyModel()
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
            device="cpu",
            checkpoint_dir=str(c_dir)
        )

    def _create_trained_run(self, ckpt_dir: Path, cfg: dict):
        trainer = self._build_trainer(config=cfg, ckpt_dir=ckpt_dir)
        train_loader, val_loader = create_dummy_loaders()
        trainer.fit(train_loader, val_loader)
        return trainer

    # =========================================================================
    # ISSUE 1 TESTS: Best checkpoint must match authoritative embedded snapshot
    # =========================================================================

    def test_01_reconcile_repairs_best_pt_with_different_weights_same_epoch(self):
        """Assert reconciliation repairs best_model.pt when weights differ from authoritative embedded snapshot."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test01"
        self._create_trained_run(ckpt_dir, cfg)

        latest_pt = ckpt_dir / "latest_model.pt"
        best_pt = ckpt_dir / "best_model.pt"

        latest_ckpt = torch.load(latest_pt, map_location="cpu")
        best_ckpt = torch.load(best_pt, map_location="cpu")

        # Mutate best_model.pt weights with random tensor values (same epoch and metric)
        mutated_weights = {}
        for k, v in best_ckpt["model_state"].items():
            mutated_weights[k] = torch.randn_like(v)
        best_ckpt["model_state"] = mutated_weights
        torch.save(best_ckpt, best_pt)

        # Reconcile
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired, "Reconciliation must detect weight discrepancy and repair best_model.pt")

        # Assert repaired weights strictly match authoritative embedded snapshot
        repaired_best = torch.load(best_pt, map_location="cpu")
        for k, auth_tensor in latest_ckpt["best_model_state"].items():
            self.assertTrue(
                torch.equal(repaired_best["model_state"][k], auth_tensor.cpu()),
                f"Weight tensor '{k}' was not restored from embedded best snapshot"
            )

    def test_02_reconcile_repairs_best_pt_with_different_metric_same_epoch(self):
        """Assert reconciliation repairs best_model.pt when validation metric differs from authoritative metric."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test02"
        self._create_trained_run(ckpt_dir, cfg)

        latest_pt = ckpt_dir / "latest_model.pt"
        best_pt = ckpt_dir / "best_model.pt"

        latest_ckpt = torch.load(latest_pt, map_location="cpu")
        best_ckpt = torch.load(best_pt, map_location="cpu")
        auth_metric = float(latest_ckpt["best_metric"])

        # Mutate best_model.pt metric (different metric, same epoch and weights)
        best_ckpt["best_metric"] = auth_metric + 15.0
        torch.save(best_ckpt, best_pt)

        # Reconcile
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired, "Reconciliation must detect metric discrepancy and repair best_model.pt")

        repaired_best = torch.load(best_pt, map_location="cpu")
        self.assertEqual(float(repaired_best["best_metric"]), auth_metric)

    def test_03_reconcile_repairs_best_pt_with_altered_split_or_run_identity(self):
        """Assert reconciliation repairs best_model.pt when split_hash or run_identity is altered."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test03"
        self._create_trained_run(ckpt_dir, cfg)

        best_pt = ckpt_dir / "best_model.pt"
        best_ckpt = torch.load(best_pt, map_location="cpu")

        # Alter split_hash inside run_identity
        if "run_identity" in best_ckpt and isinstance(best_ckpt["run_identity"], dict):
            best_ckpt["run_identity"]["split_hash"] = "altered_split_hash_9999"
        else:
            best_ckpt["run_identity"] = {"split_hash": "altered_split_hash_9999"}
        torch.save(best_ckpt, best_pt)

        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired, "Reconciliation must detect altered split_hash in run_identity and repair")

        repaired_best = torch.load(best_pt, map_location="cpu")
        self.assertEqual(repaired_best["run_identity"]["split_hash"], FROZEN_SPLIT_HASH)

    def test_04_reconcile_leaves_correct_best_checkpoint_untouched(self):
        """Assert reconciliation does not modify a completely consistent best_model.pt or sidecars."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test04"
        self._create_trained_run(ckpt_dir, cfg)

        best_pt = ckpt_dir / "best_model.pt"
        best_bytes_before = best_pt.read_bytes()

        # Run reconciliation on clean, consistent checkpoints
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertFalse(repaired, "Reconciliation must return False when all artifacts are consistent")

        best_bytes_after = best_pt.read_bytes()
        self.assertEqual(best_bytes_before, best_bytes_after, "Consistent best_model.pt must remain untouched")

    def test_05_reconciled_checkpoint_strictly_matches_authoritative_embedded_snapshot(self):
        """Assert repaired best checkpoint exactly matches all fields of authoritative embedded snapshot."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test05"
        self._create_trained_run(ckpt_dir, cfg)

        latest_pt = ckpt_dir / "latest_model.pt"
        best_pt = ckpt_dir / "best_model.pt"
        best_json = ckpt_dir / "best_metadata.json"

        latest_ckpt = torch.load(latest_pt, map_location="cpu")

        # Corrupt best_model.pt completely (wrong epoch, wrong metric, wrong weights)
        corrupted_best = {
            "epoch": 99,
            "best_epoch": 99,
            "best_metric": 0.0,
            "model_state": {k: torch.zeros_like(v) for k, v in latest_ckpt["best_model_state"].items()},
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "checkpoint_type": "best"
        }
        torch.save(corrupted_best, best_pt)

        # Reconcile
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired)

        repaired_best = torch.load(best_pt, map_location="cpu")
        self.assertEqual(repaired_best["epoch"], latest_ckpt["best_epoch"])
        self.assertEqual(repaired_best["best_epoch"], latest_ckpt["best_epoch"])
        self.assertEqual(repaired_best["best_metric"], latest_ckpt["best_metric"])
        self.assertEqual(repaired_best["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(repaired_best["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

        for k, v in latest_ckpt["best_model_state"].items():
            self.assertTrue(torch.equal(repaired_best["model_state"][k], v.cpu()))

        with open(best_json, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["best_epoch"], latest_ckpt["best_epoch"])
        self.assertEqual(sidecar["best_validation_accuracy"], latest_ckpt["best_metric"])
        self.assertEqual(sidecar["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(sidecar["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

    def test_06_reconcile_fails_closed_when_embedded_snapshot_is_missing_or_corrupt(self):
        """Assert reconciliation fails closed if authoritative latest_model.pt has missing or corrupt embedded snapshot."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "ckpt_test06"
        self._create_trained_run(ckpt_dir, cfg)

        latest_pt = ckpt_dir / "latest_model.pt"
        latest_ckpt = torch.load(latest_pt, map_location="cpu")

        # 1. Empty best_model_state
        latest_ckpt["best_model_state"] = {}
        torch.save(latest_ckpt, latest_pt)
        with self.assertRaises(ValueError) as ctx1:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("invalid or empty embedded 'best_model_state'", str(ctx1.exception))

        # 2. Inconsistent best_epoch > current epoch
        latest_ckpt["best_model_state"] = {"fc.weight": torch.randn(100, 16)}
        latest_ckpt["epoch"] = 1
        latest_ckpt["best_epoch"] = 5
        torch.save(latest_ckpt, latest_pt)
        with self.assertRaises(ValueError) as ctx2:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("best_epoch (5) > current epoch (1)", str(ctx2.exception))

    # =========================================================================
    # ISSUE 2 TESTS: Standalone Best Structural Reconciliation
    # =========================================================================

    def test_07_reconcile_standalone_best_rejects_epoch_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when epoch contradicts metadata."""
        run_id = "test_standalone_epoch_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 2,  # Contradicts best_model.pt best_epoch (1)
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("best_epoch", str(ctx.exception))

    def test_08_reconcile_standalone_best_rejects_metric_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when validation metric contradicts metadata."""
        run_id = "test_standalone_metric_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 75.0,  # Contradicts best_model.pt metric (50.0)
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("accuracy", str(ctx.exception))

    def test_09_reconcile_standalone_best_rejects_split_identity_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when split_hash does not match contract."""
        run_id = "test_standalone_split_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        bad_split_hash = "corrupted_split_hash_000000000000000000000000000000000000000000000000"
        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": bad_split_hash,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": bad_split_hash
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("split hash", str(ctx.exception))

    def test_10_reconcile_standalone_best_rejects_run_identity_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when run_identity contradicts config."""
        run_id = "test_standalone_run_id_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "run_identity": {
                "architecture": "vit_large_patch16_224",  # Mismatch (expected resnet18)
                "recipe": "recipe_a",
                "seed": 42,
                "split_hash": FROZEN_SPLIT_HASH
            },
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "model_name": "vit_large_patch16_224",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("architecture", str(ctx.exception))

    def test_11_reconcile_standalone_best_rejects_preprocessing_protocol_id_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when preprocessing protocol ID is altered."""
        run_id = "test_standalone_protocol_id_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": "tampered_protocol_id_0000",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": "tampered_protocol_id_0000",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("protocol id", str(ctx.exception).lower())

    def test_12_reconcile_standalone_best_rejects_descriptor_fingerprint_mismatch(self):
        """Assert CheckpointManager rejects standalone-best when descriptor fingerprint is altered."""
        run_id = "test_standalone_desc_sha_mismatch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": "tampered_descriptor_fingerprint_0000",
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": "tampered_descriptor_fingerprint_0000",
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            CheckpointManager.reconcile_checkpoints(run_ckpt_dir, self.base_config)
        self.assertIn("descriptor sha-256", str(ctx.exception).lower())

    def test_13_reconcile_standalone_best_leaves_consistent_checkpoint_untouched_and_repairs_missing_sidecar(self):
        """Assert CheckpointManager preserves valid standalone best and repairs missing sidecar for inspection."""
        run_id = "test_standalone_valid"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir

        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        # 1. Missing sidecar should be repaired for inspection
        self.assertFalse(best_json.is_file())
        repaired = CheckpointManager.reconcile_checkpoints(run_ckpt_dir, cfg)
        self.assertTrue(repaired)
        self.assertTrue(best_json.is_file())

        with open(best_json, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["best_epoch"], 1)
        self.assertEqual(sidecar["best_validation_accuracy"], 50.0)

        # 2. Subsequent call on consistent checkpoint should return False (untouched)
        repaired_again = CheckpointManager.reconcile_checkpoints(run_ckpt_dir, cfg)
        self.assertFalse(repaired_again)

    # =========================================================================
    # OPTION B TESTS: Evaluator Fails Closed on Standalone Best Checkpoints
    # =========================================================================

    def test_14_evaluator_rejects_valid_looking_standalone_best_without_latest_model_pt(self):
        """Assert evaluator strictly rejects valid-looking standalone best when latest_model.pt is absent."""
        run_id = "test_standalone_missing_latest"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("latest_model.pt", str(ctx.exception))
        # Verify failure occurred before any predictions or results tables were created
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

    def test_15_evaluator_rejects_standalone_best_with_altered_matching_validation_metric(self):
        """Assert evaluator rejects standalone-best when both files have same altered valid metric."""
        run_id = "test_standalone_altered_metric"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        altered_metric = 99.5
        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": altered_metric,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": altered_metric,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("latest_model.pt", str(ctx.exception))
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

    def test_16_evaluator_rejects_standalone_best_with_altered_matching_epoch(self):
        """Assert evaluator rejects standalone-best when both files have same altered epoch."""
        run_id = "test_standalone_altered_epoch"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        altered_epoch = 50
        torch.save({
            "epoch": altered_epoch,
            "best_epoch": altered_epoch,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": altered_epoch,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("latest_model.pt", str(ctx.exception))
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

    def test_17_evaluator_rejects_standalone_best_with_valid_run_identity_and_preprocessing(self):
        """Assert evaluator rejects standalone-best even when run identity and preprocessing are strictly valid."""
        run_id = "test_standalone_valid_identity"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"

        expected_config_hash = compute_config_hash(cfg, FROZEN_SPLIT_HASH)
        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "config_hash": expected_config_hash,
            "run_identity": {
                "architecture": "resnet18",
                "recipe": "recipe_a",
                "seed": 42,
                "split_hash": FROZEN_SPLIT_HASH,
                "config_hash": expected_config_hash,
            },
            "schema_version": "2.0"
        }, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt",
            "model_name": "resnet18",
            "recipe": "recipe_a",
            "seed": 42,
            "split_hash": FROZEN_SPLIT_HASH,
            "config_hash": expected_config_hash,
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256
        }), encoding="utf-8")

        with self.assertRaises(FileNotFoundError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("latest_model.pt", str(ctx.exception))
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

    @patch("src.evaluation.evaluator.write_tables")
    @patch("src.evaluation.evaluator.load_records", return_value=(None, {"logits": np.zeros((10, 100)), "labels": np.zeros(10, dtype=int)}))
    @patch("src.evaluation.evaluator.PredictionStore")
    @patch("src.evaluation.evaluator.ResultAccumulator")
    @patch("src.evaluation.evaluator.fit_temperature_scaling", return_value=1.0)
    @patch("src.evaluation.evaluator.get_cifar100c_dataloader", return_value=[])
    @patch("src.evaluation.evaluator.get_cifar100_dataloaders", return_value=(None, MagicMock(), []))
    def test_18_evaluator_passes_complete_checkpoint_directory_with_valid_latest_model_pt(
        self, mock_cifar100, mock_cifar100c, mock_temp, mock_acc_cls, mock_pred_store, mock_load_rec, mock_write_tables
    ):
        """Assert evaluator succeeds on complete directory containing authoritative latest_model.pt."""
        run_id = "test_complete_directory_passes"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        mock_acc = MagicMock()
        mock_acc.finalize.return_value = (pd.DataFrame([{"run_id": run_id}]), pd.DataFrame(), pd.DataFrame())
        mock_acc_cls.return_value = mock_acc

        mock_store = MagicMock()
        mock_store.__enter__.return_value = mock_store
        mock_store.finalize.return_value = Path(self.test_dir) / "dummy_dir"
        mock_pred_store.create.return_value = mock_store

        latest_pt = run_ckpt_dir / "latest_model.pt"
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"
        latest_json = run_ckpt_dir / "latest_metadata.json"

        weights = model.state_dict()
        latest_ckpt = {
            "epoch": 2,
            "best_epoch": 2,
            "best_metric": 72.5,
            "model_state": weights,
            "best_model_state": weights,
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }
        torch.save(latest_ckpt, latest_pt)

        best_ckpt = {
            "epoch": 2,
            "best_epoch": 2,
            "best_metric": 72.5,
            "model_state": weights,
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }
        torch.save(best_ckpt, best_pt)

        best_json.write_text(json.dumps({
            "best_epoch": 2,
            "best_validation_accuracy": 72.5,
            "checkpoint_identifier": "best_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH
        }), encoding="utf-8")

        latest_json.write_text(json.dumps({
            "latest_epoch": 2,
            "best_epoch": 2,
            "checkpoint_identifier": "latest_model.pt",
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256
        }), encoding="utf-8")

        evaluator.run_contract_evaluation(run_id)
        mock_write_tables.assert_called_once()
        passed_metadata = mock_write_tables.call_args[0][1]
        self.assertEqual(passed_metadata.run_id, run_id)
        self.assertEqual(passed_metadata.selected_epoch, 2)
        self.assertAlmostEqual(passed_metadata.best_validation_accuracy_fraction, 0.725)

    def test_19_evaluator_fails_closed_before_predictions_on_missing_or_corrupt_latest_model_pt(self):
        """Assert evaluator fails before predictions/writes on missing or corrupt latest_model.pt."""
        run_id = "test_fails_before_writes"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        # Case A: Missing latest_model.pt
        with self.assertRaises(FileNotFoundError):
            evaluator.run_contract_evaluation(run_id)
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

        # Case B: Corrupted bytes in latest_model.pt
        latest_pt = run_ckpt_dir / "latest_model.pt"
        latest_pt.write_bytes(b"CORRUPTED_CHECKPOINT_BYTES_NOT_TORCH")
        with self.assertRaises(RuntimeError):
            evaluator.run_contract_evaluation(run_id)
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

        # Case C: Invalid non-dict payload
        torch.save(["not", "a", "dict"], latest_pt)
        with self.assertRaises(ValueError):
            evaluator.run_contract_evaluation(run_id)
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "predictions" / run_id).exists())
        self.assertFalse((Path(cfg["evaluation"]["results_root"]) / "runs" / f"{run_id}.csv").exists())

    @patch("src.evaluation.evaluator.write_tables")
    @patch("src.evaluation.evaluator.load_records", return_value=(None, {"logits": np.zeros((10, 100)), "labels": np.zeros(10, dtype=int)}))
    @patch("src.evaluation.evaluator.PredictionStore")
    @patch("src.evaluation.evaluator.ResultAccumulator")
    @patch("src.evaluation.evaluator.fit_temperature_scaling", return_value=1.0)
    @patch("src.evaluation.evaluator.get_cifar100c_dataloader", return_value=[])
    @patch("src.evaluation.evaluator.get_cifar100_dataloaders", return_value=(None, MagicMock(), []))
    def test_20_evaluator_recovers_missing_best_model_pt_from_valid_latest_model_pt(
        self, mock_cifar100, mock_cifar100c, mock_temp, mock_acc_cls, mock_pred_store, mock_load_rec, mock_write_tables
    ):
        """Assert evaluator reconstructs missing best_model.pt from latest_model.pt and evaluates."""
        run_id = "test_recovers_missing_best"
        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = TinyDummyModel()
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        mock_acc = MagicMock()
        mock_acc.finalize.return_value = (pd.DataFrame([{"run_id": run_id}]), pd.DataFrame(), pd.DataFrame())
        mock_acc_cls.return_value = mock_acc

        mock_store = MagicMock()
        mock_store.__enter__.return_value = mock_store
        mock_store.finalize.return_value = Path(self.test_dir) / "dummy_dir"
        mock_pred_store.create.return_value = mock_store

        latest_pt = run_ckpt_dir / "latest_model.pt"
        best_pt = run_ckpt_dir / "best_model.pt"

        weights = model.state_dict()
        latest_ckpt = {
            "epoch": 2,
            "best_epoch": 1,
            "best_metric": 88.0,
            "model_state": weights,
            "best_model_state": weights,
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "split_hash": FROZEN_SPLIT_HASH,
            "schema_version": "2.0"
        }
        torch.save(latest_ckpt, latest_pt)

        # best_model.pt is deliberately missing
        self.assertFalse(best_pt.is_file())

        # run_contract_evaluation must reconcile and reconstruct best_model.pt, then succeed
        evaluator.run_contract_evaluation(run_id)
        self.assertTrue(best_pt.is_file(), "Reconciliation failed to reconstruct missing best_model.pt")
        mock_write_tables.assert_called_once()
        passed_metadata = mock_write_tables.call_args[0][1]
        self.assertEqual(passed_metadata.selected_epoch, 1)
        self.assertAlmostEqual(passed_metadata.best_validation_accuracy_fraction, 0.880)


if __name__ == "__main__":
    unittest.main()
