import unittest
import copy
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.data.transforms import (
    get_evaluation_preprocessing_descriptor,
    get_evaluation_preprocessing_descriptor_sha256,
    get_evaluation_preprocessing_id,
    validate_evaluation_preprocessing,
    get_transforms,
    FROZEN_EVALUATION_PREPROCESSING_ID,
    EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
)
from src.evaluation.contract import (
    RunMetadata,
    ResultAccumulator,
    write_tables,
    CORRUPTIONS,
    SEVERITIES,
)
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
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


class TestEvaluationPreprocessingIdentity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.base_config = {
            "model": {"name": "resnet18", "num_classes": 100},
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
                "split_hash": "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91"
            },
            "augmentation": {"type": "standard"},
            "experiment": {
                "seed": 42,
                "recipe": "test_recipe",
                "output_dir": self.test_dir,
                "split_hash": "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91"
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

    def test_01_frozen_protocol_id_exact_value(self):
        """Assert get_evaluation_preprocessing_id() returns the approved protocol identifier."""
        expected_id = "5256477cf00b8bcaec855fb96067bcbe6e718235b083b92261dd407b501f989e"
        self.assertEqual(FROZEN_EVALUATION_PREPROCESSING_ID, expected_id)
        self.assertEqual(get_evaluation_preprocessing_id(), expected_id)
        self.assertEqual(get_evaluation_preprocessing_id({"data": {}}), expected_id)

    def test_02_canonical_descriptor_sha256_exact_value(self):
        """Assert get_evaluation_preprocessing_descriptor_sha256() returns the approved fingerprint."""
        expected_sha256 = "dc9cb66d30e90972331e408340d0f00d8006e829665c79ddc254c7bb853f0862"
        self.assertEqual(EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256, expected_sha256)
        self.assertEqual(get_evaluation_preprocessing_descriptor_sha256(), expected_sha256)
        self.assertEqual(get_evaluation_preprocessing_descriptor_sha256({}), expected_sha256)
        self.assertEqual(get_evaluation_preprocessing_descriptor_sha256({"data": {}}), expected_sha256)

    def test_03_default_frozen_evaluation_transform_descriptor_content(self):
        """Assert the canonical descriptor matches the frozen evaluation pipeline specification."""
        desc = get_evaluation_preprocessing_descriptor()
        self.assertEqual(desc["dataset"], "CIFAR-100")
        self.assertEqual(desc["image_size"], 32)
        self.assertEqual(desc["normalization"]["mean"], [0.5071, 0.4867, 0.4408])
        self.assertEqual(desc["normalization"]["std"], [0.2675, 0.2565, 0.2761])
        self.assertEqual(len(desc["transforms"]), 2)
        self.assertEqual(desc["transforms"][0], {"type": "ToTensor"})
        self.assertEqual(
            desc["transforms"][1],
            {"type": "Normalize", "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}
        )
        # Validation passes cleanly on default config
        validate_evaluation_preprocessing(
            config={},
            evaluation_preprocessing_id=FROZEN_EVALUATION_PREPROCESSING_ID,
            evaluation_preprocessing_descriptor_sha256=EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256
        )

    def test_04_altered_normalization_rejected(self):
        """Assert altered mean or std values produce a different hash and are rejected."""
        # Altered mean
        bad_mean_cfg = {"data": {"normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.2675, 0.2565, 0.2761]}}}
        bad_mean_sha = get_evaluation_preprocessing_descriptor_sha256(bad_mean_cfg)
        self.assertNotEqual(bad_mean_sha, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_preprocessing(config=bad_mean_cfg)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

        # Altered std
        bad_std_cfg = {"data": {"normalization": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2, 0.2, 0.2]}}}
        bad_std_sha = get_evaluation_preprocessing_descriptor_sha256(bad_std_cfg)
        self.assertNotEqual(bad_std_sha, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_preprocessing(config=bad_std_cfg)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

    def test_05_altered_transform_ordering_rejected(self):
        """Assert altered transform ordering (e.g. Normalize before ToTensor) is rejected."""
        inverted_order_cfg = {
            "data": {
                "eval_transforms_order": ["Normalize", "ToTensor"]
            }
        }
        inverted_sha = get_evaluation_preprocessing_descriptor_sha256(inverted_order_cfg)
        self.assertNotEqual(inverted_sha, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_preprocessing(config=inverted_order_cfg)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

    def test_06_altered_image_dimensions_rejected(self):
        """Assert altered image dimensions (e.g. 64 instead of 32) are rejected."""
        bad_dim_cfg = {"data": {"image_size": 64}}
        bad_dim_sha = get_evaluation_preprocessing_descriptor_sha256(bad_dim_cfg)
        self.assertNotEqual(bad_dim_sha, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_preprocessing(config=bad_dim_cfg)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

    def test_07_altered_operations_rejected(self):
        """Assert addition or omission of evaluation operations is rejected."""
        missing_norm_cfg = {"data": {"eval_transforms_order": ["ToTensor"]}}
        missing_norm_sha = get_evaluation_preprocessing_descriptor_sha256(missing_norm_cfg)
        self.assertNotEqual(missing_norm_sha, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_preprocessing(config=missing_norm_cfg)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

    def test_08_resumption_rejects_mismatched_protocol_id(self):
        """Assert trainer resume rejects checkpoints with a mismatched evaluation_preprocessing_id."""
        trainer = self._build_trainer()
        ckpt_dir = Path(self.test_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / "latest_model.pt"

        run_id_dict = trainer._get_run_identity()
        state = {
            "epoch": 1,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "scaler_state": None,
            "metrics": {"accuracy": 50.0, "loss": 1.0},
            "best_epoch": 1,
            "best_metric": 50.0,
            "best_model_state": trainer.model.state_dict(),
            "best_metadata": {"best_epoch": 1, "best_validation_accuracy": 50.0, "checkpoint_identifier": "epoch_1"},
            "rng_state": trainer._capture_rng_state(),
            "run_identity": run_id_dict,
            "config_hash": run_id_dict["config_hash"],
            "evaluation_preprocessing_id": "corrupted_eval_protocol_id_12345",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }
        torch.save(state, latest_path)

        with self.assertRaises(ValueError) as ctx:
            trainer.resume_from_checkpoint(latest_path)
        self.assertIn("Evaluation preprocessing protocol ID mismatch", str(ctx.exception))

    def test_09_resumption_rejects_mismatched_descriptor_sha256(self):
        """Assert trainer resume rejects checkpoints with a mismatched evaluation_preprocessing_descriptor_sha256."""
        trainer = self._build_trainer()
        ckpt_dir = Path(self.test_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / "latest_model.pt"

        run_id_dict = trainer._get_run_identity()
        state = {
            "epoch": 1,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "scaler_state": None,
            "metrics": {"accuracy": 50.0, "loss": 1.0},
            "best_epoch": 1,
            "best_metric": 50.0,
            "best_model_state": trainer.model.state_dict(),
            "best_metadata": {"best_epoch": 1, "best_validation_accuracy": 50.0, "checkpoint_identifier": "epoch_1"},
            "rng_state": trainer._capture_rng_state(),
            "run_identity": run_id_dict,
            "config_hash": run_id_dict["config_hash"],
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": "bad_descriptor_sha256_abcdef",
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }
        torch.save(state, latest_path)

        with self.assertRaises(ValueError) as ctx:
            trainer.resume_from_checkpoint(latest_path)
        self.assertIn("Evaluation preprocessing descriptor SHA-256 mismatch", str(ctx.exception))

    def test_10_evaluator_rejects_mismatched_protocol_id_or_descriptor_sha256(self):
        """Assert Evaluator.run_contract_evaluation rejects checkpoints with altered eval ids."""
        run_id = "test_eval_mismatch"
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment"]["output_dir"] = self.test_dir
        model = TinyDummyModel()
        evaluator = Evaluator(model, cfg, device=torch.device("cpu"))

        run_ckpt_dir = Path(self.test_dir) / run_id / "checkpoints"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_pt = run_ckpt_dir / "best_model.pt"
        best_json = run_ckpt_dir / "best_metadata.json"
        latest_pt = run_ckpt_dir / "latest_model.pt"

        best_json.write_text(json.dumps({
            "best_epoch": 1,
            "best_validation_accuracy": 50.0,
            "checkpoint_identifier": "best_model.pt"
        }), encoding="utf-8")

        # 1. State with corrupted protocol id
        bad_id_ckpt = {
            "model_state": model.state_dict(),
            "best_epoch": 1,
            "best_metric": 50.0,
            "evaluation_preprocessing_id": "tampered_protocol_id",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "schema_version": "2.0",
            "checkpoint_type": "best"
        }
        torch.save(bad_id_ckpt, best_pt)
        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "best_model_state": model.state_dict(),
            "evaluation_preprocessing_id": "tampered_protocol_id",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }, latest_pt)
        with self.assertRaises(ValueError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("Evaluation preprocessing protocol ID mismatch", str(ctx.exception))

        # 2. State with corrupted descriptor sha256
        bad_sha_ckpt = {
            "model_state": model.state_dict(),
            "best_epoch": 1,
            "best_metric": 50.0,
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": "tampered_desc_sha",
            "schema_version": "2.0",
            "checkpoint_type": "best"
        }
        torch.save(bad_sha_ckpt, best_pt)
        torch.save({
            "epoch": 1,
            "best_epoch": 1,
            "best_metric": 50.0,
            "model_state": model.state_dict(),
            "best_model_state": model.state_dict(),
            "evaluation_preprocessing_id": FROZEN_EVALUATION_PREPROCESSING_ID,
            "evaluation_preprocessing_descriptor_sha256": "tampered_desc_sha",
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }, latest_pt)
        with self.assertRaises(ValueError) as ctx:
            evaluator.run_contract_evaluation(run_id)
        self.assertIn("Evaluation preprocessing descriptor SHA-256 mismatch", str(ctx.exception))

    def test_11_metadata_export_records_both_identifiers(self):
        """Assert RunMetadata and result tables export both evaluation preprocessing identifiers."""
        metadata = RunMetadata(
            run_id="test_run_metadata_export",
            architecture="resnet18",
            recipe="standard",
            training_seed=42,
            checkpoint_id="epoch_50",
            selected_epoch=50,
            best_validation_accuracy_fraction=0.75,
            split_hash="37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91",
            evaluation_preprocessing_config_id=FROZEN_EVALUATION_PREPROCESSING_ID,
            evaluation_preprocessing_descriptor_sha256=EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256
        )

        self.assertEqual(metadata.evaluation_preprocessing_config_id, FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(metadata.evaluation_preprocessing_descriptor_sha256, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

        accumulator = ResultAccumulator(metadata, temperature=1.0)
        # Mock clean logits & labels
        clean_logits = np.zeros((10, 100), dtype=np.float32)
        clean_labels = np.zeros(10, dtype=np.int64)
        accumulator.add_clean(clean_logits, clean_labels)

        # Mock all 75 corruption cells
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                accumulator.add_cell(c, s, np.zeros((5, 100), dtype=np.float32), np.zeros(5, dtype=np.int64))

        tables = accumulator.finalize()
        run_df, cells_df, bins_df = tables

        # Verify columns and exported values
        self.assertIn("evaluation_preprocessing_config_id", run_df.columns)
        self.assertIn("evaluation_preprocessing_descriptor_sha256", run_df.columns)
        self.assertEqual(run_df["evaluation_preprocessing_config_id"].iloc[0], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(run_df["evaluation_preprocessing_descriptor_sha256"].iloc[0], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

        # Verify write_tables writes these columns to CSV
        results_dir = Path(self.test_dir) / "results"
        write_tables(results_dir, metadata, tables)
        written_csv = results_dir / "runs" / f"{metadata.run_id}.csv"
        self.assertTrue(written_csv.is_file())
        saved_df = pd.read_csv(written_csv)
        self.assertEqual(saved_df["evaluation_preprocessing_config_id"].iloc[0], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(saved_df["evaluation_preprocessing_descriptor_sha256"].iloc[0], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

    def test_12_fit_validates_preprocessing_before_first_epoch(self):
        """Assert Trainer.fit validates preprocessing before epoch 1, producing no updates or checkpoints."""
        # 1. Setup with altered evaluation normalization
        bad_norm_cfg = copy.deepcopy(self.base_config)
        bad_norm_cfg["data"]["normalization"] = {"mean": [0.99, 0.99, 0.99], "std": [0.2675, 0.2565, 0.2761]}
        bad_norm_cfg["training"]["epochs"] = 2
        ckpt_dir = Path(self.test_dir) / "checkpoints_test12"

        model = TinyDummyModel()
        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
        trainer = self._build_trainer(config=bad_norm_cfg, model=model, ckpt_dir=ckpt_dir)
        train_loader, val_loader = create_dummy_loaders()

        # 2. Invoke Trainer.fit and verify immediate failure
        with self.assertRaises(ValueError) as ctx:
            trainer.fit(train_loader, val_loader)
        self.assertIn("active descriptor SHA-256", str(ctx.exception))

        # 3. Confirm zero progress, zero optimizer updates, and zero checkpoints
        self.assertEqual(trainer.start_epoch, 1)
        self.assertEqual(trainer.best_epoch, 0)
        self.assertEqual(trainer.best_metric, -float("inf"))
        self.assertIsNone(trainer.best_model_state)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, initial_weights[k]), f"Weight {k} was altered before epoch 1!")
        self.assertFalse((ckpt_dir / "latest_model.pt").exists())
        self.assertFalse((ckpt_dir / "best_model.pt").exists())

        # 4. Also test with explicit invalid evaluation_preprocessing_id in config
        bad_id_cfg = copy.deepcopy(self.base_config)
        bad_id_cfg["evaluation"] = {"evaluation_preprocessing_id": "invalid_id_9999"}
        bad_id_cfg["training"]["epochs"] = 2
        ckpt_dir_id = Path(self.test_dir) / "checkpoints_test12_id"
        trainer_id = self._build_trainer(config=bad_id_cfg, model=TinyDummyModel(), ckpt_dir=ckpt_dir_id)
        with self.assertRaises(ValueError) as ctx2:
            trainer_id.fit(train_loader, val_loader)
        self.assertIn("Evaluation preprocessing protocol ID mismatch", str(ctx2.exception))
        self.assertFalse((ckpt_dir_id / "latest_model.pt").exists())

    def test_13_reconciliation_repairs_latest_metadata_sidecar_provenance(self):
        """Assert reconciliation repairs altered or missing identifiers in latest_metadata.json while preserving unrelated fields."""
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 1
        ckpt_dir = Path(self.test_dir) / "checkpoints_test13"
        trainer = self._build_trainer(config=cfg, ckpt_dir=ckpt_dir)
        train_loader, val_loader = create_dummy_loaders()
        trainer.fit(train_loader, val_loader)

        latest_json = ckpt_dir / "latest_metadata.json"
        self.assertTrue(latest_json.is_file())

        # Add unrelated custom metadata fields to the sidecar
        with open(latest_json, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar["custom_experiment_note"] = "preserved_custom_value"
        sidecar["user_audit_tag"] = 12345
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        # 1. Modify protocol ID in parseable sidecar
        sidecar["evaluation_preprocessing_id"] = "corrupted_sidecar_protocol_id"
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired, "Reconciliation must detect altered protocol ID and repair sidecar")

        with open(latest_json, "r", encoding="utf-8") as f:
            repaired_sidecar = json.load(f)
        self.assertEqual(repaired_sidecar["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(repaired_sidecar["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        self.assertEqual(repaired_sidecar.get("custom_experiment_note"), "preserved_custom_value")
        self.assertEqual(repaired_sidecar.get("user_audit_tag"), 12345)

        # 2. Modify descriptor fingerprint in parseable sidecar
        repaired_sidecar["evaluation_preprocessing_descriptor_sha256"] = "corrupted_descriptor_fingerprint"
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(repaired_sidecar, f)

        repaired2 = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired2, "Reconciliation must detect altered descriptor sha256 and repair sidecar")

        with open(latest_json, "r", encoding="utf-8") as f:
            repaired_sidecar2 = json.load(f)
        self.assertEqual(repaired_sidecar2["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        self.assertEqual(repaired_sidecar2["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(repaired_sidecar2.get("custom_experiment_note"), "preserved_custom_value")
        self.assertEqual(repaired_sidecar2.get("user_audit_tag"), 12345)

    def test_14_reconciliation_repairs_best_metadata_sidecar_provenance(self):
        """Assert reconciliation repairs altered or missing identifiers in best_metadata.json while preserving unrelated fields."""
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 1
        ckpt_dir = Path(self.test_dir) / "checkpoints_test14"
        trainer = self._build_trainer(config=cfg, ckpt_dir=ckpt_dir)
        train_loader, val_loader = create_dummy_loaders()
        trainer.fit(train_loader, val_loader)

        best_json = ckpt_dir / "best_metadata.json"
        self.assertTrue(best_json.is_file())

        with open(best_json, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar["custom_best_label"] = "golden_candidate_v1"
        sidecar["best_extra_flag"] = True
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        # 1. Modify protocol ID in best_metadata.json
        sidecar["evaluation_preprocessing_id"] = "tampered_best_protocol_id"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired)

        with open(best_json, "r", encoding="utf-8") as f:
            repaired_best = json.load(f)
        self.assertEqual(repaired_best["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(repaired_best["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        self.assertEqual(repaired_best.get("custom_best_label"), "golden_candidate_v1")
        self.assertEqual(repaired_best.get("best_extra_flag"), True)

        # 2. Modify descriptor sha256 in best_metadata.json
        repaired_best["evaluation_preprocessing_descriptor_sha256"] = "tampered_best_sha"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(repaired_best, f)

        repaired2 = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertTrue(repaired2)

        with open(best_json, "r", encoding="utf-8") as f:
            repaired_best2 = json.load(f)
        self.assertEqual(repaired_best2["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)
        self.assertEqual(repaired_best2["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(repaired_best2.get("custom_best_label"), "golden_candidate_v1")
        self.assertEqual(repaired_best2.get("best_extra_flag"), True)

    def test_15_reconciliation_fails_closed_on_inconsistent_checkpoint_payload(self):
        """Assert reconciliation fails closed if authoritative checkpoint payload is missing or internally inconsistent."""
        cfg = copy.deepcopy(self.base_config)
        ckpt_dir = Path(self.test_dir) / "checkpoints_test15"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Metadata sidecar exists without any checkpoint payload -> FileNotFoundError
        orphan_json = ckpt_dir / "latest_metadata.json"
        orphan_json.write_text(json.dumps({"latest_epoch": 1}), encoding="utf-8")
        with self.assertRaises(FileNotFoundError) as ctx1:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("authoritative checkpoint payload", str(ctx1.exception))
        orphan_json.unlink()

        # 2. Authoritative latest_model.pt has tampered evaluation_preprocessing_id -> ValueError
        latest_pt = ckpt_dir / "latest_model.pt"
        trainer = self._build_trainer(config=cfg, ckpt_dir=ckpt_dir)
        run_id_dict = trainer._get_run_identity()
        bad_payload = {
            "epoch": 1,
            "model_state": trainer.model.state_dict(),
            "best_epoch": 1,
            "best_metric": 50.0,
            "best_model_state": trainer.model.state_dict(),
            "run_identity": run_id_dict,
            "config_hash": run_id_dict["config_hash"],
            "evaluation_preprocessing_id": "tampered_checkpoint_id",
            "evaluation_preprocessing_descriptor_sha256": EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
            "schema_version": "2.0",
            "checkpoint_type": "latest"
        }
        torch.save(bad_payload, latest_pt)
        with self.assertRaises(ValueError) as ctx2:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("Evaluation preprocessing protocol ID mismatch", str(ctx2.exception))

        # 3. Authoritative latest_model.pt has tampered evaluation_preprocessing_descriptor_sha256 -> ValueError
        bad_payload["evaluation_preprocessing_id"] = FROZEN_EVALUATION_PREPROCESSING_ID
        bad_payload["evaluation_preprocessing_descriptor_sha256"] = "tampered_payload_sha"
        torch.save(bad_payload, latest_pt)
        with self.assertRaises(ValueError) as ctx3:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("Evaluation preprocessing descriptor SHA-256 mismatch", str(ctx3.exception))

        # 4. Authoritative latest_model.pt is missing required provenance field -> ValueError
        del bad_payload["evaluation_preprocessing_descriptor_sha256"]
        torch.save(bad_payload, latest_pt)
        with self.assertRaises(ValueError) as ctx4:
            CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)
        self.assertIn("missing required field", str(ctx4.exception))

    def test_16_reconciliation_reconstructs_missing_best_model_pt(self):
        """Assert reconciliation reconstructs a missing best_model.pt from authoritative latest_model.pt."""
        cfg = copy.deepcopy(self.base_config)
        cfg["training"]["epochs"] = 1
        ckpt_dir = Path(self.test_dir) / "checkpoints_test16"
        trainer = self._build_trainer(config=cfg, ckpt_dir=ckpt_dir)
        train_loader, val_loader = create_dummy_loaders()
        trainer.fit(train_loader, val_loader)

        latest_pt = ckpt_dir / "latest_model.pt"
        best_pt = ckpt_dir / "best_model.pt"
        latest_json = ckpt_dir / "latest_metadata.json"
        best_json = ckpt_dir / "best_metadata.json"

        # 1. Verify minimal training run produced all 4 artifacts
        self.assertTrue(latest_pt.is_file(), "latest_model.pt must exist")
        self.assertTrue(best_pt.is_file(), "best_model.pt must exist")
        self.assertTrue(latest_json.is_file(), "latest_metadata.json must exist")
        self.assertTrue(best_json.is_file(), "best_metadata.json must exist")

        # 2. Load and inspect latest_model.pt to confirm complete, valid embedded best-model snapshot
        latest_ckpt = torch.load(latest_pt, map_location="cpu")
        self.assertIn("best_model_state", latest_ckpt, "latest_model.pt must contain embedded best_model_state")
        self.assertIsInstance(latest_ckpt["best_model_state"], dict)
        self.assertTrue(len(latest_ckpt["best_model_state"]) > 0)

        # 3. Record embedded best-model weights, best epoch, best metric, approved preprocessing protocol ID and descriptor fingerprint
        embedded_weights = latest_ckpt["best_model_state"]
        expected_best_epoch = latest_ckpt["best_epoch"]
        expected_best_metric = latest_ckpt["best_metric"]
        expected_eval_id = latest_ckpt["evaluation_preprocessing_id"]
        expected_eval_sha256 = latest_ckpt["evaluation_preprocessing_descriptor_sha256"]

        self.assertEqual(expected_eval_id, FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(expected_eval_sha256, EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

        # 4. Delete only best_model.pt. Leave latest_model.pt, its embedded snapshot, and best_metadata.json untouched
        best_pt.unlink()
        self.assertFalse(best_pt.exists(), "best_model.pt must be deleted for this test")
        self.assertTrue(latest_pt.exists(), "latest_model.pt must remain untouched")
        self.assertTrue(best_json.exists(), "best_metadata.json must remain untouched")

        # 5. Call CheckpointManager.reconcile_checkpoints()
        repaired = CheckpointManager.reconcile_checkpoints(ckpt_dir, cfg)

        # 6. Assertions
        # - Reconciliation succeeds and reports repair (True)
        self.assertTrue(repaired, "reconcile_checkpoints must report repair when best_model.pt was missing")
        # - best_model.pt is recreated
        self.assertTrue(best_pt.is_file(), "best_model.pt must be recreated by reconciliation")

        # - Inspect recreated best_model.pt
        recreated_best = torch.load(best_pt, map_location="cpu")
        self.assertEqual(recreated_best["checkpoint_type"], "best")
        self.assertEqual(recreated_best["epoch"], expected_best_epoch)
        self.assertEqual(recreated_best["best_epoch"], expected_best_epoch)
        self.assertEqual(recreated_best["best_metric"], expected_best_metric)
        self.assertEqual(recreated_best["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(recreated_best["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)

        # - Recreated model weights identical to embedded best snapshot
        self.assertEqual(set(recreated_best["model_state"].keys()), set(embedded_weights.keys()))
        for k, v in embedded_weights.items():
            self.assertTrue(
                torch.equal(recreated_best["model_state"][k], v.cpu()),
                f"Weight tensor for key '{k}' does not match embedded best snapshot"
            )

        # - The existing metadata sidecar is valid and consistent
        self.assertTrue(best_json.is_file(), "best_metadata.json must still exist")
        with open(best_json, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["checkpoint_identifier"], "best_model.pt")
        self.assertEqual(sidecar["best_epoch"], expected_best_epoch)
        self.assertEqual(sidecar["best_validation_accuracy"], expected_best_metric)
        self.assertEqual(sidecar["evaluation_preprocessing_id"], FROZEN_EVALUATION_PREPROCESSING_ID)
        self.assertEqual(sidecar["evaluation_preprocessing_descriptor_sha256"], EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256)


if __name__ == "__main__":
    unittest.main()
