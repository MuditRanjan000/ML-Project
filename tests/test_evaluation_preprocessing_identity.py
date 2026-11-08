import unittest
import copy
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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


if __name__ == "__main__":
    unittest.main()
