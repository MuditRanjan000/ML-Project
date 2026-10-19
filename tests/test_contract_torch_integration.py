import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluation.contract import CORRUPTIONS, SEVERITIES, PredictionStore, ResultAccumulator, RunMetadata, load_records, write_tables
from src.evaluation.evaluator import Evaluator
from src.training.checkpointing import CheckpointManager


class ContractTorchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = RunMetadata(
            "torch_contract_seed42", "resnet18", "recipe_a", 42, "best_model.pt", 2,
            0.625, "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91",
            "5256477cf00b8bcaec855fb96067bcbe6e718235b083b92261dd407b501f989e",
        )

    def test_checkpoint_metadata_and_percent_conversion(self):
        model = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as root:
            manager = CheckpointManager(root)
            manager.save_checkpoint({"model_state": model.state_dict()}, epoch=2, metric=62.5)
            metadata = json.loads((Path(root) / "best_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["checkpoint_identifier"], "best_model.pt")
        self.assertEqual(metadata["best_epoch"], 2)
        self.assertEqual(metadata["best_validation_accuracy"], 62.5)
        self.assertEqual(Evaluator.validation_accuracy_fraction(62.5), 0.625)
        with self.assertRaises(ValueError):
            Evaluator.validation_accuracy_fraction(100.1)

    def test_torch_logits_produce_valid_records_and_canonical_tables(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(4, 2)
        inputs = torch.randn(6, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        logits = model(inputs).detach().numpy().astype(np.float32)
        targets = labels.numpy()
        with tempfile.TemporaryDirectory() as root:
            store = PredictionStore.create(root, self.metadata, record_type="clean_cifar100", expected_samples=6, num_classes=2, temperature=1.3)
            store.append(np.array([3, 0, 5]), logits[[3, 0, 5]], targets[[3, 0, 5]])
            store.append(np.array([1, 4, 2]), logits[[1, 4, 2]], targets[[1, 4, 2]])
            _, clean = load_records(store.finalize())
            self.assertEqual(set(clean), {"image_indices", "logits", "labels", "predictions", "confidence_raw", "confidence_scaled"})

            accumulator = ResultAccumulator(self.metadata, 1.3)
            accumulator.add_clean(clean["logits"], clean["labels"])
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    accumulator.add_cell(corruption, severity, logits, targets)
            tables = accumulator.finalize()
            write_tables(root, self.metadata, tables)
            self.assertTrue((Path(root) / "runs" / f"{self.metadata.run_id}.csv").is_file())
            self.assertTrue((Path(root) / "corruption_cells" / f"{self.metadata.run_id}.csv").is_file())
            self.assertTrue((Path(root) / "reliability_bins" / f"{self.metadata.run_id}.csv").is_file())
        runs, cells, bins = tables
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(cells), 75)
        self.assertEqual(len(bins), 210)
        self.assertIn("best_validation_accuracy_fraction", runs.columns)
        self.assertIn("corrupted_ece_scaled_fraction", runs.columns)
        self.assertEqual(set(bins["calibration"]), {"raw", "scaled"})


if __name__ == "__main__":
    unittest.main()
