import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.contract import CORRUPTIONS, SEVERITIES, PredictionStore, ResultAccumulator, RunMetadata, calibration, load_records


class ContractEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = RunMetadata("synthetic_A_42", "resnet18", "A", 42, "best_model.pt", 4, 0.75, "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91", "5256477cf00b8bcaec855fb96067bcbe6e718235b083b92261dd407b501f989e")

    def test_all_75_cells_reproduce_pooled_ece_from_saved_records(self):
        clean_logits = np.array([[3., 0.], [0., 3.], [1., 0.]], dtype=np.float32)
        labels = np.array([0, 1, 1])
        with tempfile.TemporaryDirectory() as root:
            clean = PredictionStore.create(root, self.metadata, record_type="clean_cifar100", expected_samples=3, num_classes=2, temperature=1.7)
            clean.append(np.array([7, 9, 11]), clean_logits, labels)
            clean_path = clean.finalize()
            _, clean_records = load_records(clean_path)
            accumulator = ResultAccumulator(self.metadata, 1.7)
            accumulator.add_clean(clean_records["logits"], clean_records["labels"])
            stored_cells = {}
            for number, corruption in enumerate(CORRUPTIONS):
                for severity in SEVERITIES:
                    logits = clean_logits.copy()
                    if (number + severity) % 2: logits[0] = [0., 2.]
                    store = PredictionStore.create(root, self.metadata, record_type="cifar100c", corruption=corruption, severity=severity, expected_samples=3, num_classes=2, temperature=1.7)
                    store.append(np.array([2, 0]), logits[[2, 0]], labels[[2, 0]])
                    store.append(np.array([1]), logits[[1]], labels[[1]])
                    _, records = load_records(store.finalize())
                    stored_cells[(corruption, severity)] = records
                    self.assertEqual(records["image_indices"].tolist(), [2, 0, 1])
                    accumulator.add_cell(corruption, severity, records["logits"], records["labels"])
            runs, cells, bins = accumulator.finalize()
        self.assertEqual(len(cells), 75)
        self.assertEqual(len(bins), 210)
        self.assertTrue(np.isfinite(runs.loc[0, "corrupted_ece_raw_fraction"]))
        self.assertTrue(np.isfinite(runs.loc[0, "corrupted_ece_scaled_fraction"]))
        pooled_logits = np.concatenate([item["logits"] for item in stored_cells.values()])
        pooled_labels = np.concatenate([item["labels"] for item in stored_cells.values()])
        self.assertAlmostEqual(runs.loc[0, "corrupted_ece_raw_fraction"], calibration(pooled_logits, pooled_labels, 1.0)[0])
        self.assertAlmostEqual(runs.loc[0, "corrupted_ece_scaled_fraction"], calibration(pooled_logits, pooled_labels, 1.7)[0])
        for severity in SEVERITIES:
            logits = np.concatenate([stored_cells[(corruption, severity)]["logits"] for corruption in CORRUPTIONS])
            targets = np.concatenate([stored_cells[(corruption, severity)]["labels"] for corruption in CORRUPTIONS])
            for mode, temperature in (("raw", 1.0), ("scaled", 1.7)):
                expected = calibration(logits, targets, temperature)[0]
                subset = bins[(bins.slice_type == "severity") & (bins.slice_value == severity) & (bins.calibration == mode)]
                reconstructed = sum((row.n / subset.n.sum()) * abs(row.accuracy_fraction - row.mean_confidence_fraction) for row in subset.itertuples() if row.n)
                self.assertAlmostEqual(reconstructed, expected)

    def test_frozen_artifact_matches_compact_index_hash(self):
        artifact = Path(__file__).resolve().parents[1] / "splits" / "cifar100_split.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        encoded = json.dumps({"train_indices": payload["train_indices"], "val_indices": payload["val_indices"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), payload["split_hash"])

    def test_baseline_split_loader_validates_the_tracked_artifact(self):
        from src.data.split_manager import FROZEN_SPLIT_HASH, get_or_create_split
        train, validation, split_hash = get_or_create_split()
        self.assertEqual((len(train), len(validation)), (45_000, 5_000))
        self.assertEqual(split_hash, FROZEN_SPLIT_HASH)

    def test_incomplete_store_is_removed_and_reload_rejects_tampering(self):
        logits = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                with PredictionStore.create(root, self.metadata, record_type="clean_cifar100", expected_samples=3, num_classes=2, temperature=1.5) as store:
                    store.append(np.array([0, 1]), logits, labels)
                    raise RuntimeError("simulated evaluation failure")
            self.assertFalse((Path(root) / self.metadata.run_id / "clean_cifar100").exists())

            store = PredictionStore.create(root, self.metadata, record_type="clean_cifar100", expected_samples=2, num_classes=2, temperature=1.5)
            store.append(np.array([0, 1]), logits, labels)
            directory = store.finalize()
            predictions = np.load(directory / "predictions.npy", mmap_mode="r+")
            predictions[0] = 1
            predictions.flush()
            del predictions
            with self.assertRaisesRegex(ValueError, "do not reproduce"):
                load_records(directory)
