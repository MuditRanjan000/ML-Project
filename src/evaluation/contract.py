"""Frozen evaluation contract: metrics, streaming records, and result tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
SEVERITIES = (1, 2, 3, 4, 5)
ECE_BINS = 15


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    architecture: str
    recipe: str
    training_seed: int
    checkpoint_id: str
    selected_epoch: int
    best_validation_accuracy_fraction: float
    split_hash: str
    evaluation_preprocessing_config_id: str


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibration(logits: np.ndarray, labels: np.ndarray, temperature: float) -> tuple[float, np.ndarray, np.ndarray, list[dict]]:
    """Return exact 15-bin top-label ECE and sufficient reliability statistics."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive.")
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = softmax(np.asarray(logits) / temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(labels.size), predictions]
    correct = predictions == labels
    bins: list[dict] = []
    ece = 0.0
    for index in range(ECE_BINS):
        lower, upper = index / ECE_BINS, (index + 1) / ECE_BINS
        # First bin is [0, 1/15]; all later bins are (lower, upper].
        mask = (confidence >= lower if index == 0 else confidence > lower) & (confidence <= upper)
        n = int(mask.sum())
        count_correct = int(correct[mask].sum())
        mean_confidence = float(confidence[mask].mean()) if n else None
        accuracy = count_correct / n if n else None
        if n:
            ece += n / labels.size * abs(accuracy - mean_confidence)
        bins.append({"bin_index": index, "bin_lower": lower, "bin_upper": upper,
                     "n": n, "correct": count_correct,
                     "mean_confidence_fraction": mean_confidence,
                     "accuracy_fraction": accuracy})
    return float(ece), predictions, confidence, bins


def _family(corruption: str) -> str:
    if corruption in {"gaussian_noise", "shot_noise", "impulse_noise"}: return "noise"
    if corruption in {"defocus_blur", "glass_blur", "motion_blur", "zoom_blur"}: return "blur"
    if corruption in {"snow", "frost", "fog", "brightness"}: return "weather"
    return "digital"


class PredictionStore:
    """Append-only per-slice NPY records without retaining all run predictions."""
    def __init__(self, directory: Path, manifest: dict, arrays: dict[str, np.memmap]) -> None:
        self.directory, self.manifest, self.arrays = directory, manifest, arrays
        self.offset = 0
        self._closed = False
        self._seen_indices = np.zeros(manifest["index_upper_bound"], dtype=bool)

    @classmethod
    def create(cls, root: str | Path, metadata: RunMetadata, *, record_type: str,
               expected_samples: int, num_classes: int, temperature: float,
               corruption: str | None = None, severity: int | None = None) -> "PredictionStore":
        if record_type == "clean_cifar100":
            directory = Path(root) / metadata.run_id / "clean_cifar100"
            index_field, ordering, upper = "original_test_image_index", "zero-based CIFAR-100 clean-test order", 10_000
        elif record_type == "cifar100c" and corruption in CORRUPTIONS and severity in SEVERITIES:
            directory = Path(root) / metadata.run_id / "cifar100c" / corruption / f"severity_{severity}"
            index_field, ordering, upper = "within_cell_image_index", "zero-based row offset in this published corruption/severity cell", expected_samples
        else:
            raise ValueError("Invalid prediction-record slice.")
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": "1.0", "run_id": metadata.run_id, "record_type": record_type,
                    "corruption": corruption, "severity": severity, "index_field": index_field,
                    "index_ordering": ordering, "expected_samples": expected_samples,
                    "index_upper_bound": upper, "num_classes": num_classes,
                    "temperature": float(temperature), "finalized": False}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        arrays = {"image_indices": np.lib.format.open_memmap(directory / "image_indices.npy", mode="w+", dtype=np.int64, shape=(expected_samples,)),
                  "logits": np.lib.format.open_memmap(directory / "logits.npy", mode="w+", dtype=np.float32, shape=(expected_samples, num_classes)),
                  "labels": np.lib.format.open_memmap(directory / "labels.npy", mode="w+", dtype=np.int64, shape=(expected_samples,)),
                  "predictions": np.lib.format.open_memmap(directory / "predictions.npy", mode="w+", dtype=np.int64, shape=(expected_samples,)),
                  "confidence_raw": np.lib.format.open_memmap(directory / "confidence_raw.npy", mode="w+", dtype=np.float32, shape=(expected_samples,)),
                  "confidence_scaled": np.lib.format.open_memmap(directory / "confidence_scaled.npy", mode="w+", dtype=np.float32, shape=(expected_samples,))}
        return cls(directory, manifest, arrays)

    def append(self, image_indices: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> None:
        indices, values, targets = np.asarray(image_indices), np.asarray(logits), np.asarray(labels)
        stop = self.offset + targets.size
        if values.shape != (targets.size, self.manifest["num_classes"]) or indices.shape != targets.shape or stop > self.manifest["expected_samples"]:
            raise ValueError("Prediction batch has an invalid shape or exceeds capacity.")
        if self._closed:
            raise RuntimeError("Cannot append to closed prediction records.")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Prediction indices and labels must be integers.")
        if (indices < 0).any() or (indices >= self.manifest["index_upper_bound"]).any():
            raise ValueError("Prediction indices are outside this record slice.")
        if len(np.unique(indices)) != indices.size or self._seen_indices[indices].any():
            raise ValueError("Prediction indices must be unique within a record slice.")
        if (targets < 0).any() or (targets >= self.manifest["num_classes"]).any() or not np.isfinite(values).all():
            raise ValueError("Prediction logits or labels are invalid.")
        _, raw_prediction, raw_confidence, _ = calibration(values, targets, 1.0)
        _, scaled_prediction, scaled_confidence, _ = calibration(values, targets, self.manifest["temperature"])
        if not np.array_equal(raw_prediction, scaled_prediction):
            raise RuntimeError("Temperature scaling changed predicted classes.")
        for name, value in {"image_indices": indices, "logits": values, "labels": targets, "predictions": raw_prediction, "confidence_raw": raw_confidence, "confidence_scaled": scaled_confidence}.items():
            self.arrays[name][self.offset:stop] = value
        self._seen_indices[indices] = True
        self.offset = stop

    def close(self) -> None:
        if not self._closed:
            for item in self.arrays.values():
                item.flush()
                item._mmap.close()
            self._closed = True

    def cleanup_incomplete(self) -> None:
        """Close and remove only this store when it has not been finalized."""
        if self.manifest.get("finalized"):
            return
        self.close()
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def __enter__(self) -> "PredictionStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None or not self.manifest.get("finalized"):
            self.cleanup_incomplete()
        return False

    def finalize(self) -> Path:
        if self.offset != self.manifest["expected_samples"]:
            raise RuntimeError("Cannot finalize incomplete prediction records.")
        self.close()
        self.manifest["finalized"] = True
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return self.directory


def load_records(directory: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("finalized"):
        raise ValueError("Prediction records are incomplete.")
    records = {name: np.load(directory / f"{name}.npy", allow_pickle=False)
               for name in ("image_indices", "logits", "labels", "predictions", "confidence_raw", "confidence_scaled")}
    n, classes = manifest["expected_samples"], manifest["num_classes"]
    if records["logits"].shape != (n, classes) or any(records[name].shape != (n,) for name in records if name != "logits"):
        raise ValueError("Prediction record array shapes do not match the manifest.")
    indices, labels, prediction = records["image_indices"], records["labels"], records["predictions"]
    if (not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != n
            or (indices < 0).any() or (indices >= manifest["index_upper_bound"]).any()):
        raise ValueError("Prediction record indices are invalid.")
    if (not np.isfinite(records["logits"]).all() or (labels < 0).any() or (labels >= classes).any()
            or (prediction < 0).any() or (prediction >= classes).any()):
        raise ValueError("Prediction record values are invalid.")
    raw_ece, expected_prediction, expected_raw, _ = calibration(records["logits"], labels, 1.0)
    _, scaled_prediction, expected_scaled, _ = calibration(records["logits"], labels, manifest["temperature"])
    if (not np.array_equal(prediction, expected_prediction) or not np.array_equal(expected_prediction, scaled_prediction)
            or not np.allclose(records["confidence_raw"], expected_raw, rtol=1e-6, atol=1e-7)
            or not np.allclose(records["confidence_scaled"], expected_scaled, rtol=1e-6, atol=1e-7)):
        raise ValueError("Prediction records do not reproduce their logits or confidence fields.")
    return manifest, records


class _BinAccumulator:
    def __init__(self) -> None:
        self.n = np.zeros(ECE_BINS, dtype=np.int64); self.correct = np.zeros(ECE_BINS, dtype=np.int64); self.confidence = np.zeros(ECE_BINS)

    def add(self, rows: list[dict]) -> None:
        for row in rows:
            index, count = row["bin_index"], row["n"]
            self.n[index] += count; self.correct[index] += row["correct"]
            if count: self.confidence[index] += count * row["mean_confidence_fraction"]

    def rows(self) -> tuple[float, list[dict]]:
        total = int(self.n.sum()); result = []; ece = 0.0
        if not total: raise ValueError("Cannot compute ECE without predictions.")
        for index, count in enumerate(self.n):
            lower, upper = index / ECE_BINS, (index + 1) / ECE_BINS
            confidence = float(self.confidence[index] / count) if count else None
            accuracy = float(self.correct[index] / count) if count else None
            if count: ece += count / total * abs(accuracy - confidence)
            result.append({"bin_index": index, "bin_lower": lower, "bin_upper": upper, "n": int(count), "correct": int(self.correct[index]), "mean_confidence_fraction": confidence, "accuracy_fraction": accuracy})
        return float(ece), result


class ResultAccumulator:
    """Incremental tables retaining only 15-bin sufficient statistics per slice."""
    def __init__(self, metadata: RunMetadata, temperature: float) -> None:
        self.metadata, self.temperature = metadata, float(temperature)
        self.clean = None; self.rows = []; self.seen = set()
        self.pooled = {"raw": _BinAccumulator(), "scaled": _BinAccumulator()}
        self.by_severity = {severity: {"raw": _BinAccumulator(), "scaled": _BinAccumulator()} for severity in SEVERITIES}

    def add_clean(self, logits: np.ndarray, labels: np.ndarray) -> None:
        if self.clean is not None: raise RuntimeError("Clean records already supplied.")
        raw, prediction, _, raw_bins = calibration(logits, labels, 1.0)
        scaled, scaled_prediction, _, scaled_bins = calibration(logits, labels, self.temperature)
        if not np.array_equal(prediction, scaled_prediction): raise RuntimeError("Temperature changed clean classes.")
        self.clean = (int(labels.size), int((prediction == labels).sum()), raw, scaled, raw_bins, scaled_bins)

    def add_cell(self, corruption: str, severity: int, logits: np.ndarray, labels: np.ndarray) -> None:
        key = (corruption, severity)
        if key not in {(c, s) for c in CORRUPTIONS for s in SEVERITIES} or key in self.seen: raise ValueError("Invalid or duplicate corruption cell.")
        raw, prediction, raw_confidence, raw_bins = calibration(logits, labels, 1.0)
        scaled, scaled_prediction, scaled_confidence, scaled_bins = calibration(logits, labels, self.temperature)
        if not np.array_equal(prediction, scaled_prediction): raise RuntimeError("Temperature changed corruption classes.")
        n, correct = int(labels.size), int((prediction == labels).sum())
        self.rows.append({"run_id": self.metadata.run_id, "corruption": corruption, "family": _family(corruption), "severity": severity, "n": n, "correct": correct, "incorrect": n-correct, "accuracy_fraction": correct/n, "error_fraction": 1-correct/n, "ece_raw_fraction": raw, "ece_scaled_fraction": scaled, "mean_confidence_raw_fraction": float(raw_confidence.mean()), "mean_confidence_scaled_fraction": float(scaled_confidence.mean())})
        for mode, bins in (("raw", raw_bins), ("scaled", scaled_bins)):
            self.pooled[mode].add(bins); self.by_severity[severity][mode].add(bins)
        self.seen.add(key)

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.clean is None or len(self.seen) != 75: raise RuntimeError("All clean and 75 corruption cells are required.")
        clean_n, clean_correct, clean_raw, clean_scaled, clean_raw_bins, clean_scaled_bins = self.clean
        pooled_raw, pooled_raw_bins = self.pooled["raw"].rows(); pooled_scaled, pooled_scaled_bins = self.pooled["scaled"].rows()
        cells = pd.DataFrame(self.rows); clean_error = 1-clean_correct/clean_n; mce = float(cells.error_fraction.mean())
        run = {**asdict(self.metadata), "temperature": self.temperature, "clean_correct": clean_correct, "clean_n": clean_n, "clean_accuracy_fraction": clean_correct/clean_n, "clean_error_fraction": clean_error, "mean_corrupted_error_fraction": mce, "aei_fraction": mce-clean_error, "clean_ece_raw_fraction": clean_raw, "clean_ece_scaled_fraction": clean_scaled, "corrupted_ece_raw_fraction": pooled_raw, "corrupted_ece_scaled_fraction": pooled_scaled}
        bins = []
        for split, kind, value, mode, source in (("test_clean", "all", "all", "raw", clean_raw_bins), ("test_clean", "all", "all", "scaled", clean_scaled_bins), ("cifar100c", "pooled", "all", "raw", pooled_raw_bins), ("cifar100c", "pooled", "all", "scaled", pooled_scaled_bins)):
            bins.extend({"run_id": self.metadata.run_id, "split": split, "slice_type": kind, "slice_value": value, "calibration": mode, **row} for row in source)
        for severity in SEVERITIES:
            for mode in ("raw", "scaled"):
                _, source = self.by_severity[severity][mode].rows()
                bins.extend({"run_id": self.metadata.run_id, "split": "cifar100c", "slice_type": "severity", "slice_value": severity, "calibration": mode, **row} for row in source)
        return pd.DataFrame([run]), cells, pd.DataFrame(bins)


def write_tables(root: str | Path, metadata: RunMetadata, tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    root = Path(root)
    for directory, table in zip((root / "runs", root / "corruption_cells", root / "reliability_bins"), tables):
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / f"{metadata.run_id}.csv", index=False)
