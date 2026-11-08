import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch

class CheckpointManager:
    """
    Manages saving, loading, and reconciling model checkpoints with atomic interruption-safe writes.
    Maintains latest_model.pt as the authoritative training resumption checkpoint and ensures
    transactional consistency with best_model.pt and metadata sidecars.
    """
    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.logger = logging.getLogger("CheckpointManager")

    def set_best(self, epoch: int, metric: float, model_state: Optional[Dict[str, Any]] = None) -> None:
        """Restores best checkpoint tracking state upon resumption."""
        self.best_epoch = epoch
        self.best_metric = metric
        if model_state is not None:
            self.best_model_state = {
                k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            }

    @staticmethod
    def _atomic_torch_save(obj: Any, target_path: Path) -> None:
        """
        Safely serializes a PyTorch object to a temporary file before atomically
        replacing the target. Ensures an interrupted write cannot corrupt the target.
        """
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)

    @staticmethod
    def _atomic_json_save(data: Dict[str, Any], target_path: Path) -> None:
        """
        Safely writes JSON metadata to a temporary file, flushes to disk, and
        atomically replaces the destination.
        """
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    @classmethod
    def reconcile_checkpoints(
        cls,
        checkpoint_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Reconciles stale, missing, or interrupted checkpoint artifacts using latest_model.pt
        as the authoritative source of truth.
        Repairs latest_metadata.json, best_model.pt, and best_metadata.json if inconsistent.
        Returns True if any repairs were performed, False if already consistent.
        """
        logger = logging.getLogger("CheckpointManager")
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            return False

        # Clean up any abandoned temporary files from previous crashed writes
        for tmp_file in ckpt_dir.glob("*.tmp"):
            try:
                tmp_file.unlink()
                logger.info(f"Cleaned up abandoned temporary file: {tmp_file.name}")
            except OSError:
                pass

        latest_pt = ckpt_dir / "latest_model.pt"
        if not latest_pt.is_file():
            # No authoritative latest checkpoint exists
            return False

        try:
            latest_ckpt = torch.load(latest_pt, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Authoritative latest checkpoint '{latest_pt}' is corrupt or unreadable: {e}. "
                "Resumption cannot proceed with a corrupted authoritative checkpoint."
            ) from e

        if not isinstance(latest_ckpt, dict):
            raise ValueError(
                f"Authoritative checkpoint '{latest_pt}' payload is invalid (not a dictionary)."
            )

        required_keys = ["epoch", "model_state", "best_epoch", "best_metric", "best_model_state"]
        for key in required_keys:
            if key not in latest_ckpt or latest_ckpt[key] is None:
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' is missing required recovery field '{key}'. "
                    "Cannot safely reconcile checkpoint state."
                )

        repaired = False
        authoritative_epoch = int(latest_ckpt["epoch"])
        authoritative_best_epoch = int(latest_ckpt["best_epoch"])
        authoritative_best_metric = float(latest_ckpt["best_metric"])
        schema_version = latest_ckpt.get("schema_version", "2.0")

        # 1. Reconcile / Repair latest_metadata.json
        latest_json = ckpt_dir / "latest_metadata.json"
        needs_latest_meta_repair = True
        if latest_json.is_file():
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    curr_latest_meta = json.load(f)
                if (
                    curr_latest_meta.get("latest_epoch") == authoritative_epoch
                    and curr_latest_meta.get("best_epoch") == authoritative_best_epoch
                    and curr_latest_meta.get("checkpoint_identifier") == "latest_model.pt"
                ):
                    needs_latest_meta_repair = False
            except Exception:
                needs_latest_meta_repair = True

        if needs_latest_meta_repair:
            logger.warning(f"Reconciling stale/missing latest_metadata.json from {latest_pt.name}")
            latest_metadata = {
                "latest_epoch": authoritative_epoch,
                "latest_training_state": latest_ckpt.get("latest_training_state", "running"),
                "metric": float(latest_ckpt.get("metrics", {}).get("accuracy", authoritative_best_metric)),
                "best_epoch": authoritative_best_epoch,
                "best_validation_accuracy": authoritative_best_metric,
                "checkpoint_identifier": "latest_model.pt",
                "schema_version": schema_version,
                "config_hash": latest_ckpt.get("config_hash"),
                "evaluation_preprocessing_id": latest_ckpt.get("evaluation_preprocessing_id"),
                "split_hash": latest_ckpt.get("run_identity", {}).get("split_hash")
            }
            if latest_ckpt.get("best_metadata"):
                latest_metadata.update({
                    k: v for k, v in latest_ckpt["best_metadata"].items()
                    if k not in ["checkpoint_identifier", "latest_epoch", "metric"]
                })
            cls._atomic_json_save(latest_metadata, latest_json)
            repaired = True

        # 2. Reconcile / Repair best_model.pt and best_metadata.json
        best_pt = ckpt_dir / "best_model.pt"
        best_json = ckpt_dir / "best_metadata.json"

        needs_best_repair = False
        if not best_pt.is_file() or not best_json.is_file():
            needs_best_repair = True
        else:
            try:
                with open(best_json, "r", encoding="utf-8") as f:
                    curr_best_meta = json.load(f)
                if (
                    curr_best_meta.get("best_epoch") != authoritative_best_epoch
                    or curr_best_meta.get("best_validation_accuracy") != authoritative_best_metric
                ):
                    needs_best_repair = True
                else:
                    best_ckpt = torch.load(best_pt, map_location="cpu")
                    if (
                        best_ckpt.get("epoch") != authoritative_best_epoch
                        or best_ckpt.get("best_epoch") != authoritative_best_epoch
                    ):
                        needs_best_repair = True
            except Exception:
                needs_best_repair = True

        if needs_best_repair:
            logger.warning(
                f"Reconciling best_model.pt and best_metadata.json to match authoritative "
                f"best epoch {authoritative_best_epoch} ({authoritative_best_metric:.2f}%)"
            )
            best_state = {
                "epoch": authoritative_best_epoch,
                "model_state": latest_ckpt["best_model_state"],
                "best_epoch": authoritative_best_epoch,
                "best_metric": authoritative_best_metric,
                "checkpoint_type": "best",
                "schema_version": schema_version,
                "run_identity": latest_ckpt.get("run_identity"),
                "config_hash": latest_ckpt.get("config_hash"),
                "evaluation_preprocessing_id": latest_ckpt.get("evaluation_preprocessing_id"),
            }
            cls._atomic_torch_save(best_state, best_pt)

            best_metadata = {
                "best_epoch": authoritative_best_epoch,
                "best_validation_accuracy": authoritative_best_metric,
                "checkpoint_identifier": "best_model.pt",
                "schema_version": schema_version,
                "config_hash": latest_ckpt.get("config_hash"),
                "evaluation_preprocessing_id": latest_ckpt.get("evaluation_preprocessing_id"),
                "split_hash": latest_ckpt.get("run_identity", {}).get("split_hash")
            }
            if latest_ckpt.get("best_metadata"):
                best_metadata.update(latest_ckpt["best_metadata"])
            cls._atomic_json_save(best_metadata, best_json)
            repaired = True

        return repaired

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric: float,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save the model checkpoint with guaranteed consistency.
        latest_model.pt is atomically committed first as the authoritative state
        containing the complete best model snapshot, followed by metadata and best artifacts.
        """
        is_new_best = metric > self.best_metric
        if is_new_best:
            self.logger.info(f"New best model found! Metric improved from {self.best_metric:.4f} to {metric:.4f}")
            self.best_metric = metric
            self.best_epoch = epoch
            # Take an unmutated, detached snapshot of model parameters for the best model
            self.best_model_state = {
                k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state_dict["model_state"].items()
            }
        elif self.best_model_state is None:
            # Fallback for epoch 1 initialization if metric was <= -inf
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_model_state = {
                k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state_dict["model_state"].items()
            }

        # 1. Prepare best metadata sidecar content
        best_metadata = {
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_metric,
            "checkpoint_identifier": "best_model.pt",
            "schema_version": state_dict.get("schema_version", "2.0"),
            "config_hash": state_dict.get("config_hash"),
            "evaluation_preprocessing_id": state_dict.get("evaluation_preprocessing_id"),
            "split_hash": state_dict.get("run_identity", {}).get("split_hash")
        }
        if extra_metadata:
            best_metadata.update(extra_metadata)

        # 2. Save latest model atomically (AUTHORITATIVE RECORD)
        latest_state = dict(state_dict)
        latest_state["checkpoint_type"] = "latest"
        latest_state["epoch"] = epoch
        latest_state["best_metric"] = self.best_metric
        latest_state["best_epoch"] = self.best_epoch
        latest_state["best_model_state"] = self.best_model_state
        latest_state["best_metadata"] = best_metadata

        latest_filename = self.checkpoint_dir / "latest_model.pt"
        self._atomic_torch_save(latest_state, latest_filename)

        # 3. Save latest metadata sidecar atomically
        latest_metadata = {
            "latest_epoch": epoch,
            "latest_training_state": "running" if not extra_metadata or extra_metadata.get("latest_training_state") != "completed" else "completed",
            "metric": metric,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_metric,
            "checkpoint_identifier": "latest_model.pt",
            "schema_version": state_dict.get("schema_version", "2.0"),
            "config_hash": state_dict.get("config_hash"),
            "evaluation_preprocessing_id": state_dict.get("evaluation_preprocessing_id"),
            "split_hash": state_dict.get("run_identity", {}).get("split_hash")
        }
        if extra_metadata:
            latest_metadata.update(extra_metadata)

        self._atomic_json_save(latest_metadata, self.checkpoint_dir / "latest_metadata.json")

        # 4. If new best or best_model.pt doesn't exist, materialize best_model.pt and best_metadata.json
        best_filename = self.checkpoint_dir / "best_model.pt"
        if is_new_best or not best_filename.is_file():
            best_state = {
                "epoch": self.best_epoch,
                "model_state": self.best_model_state,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "checkpoint_type": "best",
                "schema_version": state_dict.get("schema_version", "2.0"),
                "run_identity": state_dict.get("run_identity"),
                "config_hash": state_dict.get("config_hash"),
                "evaluation_preprocessing_id": state_dict.get("evaluation_preprocessing_id"),
            }
            self._atomic_torch_save(best_state, best_filename)
            self._atomic_json_save(best_metadata, self.checkpoint_dir / "best_metadata.json")

        return is_new_best

    def load_checkpoint(self, filepath: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
        """
        Load a model checkpoint safely.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location)
