import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch

from src.data.transforms import (
    validate_evaluation_preprocessing,
    FROZEN_EVALUATION_PREPROCESSING_ID,
    EXPECTED_PREPROCESSING_DESCRIPTOR_SHA256,
)
from src.data.split_manager import FROZEN_SPLIT_HASH

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
        best_pt = ckpt_dir / "best_model.pt"
        latest_json = ckpt_dir / "latest_metadata.json"
        best_json = ckpt_dir / "best_metadata.json"

        if not latest_pt.is_file() and not best_pt.is_file():
            # If sidecars exist without any checkpoint payload, fail closed
            if latest_json.is_file() or best_json.is_file():
                raise FileNotFoundError(
                    f"Cannot reconcile checkpoint sidecars in '{ckpt_dir}': "
                    "authoritative checkpoint payload (latest_model.pt or best_model.pt) is missing."
                )
            return False

        if latest_pt.is_file():
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

            auth_best_state = latest_ckpt["best_model_state"]
            if not isinstance(auth_best_state, dict) or len(auth_best_state) == 0:
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' has an invalid or empty embedded 'best_model_state'. "
                    "Cannot safely reconcile checkpoint state."
                )
            if not isinstance(latest_ckpt.get("best_epoch"), int) or latest_ckpt["best_epoch"] < 1:
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' has an invalid 'best_epoch'. "
                    "Cannot safely reconcile checkpoint state."
                )
            if int(latest_ckpt["best_epoch"]) > int(latest_ckpt["epoch"]):
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' is internally inconsistent: "
                    f"best_epoch ({latest_ckpt['best_epoch']}) > current epoch ({latest_ckpt['epoch']})."
                )

            # Validate authoritative preprocessing provenance
            auth_eval_id = latest_ckpt.get("evaluation_preprocessing_id")
            auth_eval_sha256 = latest_ckpt.get("evaluation_preprocessing_descriptor_sha256")
            if not isinstance(auth_eval_id, str) or not auth_eval_id:
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' is missing required field 'evaluation_preprocessing_id'."
                )
            if not isinstance(auth_eval_sha256, str) or not auth_eval_sha256:
                raise ValueError(
                    f"Authoritative checkpoint '{latest_pt}' is missing required field 'evaluation_preprocessing_descriptor_sha256'."
                )

            # Validate against frozen protocol identifier and descriptor SHA-256
            validate_evaluation_preprocessing(
                config=config,
                evaluation_preprocessing_id=auth_eval_id,
                evaluation_preprocessing_descriptor_sha256=auth_eval_sha256
            )

            # Check internal consistency with run_identity if present
            if isinstance(latest_ckpt.get("run_identity"), dict):
                run_id_dict = latest_ckpt["run_identity"]
                if run_id_dict.get("evaluation_preprocessing_id") != auth_eval_id:
                    raise ValueError(
                        f"Authoritative checkpoint '{latest_pt}' is internally inconsistent: "
                        f"run_identity has evaluation_preprocessing_id='{run_id_dict.get('evaluation_preprocessing_id')}', "
                        f"but payload has '{auth_eval_id}'."
                    )
                if run_id_dict.get("evaluation_preprocessing_descriptor_sha256") != auth_eval_sha256:
                    raise ValueError(
                        f"Authoritative checkpoint '{latest_pt}' is internally inconsistent: "
                        f"run_identity has evaluation_preprocessing_descriptor_sha256='{run_id_dict.get('evaluation_preprocessing_descriptor_sha256')}', "
                        f"but payload has '{auth_eval_sha256}'."
                    )

            repaired = False
            authoritative_epoch = int(latest_ckpt["epoch"])
            authoritative_best_epoch = int(latest_ckpt["best_epoch"])
            authoritative_best_metric = float(latest_ckpt["best_metric"])
            schema_version = latest_ckpt.get("schema_version", "2.0")

            # 1. Reconcile / Repair latest_metadata.json
            needs_latest_meta_repair = True
            curr_latest_meta = None
            if latest_json.is_file():
                try:
                    with open(latest_json, "r", encoding="utf-8") as f:
                        curr_latest_meta = json.load(f)
                    if (
                        isinstance(curr_latest_meta, dict)
                        and curr_latest_meta.get("latest_epoch") == authoritative_epoch
                        and curr_latest_meta.get("best_epoch") == authoritative_best_epoch
                        and curr_latest_meta.get("checkpoint_identifier") == "latest_model.pt"
                        and curr_latest_meta.get("evaluation_preprocessing_id") == auth_eval_id
                        and curr_latest_meta.get("evaluation_preprocessing_descriptor_sha256") == auth_eval_sha256
                    ):
                        needs_latest_meta_repair = False
                except Exception:
                    needs_latest_meta_repair = True

            if needs_latest_meta_repair:
                logger.warning(f"Reconciling stale/missing/inconsistent latest_metadata.json from {latest_pt.name}")
                repaired_latest_meta = dict(curr_latest_meta) if isinstance(curr_latest_meta, dict) else {}
                repaired_latest_meta.update({
                    "latest_epoch": authoritative_epoch,
                    "latest_training_state": latest_ckpt.get("latest_training_state", repaired_latest_meta.get("latest_training_state", "running")),
                    "metric": float(latest_ckpt.get("metrics", {}).get("accuracy", authoritative_best_metric)),
                    "best_epoch": authoritative_best_epoch,
                    "best_validation_accuracy": authoritative_best_metric,
                    "checkpoint_identifier": "latest_model.pt",
                    "schema_version": schema_version,
                    "config_hash": latest_ckpt.get("config_hash", repaired_latest_meta.get("config_hash")),
                    "evaluation_preprocessing_id": auth_eval_id,
                    "evaluation_preprocessing_descriptor_sha256": auth_eval_sha256,
                    "split_hash": latest_ckpt.get("run_identity", {}).get("split_hash", repaired_latest_meta.get("split_hash"))
                })
                if latest_ckpt.get("best_metadata"):
                    for k, v in latest_ckpt["best_metadata"].items():
                        if k not in ["checkpoint_identifier", "latest_epoch", "metric"]:
                            repaired_latest_meta.setdefault(k, v)
                cls._atomic_json_save(repaired_latest_meta, latest_json)
                repaired = True

            # 2. Reconcile / Repair best_model.pt and best_metadata.json
            needs_best_pt_repair = False
            if not best_pt.is_file():
                needs_best_pt_repair = True
            else:
                try:
                    best_ckpt = torch.load(best_pt, map_location="cpu")
                    if not isinstance(best_ckpt, dict):
                        needs_best_pt_repair = True
                    elif best_ckpt.get("epoch") != authoritative_best_epoch or best_ckpt.get("best_epoch") != authoritative_best_epoch:
                        needs_best_pt_repair = True
                    elif "best_metric" not in best_ckpt or abs(float(best_ckpt["best_metric"]) - authoritative_best_metric) > 1e-7:
                        needs_best_pt_repair = True
                    elif best_ckpt.get("evaluation_preprocessing_id") != auth_eval_id or best_ckpt.get("evaluation_preprocessing_descriptor_sha256") != auth_eval_sha256:
                        needs_best_pt_repair = True
                    elif "model_state" not in best_ckpt or not isinstance(best_ckpt["model_state"], dict):
                        needs_best_pt_repair = True
                    else:
                        best_weights = best_ckpt["model_state"]
                        if set(best_weights.keys()) != set(auth_best_state.keys()):
                            needs_best_pt_repair = True
                        else:
                            for t_name, auth_tensor in auth_best_state.items():
                                b_tensor = best_weights[t_name]
                                if isinstance(auth_tensor, torch.Tensor) and isinstance(b_tensor, torch.Tensor):
                                    if auth_tensor.dtype != b_tensor.dtype or auth_tensor.shape != b_tensor.shape or not torch.equal(auth_tensor.cpu(), b_tensor.cpu()):
                                        needs_best_pt_repair = True
                                        break
                                elif auth_tensor != b_tensor:
                                    needs_best_pt_repair = True
                                    break

                        # Compare run identity and split identity
                        if not needs_best_pt_repair:
                            if isinstance(latest_ckpt.get("run_identity"), dict):
                                auth_rid = latest_ckpt["run_identity"]
                                best_rid = best_ckpt.get("run_identity")
                                if not isinstance(best_rid, dict):
                                    needs_best_pt_repair = True
                                else:
                                    for id_key in ["run_id", "architecture", "recipe", "seed", "split_hash", "config_hash"]:
                                        if id_key in auth_rid and best_rid.get(id_key) != auth_rid.get(id_key):
                                            needs_best_pt_repair = True
                                            break
                            if latest_ckpt.get("config_hash") and best_ckpt.get("config_hash") != latest_ckpt.get("config_hash"):
                                needs_best_pt_repair = True
                except Exception:
                    needs_best_pt_repair = True

            needs_best_meta_repair = False
            curr_best_meta = None
            if not best_json.is_file():
                needs_best_meta_repair = True
            else:
                try:
                    with open(best_json, "r", encoding="utf-8") as f:
                        curr_best_meta = json.load(f)
                    if (
                        not isinstance(curr_best_meta, dict)
                        or curr_best_meta.get("best_epoch") != authoritative_best_epoch
                        or abs(float(curr_best_meta.get("best_validation_accuracy", -1.0)) - authoritative_best_metric) > 1e-7
                        or curr_best_meta.get("checkpoint_identifier") != "best_model.pt"
                        or curr_best_meta.get("evaluation_preprocessing_id") != auth_eval_id
                        or curr_best_meta.get("evaluation_preprocessing_descriptor_sha256") != auth_eval_sha256
                        or curr_best_meta.get("split_hash") != latest_ckpt.get("run_identity", {}).get("split_hash")
                        or (latest_ckpt.get("config_hash") and curr_best_meta.get("config_hash") != latest_ckpt.get("config_hash"))
                    ):
                        needs_best_meta_repair = True
                except Exception:
                    needs_best_meta_repair = True

            if needs_best_pt_repair:
                logger.warning(
                    f"Reconciling best_model.pt to match authoritative "
                    f"best epoch {authoritative_best_epoch} ({authoritative_best_metric:.2f}%)"
                )
                best_state = {
                    "epoch": authoritative_best_epoch,
                    "model_state": auth_best_state,
                    "best_epoch": authoritative_best_epoch,
                    "best_metric": authoritative_best_metric,
                    "checkpoint_type": "best",
                    "schema_version": schema_version,
                    "run_identity": latest_ckpt.get("run_identity"),
                    "config_hash": latest_ckpt.get("config_hash"),
                    "evaluation_preprocessing_id": auth_eval_id,
                    "evaluation_preprocessing_descriptor_sha256": auth_eval_sha256,
                }
                cls._atomic_torch_save(best_state, best_pt)
                repaired = True

            if needs_best_meta_repair:
                logger.warning(
                    f"Reconciling best_metadata.json to match authoritative "
                    f"best epoch {authoritative_best_epoch} ({authoritative_best_metric:.2f}%)"
                )
                repaired_best_meta = dict(curr_best_meta) if isinstance(curr_best_meta, dict) else {}
                repaired_best_meta.update({
                    "best_epoch": authoritative_best_epoch,
                    "best_validation_accuracy": authoritative_best_metric,
                    "checkpoint_identifier": "best_model.pt",
                    "schema_version": schema_version,
                    "config_hash": latest_ckpt.get("config_hash", repaired_best_meta.get("config_hash")),
                    "evaluation_preprocessing_id": auth_eval_id,
                    "evaluation_preprocessing_descriptor_sha256": auth_eval_sha256,
                    "split_hash": latest_ckpt.get("run_identity", {}).get("split_hash", repaired_best_meta.get("split_hash"))
                })
                if latest_ckpt.get("best_metadata"):
                    for k, v in latest_ckpt["best_metadata"].items():
                        repaired_best_meta.setdefault(k, v)
                cls._atomic_json_save(repaired_best_meta, best_json)
                repaired = True

            return repaired

        else:
            # Standalone best checkpoint artifact (latest_model.pt is absent).
            # NOTE: An isolated best_model.pt cannot independently verify historical training
            # trajectory, epoch sequence, or peak metric authenticity without latest_model.pt.
            # Structural inspection and sidecar repair are provided for ad-hoc inspection only;
            # official contract evaluation strictly fails closed and requires latest_model.pt.
            if not best_pt.is_file():
                return False

            try:
                best_ckpt = torch.load(best_pt, map_location="cpu")
            except Exception as e:
                raise RuntimeError(
                    f"Standalone checkpoint '{best_pt}' is corrupt or unreadable: {e}."
                ) from e

            if not isinstance(best_ckpt, dict):
                raise ValueError(f"Standalone checkpoint '{best_pt}' payload is invalid (not a dictionary).")

            # 1. Preprocessing provenance validation
            auth_best_eval_id = best_ckpt.get("evaluation_preprocessing_id")
            auth_best_eval_sha256 = best_ckpt.get("evaluation_preprocessing_descriptor_sha256")
            if not isinstance(auth_best_eval_id, str) or not auth_best_eval_id:
                raise ValueError(f"Standalone checkpoint '{best_pt}' is missing required field 'evaluation_preprocessing_id'.")
            if not isinstance(auth_best_eval_sha256, str) or not auth_best_eval_sha256:
                raise ValueError(f"Standalone checkpoint '{best_pt}' is missing required field 'evaluation_preprocessing_descriptor_sha256'.")

            validate_evaluation_preprocessing(
                config=config,
                evaluation_preprocessing_id=auth_best_eval_id,
                evaluation_preprocessing_descriptor_sha256=auth_best_eval_sha256
            )

            # 2. Validate epoch fields
            best_epoch_val = best_ckpt.get("best_epoch", best_ckpt.get("epoch"))
            if best_epoch_val is None or not isinstance(best_epoch_val, int) or best_epoch_val < 1:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' has invalid or missing best_epoch.")
            if "epoch" in best_ckpt and best_ckpt["epoch"] != best_epoch_val:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' has epoch ({best_ckpt['epoch']}) != best_epoch ({best_epoch_val}).")
            max_epochs = config.get("training", {}).get("epochs", 100) if isinstance(config, dict) else 100
            if best_epoch_val > max_epochs:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' best_epoch ({best_epoch_val}) exceeds configured max epochs ({max_epochs}).")

            # 3. Validate validation metric field
            if "best_metric" not in best_ckpt or best_ckpt["best_metric"] is None:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' is missing required field 'best_metric'.")
            try:
                best_metric_val = float(best_ckpt["best_metric"])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' has non-numeric best_metric: {best_ckpt.get('best_metric')}.") from e
            if not 0.0 <= best_metric_val <= 100.0:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' best_metric ({best_metric_val}) is outside percentage range [0, 100].")

            # 4. Validate split identity
            expected_split_hash = (
                config.get("data", {}).get("split_hash")
                or config.get("experiment", {}).get("split_hash")
                or FROZEN_SPLIT_HASH
            ) if isinstance(config, dict) else FROZEN_SPLIT_HASH

            ckpt_split_hash = best_ckpt.get("split_hash") or (best_ckpt.get("run_identity", {}).get("split_hash") if isinstance(best_ckpt.get("run_identity"), dict) else None)
            if ckpt_split_hash is not None and ckpt_split_hash != expected_split_hash:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' split hash ('{ckpt_split_hash}') does not match expected split hash ('{expected_split_hash}').")

            # 5. Validate run identity and config hash against active config
            if isinstance(config, dict):
                from src.training.trainer import compute_config_hash
                expected_config_hash = compute_config_hash(config, expected_split_hash)
                if best_ckpt.get("config_hash") and best_ckpt.get("config_hash") != expected_config_hash:
                    raise ValueError(f"Standalone best checkpoint '{best_pt}' config_hash ('{best_ckpt.get('config_hash')}') does not match expected config hash ('{expected_config_hash}').")

                if isinstance(best_ckpt.get("run_identity"), dict):
                    b_rid = best_ckpt["run_identity"]
                    exp_arch = config.get("experiment", {}).get("model_name") or config.get("model", {}).get("architecture")
                    exp_recipe = config.get("experiment", {}).get("recipe")
                    exp_seed = config.get("experiment", {}).get("seed")
                    if exp_arch and b_rid.get("architecture") and b_rid.get("architecture") != exp_arch:
                        raise ValueError(f"Standalone best checkpoint '{best_pt}' architecture ('{b_rid.get('architecture')}') does not match expected ('{exp_arch}').")
                    if exp_recipe and b_rid.get("recipe") and b_rid.get("recipe") != exp_recipe:
                        raise ValueError(f"Standalone best checkpoint '{best_pt}' recipe ('{b_rid.get('recipe')}') does not match expected ('{exp_recipe}').")
                    if exp_seed is not None and b_rid.get("seed") is not None and b_rid.get("seed") != exp_seed:
                        raise ValueError(f"Standalone best checkpoint '{best_pt}' seed ('{b_rid.get('seed')}') does not match expected ('{exp_seed}').")

            # 6. Validate model_state
            if "model_state" not in best_ckpt or not isinstance(best_ckpt["model_state"], dict) or len(best_ckpt["model_state"]) == 0:
                raise ValueError(f"Standalone best checkpoint '{best_pt}' has invalid or empty 'model_state'.")

            # 7. Validate / Reconcile best_metadata.json
            needs_best_meta_repair = False
            curr_best_meta = None
            if not best_json.is_file():
                needs_best_meta_repair = True
            else:
                try:
                    with open(best_json, "r", encoding="utf-8") as f:
                        curr_best_meta = json.load(f)
                    if not isinstance(curr_best_meta, dict):
                        needs_best_meta_repair = True
                    else:
                        # Sidecar must not contradict standalone best payload!
                        if curr_best_meta.get("best_epoch") != best_epoch_val:
                            raise ValueError(
                                f"Standalone best metadata sidecar 'best_metadata.json' best_epoch ({curr_best_meta.get('best_epoch')}) "
                                f"contradicts best_model.pt best_epoch ({best_epoch_val})."
                            )
                        if abs(float(curr_best_meta.get("best_validation_accuracy", -1.0)) - best_metric_val) > 1e-7:
                            raise ValueError(
                                f"Standalone best metadata sidecar 'best_metadata.json' accuracy ({curr_best_meta.get('best_validation_accuracy')}) "
                                f"contradicts best_model.pt best_metric ({best_metric_val})."
                            )
                        if curr_best_meta.get("evaluation_preprocessing_id") != auth_best_eval_id:
                            raise ValueError(
                                f"Standalone best metadata sidecar evaluation_preprocessing_id ('{curr_best_meta.get('evaluation_preprocessing_id')}') "
                                f"contradicts checkpoint ('{auth_best_eval_id}')."
                            )
                        if curr_best_meta.get("evaluation_preprocessing_descriptor_sha256") != auth_best_eval_sha256:
                            raise ValueError(
                                f"Standalone best metadata sidecar evaluation_preprocessing_descriptor_sha256 ('{curr_best_meta.get('evaluation_preprocessing_descriptor_sha256')}') "
                                f"contradicts checkpoint ('{auth_best_eval_sha256}')."
                            )
                        if curr_best_meta.get("split_hash") and curr_best_meta.get("split_hash") != expected_split_hash:
                            raise ValueError(
                                f"Standalone best metadata sidecar split_hash ('{curr_best_meta.get('split_hash')}') "
                                f"does not match expected split hash ('{expected_split_hash}')."
                            )
                        if curr_best_meta.get("checkpoint_identifier") != "best_model.pt":
                            needs_best_meta_repair = True
                except ValueError:
                    raise
                except Exception:
                    needs_best_meta_repair = True

            if needs_best_meta_repair:
                logger.warning(
                    f"Reconciling best_metadata.json from standalone {best_pt.name} for structural inspection. "
                    f"WARNING: Standalone artifacts lack historical trajectory provenance and cannot be evaluated under official contract evaluation."
                )
                repaired_best_meta = dict(curr_best_meta) if isinstance(curr_best_meta, dict) else {}
                repaired_best_meta.update({
                    "best_epoch": best_epoch_val,
                    "best_validation_accuracy": best_metric_val,
                    "checkpoint_identifier": "best_model.pt",
                    "schema_version": best_ckpt.get("schema_version", "2.0"),
                    "config_hash": best_ckpt.get("config_hash", repaired_best_meta.get("config_hash")),
                    "evaluation_preprocessing_id": auth_best_eval_id,
                    "evaluation_preprocessing_descriptor_sha256": auth_best_eval_sha256,
                    "split_hash": ckpt_split_hash or expected_split_hash
                })
                cls._atomic_json_save(repaired_best_meta, best_json)
                return True

            return False

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
            "evaluation_preprocessing_descriptor_sha256": state_dict.get("evaluation_preprocessing_descriptor_sha256"),
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
            "evaluation_preprocessing_descriptor_sha256": state_dict.get("evaluation_preprocessing_descriptor_sha256"),
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
                "evaluation_preprocessing_descriptor_sha256": state_dict.get("evaluation_preprocessing_descriptor_sha256"),
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
