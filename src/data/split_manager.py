import json
import hashlib
from pathlib import Path
import numpy as np

FROZEN_SPLIT_HASH = "37a0b25b0d19c41318705a236055a7f375d9729dc51b8aa5d2b070f7716c9f91"
FROZEN_SPLIT_SEED = 42


def _canonical_hash(train_indices: list, val_indices: list) -> str:
    """SHA-256 contract used by the tracked frozen JSON artifact."""
    encoded = json.dumps(
        {"train_indices": train_indices, "val_indices": val_indices},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_split(data: dict) -> None:
    """Reject a modified or non-canonical final split artifact."""
    train, val = data.get("train_indices"), data.get("val_indices")
    metadata = data.get("metadata", {})
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError("Frozen split must contain train_indices and val_indices lists.")
    if len(train) != 45_000 or len(val) != 5_000:
        raise ValueError("Frozen split must contain exactly 45k train and 5k validation indices.")
    combined = np.asarray(train + val, dtype=np.int64)
    if np.unique(combined).size != 50_000 or combined.min() != 0 or combined.max() != 49_999:
        raise ValueError("Frozen split indices must be disjoint and cover CIFAR-100 training indices 0..49999.")
    if metadata.get("seed") != FROZEN_SPLIT_SEED:
        raise ValueError("Frozen split metadata seed must be 42.")
    calculated = _canonical_hash(train, val)
    if data.get("split_hash") != calculated or calculated != FROZEN_SPLIT_HASH:
        raise ValueError("Frozen split hash does not match the approved canonical artifact.")

def get_or_create_split(split_dir: str = "splits", seed: int = 42) -> tuple[list, list, str]:
    """
    Ensures a consistent 45k/5k CIFAR-100 train/validation split across all experiments.
    Generates and saves the split indices and computes a SHA-256 hash.
    """
    split_dir_path = Path(split_dir)
    split_dir_path.mkdir(exist_ok=True, parents=True)
    split_file = split_dir_path / "cifar100_split.json"
    
    if split_file.exists():
        with open(split_file, "r") as f:
            data = json.load(f)
        if split_dir_path.resolve() == Path("splits").resolve():
            validate_frozen_split(data)
        return data["train_indices"], data["val_indices"], data["split_hash"]

    if split_dir_path.resolve() == Path("splits").resolve():
        raise FileNotFoundError(
            "The tracked frozen split artifact is required at splits/cifar100_split.json; do not regenerate it."
        )
        
    # Total CIFAR-100 train size is 50000
    indices = np.arange(50000)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    train_indices = indices[:45000].tolist()
    val_indices = indices[45000:].tolist()
    
    # Compute deterministic hash
    split_hash = _canonical_hash(train_indices, val_indices)
    
    data = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "split_hash": split_hash,
        "metadata": {
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "seed": seed
        }
    }
    
    with open(split_file, "w") as f:
        json.dump(data, f, indent=4)
        
    return train_indices, val_indices, split_hash
