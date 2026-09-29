import json
import hashlib
from pathlib import Path
import numpy as np

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
        return data["train_indices"], data["val_indices"], data["split_hash"]
        
    # Total CIFAR-100 train size is 50000
    indices = np.arange(50000)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    train_indices = indices[:45000].tolist()
    val_indices = indices[45000:].tolist()
    
    # Compute deterministic hash
    hash_input = json.dumps({
        "train_indices": train_indices,
        "val_indices": val_indices
    }, sort_keys=True, separators=(',', ':')).encode('utf-8')
    split_hash = hashlib.sha256(hash_input).hexdigest()
    
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
