import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config if config is not None else {}

def merge_configs(base_config: Dict, update_config: Dict) -> Dict:
    """
    Recursively merge update_config into base_config.
    """
    for key, value in update_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config
