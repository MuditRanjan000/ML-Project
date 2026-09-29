import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger that logs to both console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logging
    if logger.handlers:
        return logger
        
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    # File handler
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
