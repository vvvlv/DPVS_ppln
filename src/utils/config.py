"""Configuration loading utilities."""

from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        config_path: Path to experiment config file
    
    Returns:
        Complete configuration dictionary
    """
    config_path = Path(config_path)
    
    # Load experiment config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load referenced dataset config
    if 'dataset' in config:
        dataset_path = Path(config['dataset'])
        with open(dataset_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        config['dataset'] = dataset_config
    
    return config 