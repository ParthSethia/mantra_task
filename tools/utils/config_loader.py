"""
Configuration loader utility for handling config files in different locations
"""
import yaml
import os

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from YAML file, checking multiple possible locations
    
    Args:
        config_path: Name of config file or path to config file
        
    Returns:
        dict: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file cannot be found in any location
    """
    # Try multiple possible config locations
    possible_paths = [
        config_path,
        f"config/{config_path}",
        f"../config/{config_path}",
        f"../../config/{config_path}",
        f"../../../config/{config_path}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    
    raise FileNotFoundError(f"Could not find config file. Tried: {possible_paths}")