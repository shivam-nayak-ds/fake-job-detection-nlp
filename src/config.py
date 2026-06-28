import os
import yaml


def get_config() -> dict:
    """Load and return the project config from config/config.yaml."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Convenience: load once at import time
CONFIG = get_config()