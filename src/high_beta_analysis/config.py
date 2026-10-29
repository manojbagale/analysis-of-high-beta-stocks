"""Load a versioned experiment configuration."""
from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict) or not isinstance(settings.get("data"), dict):
        raise ValueError("Invalid project configuration")
    return settings
