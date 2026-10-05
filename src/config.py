"""Loading of the YAML configuration file.

All parameters of the pipeline (paths, URLs, epoch boundaries, sample
sizes, ...) are stored in ``config.yaml`` so that data and logic are kept
apart.  This module turns the file into a plain dictionary and makes all
paths absolute, so the pipeline can be started from any directory.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    """Reads the configuration file and resolves all paths.

    Args:
      config_path (str or Path): Path to the YAML configuration file.

    Returns:
      dict: The configuration.  Every entry in the ``paths`` section is a
        ``pathlib.Path`` relative to the project root.

    Raises:
      FileNotFoundError: If the configuration file does not exist.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    config["paths"] = {
        name: PROJECT_ROOT / relative_path
        for name, relative_path in config["paths"].items()
    }
    return config
