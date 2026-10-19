"""Load the project config (configs/config.yaml) as a plain dict.

Why a config file: paths differ between Kaggle and a laptop, and settings such as the
categorical column list are used by several modules. Keeping them in one place means
changing them once, not hunting through code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Priority: argument > AMEX_CONFIG env var > repo default."""
    path = Path(path or os.environ.get("AMEX_CONFIG", DEFAULT_CONFIG))
    with open(path) as f:
        return yaml.safe_load(f)
