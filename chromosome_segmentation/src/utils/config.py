"""Small config helpers shared by local, Kaggle, and Colab entry points."""
from __future__ import annotations

from pathlib import Path


def dataset_path(config: dict, configured_path: str | Path) -> Path:
    """Resolve a dataset path against optional ``paths.data_root``.

    Absolute configured paths always win. Relative paths preserve their
    existing local behavior when ``data_root`` is ``.``.
    """
    path = Path(configured_path).expanduser()
    if path.is_absolute(): return path
    return Path(config.get("paths", {}).get("data_root", ".")).expanduser() / path
