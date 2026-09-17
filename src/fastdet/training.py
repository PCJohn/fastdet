"""Booster fitting and importance-based column pruning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from catboost import CatBoostClassifier

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from .config import ModelConfig

__all__ = [
    "default_feature_ranks_path",
    "fit_booster",
    "load_ranking",
    "select_columns",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"


def default_feature_ranks_path() -> Path:
    """Path of the bundled gain-importance ranking for the frozen front-end."""
    return _DATA_DIR / "feature_ranks.json"


def load_ranking(path: str | Path | None = None) -> dict[str, Any]:
    """Load a gain-importance ranking dict with a ``names`` list (descending)."""
    resolved = Path(path) if path is not None else default_feature_ranks_path()
    with resolved.open(encoding="utf-8") as fh:
        ranking: dict[str, Any] = json.load(fh)
    if not ranking.get("names"):
        msg = f"{resolved}: ranking has no 'names'"
        raise ValueError(msg)
    return ranking


def select_columns(
    base_names: list[str], ranking: Mapping[str, Any], top_k: int
) -> NDArray[np.int64]:
    """Indices (canonical order) of the top-``top_k`` ranked base columns.

    ``top_k <= 0`` keeps every column.  Missing ranked names are an error: a
    ranking for a different front-end silently drops real columns otherwise.
    """
    if top_k <= 0:
        return np.arange(len(base_names), dtype=np.int64)
    ranked = list(ranking["names"][:top_k])
    keep = set(ranked)
    missing = [n for n in ranked if n not in set(base_names)]
    if missing:
        msg = f"top-{top_k} ranked names not in this front-end: {missing[:5]}"
        raise ValueError(msg)
    columns: NDArray[np.int64] = np.array(
        [i for i, n in enumerate(base_names) if n in keep], dtype=np.int64
    )
    return columns


def fit_booster(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
) -> CatBoostClassifier:
    """Fit the frozen symmetric CatBoost booster and return it."""
    booster = CatBoostClassifier(
        loss_function=model_cfg.loss_function,
        iterations=model_cfg.n_trees,
        depth=model_cfg.depth,
        learning_rate=model_cfg.learning_rate,
        grow_policy=model_cfg.grow_policy,
        random_seed=model_cfg.random_seed,
        border_count=model_cfg.border_count,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    booster.fit(features, labels)
    return booster
