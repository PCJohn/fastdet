"""Booster fitting and importance-based column pruning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from catboost import CatBoostClassifier

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from .config import ModelConfig

__all__ = [
    "build_ranking",
    "default_feature_ranks_path",
    "fit_booster",
    "load_ranking",
    "quantise_leaves",
    "select_columns",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_LEAF_TABLE_NDIM = 2  # leaves are (n_trees, n_leaves)


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


def quantise_leaves(leaves: NDArray[np.floating], bits: int, chunk: int) -> NDArray[np.float64]:
    """Put leaf values on a low-bit grid: ``offset[tree] + code * step[chunk]``.

    ``leaves`` is ``(n_trees, n_leaves)``.  Each tree keeps its own offset (its
    smallest leaf) and each chunk of ``chunk`` trees shares one step, because leaf
    magnitudes shrink as boosting proceeds: one step for the whole model would
    quantise the later, smaller trees into noise.  ``code`` spans ``[0, 2**bits)``.

    Training (quantisation-aware fitting) and export call this, so the values a
    quantised model is fitted against are exactly the values the blob stores.
    """
    values = np.asarray(leaves, dtype=np.float64)
    if values.ndim != _LEAF_TABLE_NDIM:
        msg = "leaves must be (n_trees, n_leaves)"
        raise ValueError(msg)
    out = np.empty_like(values)
    levels = float((1 << bits) - 1)
    for start in range(0, len(values), chunk):
        block = values[start : start + chunk]
        offset = block.min(axis=1, keepdims=True)
        step = max(float((block.max(axis=1) - block.min(axis=1)).max()) / levels, 1e-12)
        out[start : start + chunk] = offset + np.round((block - offset) / step) * step
    return out


def _leaf_table(booster: CatBoostClassifier, n_trees: int) -> NDArray[np.float64]:
    """Leaves as ``(n_trees, max_leaves)``; short trees padded with their last leaf.

    Padding only fills positions no leaf index can reach, and repeats a real value,
    so it changes neither predictions nor a tree's leaf range.
    """
    flat = np.asarray(booster.get_leaf_values(), dtype=np.float64)
    counts = list(booster.get_tree_leaf_counts())
    table = np.zeros((n_trees, max(counts)), dtype=np.float64)
    at = 0
    for tree, count in enumerate(counts):
        table[tree, :count] = flat[at : at + count]
        table[tree, count:] = flat[at + count - 1]
        at += count
    return table


def _fit_quantisation_aware(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
    params: Mapping[str, Any],
) -> CatBoostClassifier:
    """Fit in chunks, quantising each chunk's leaves before the next one starts.

    Each chunk is fitted on the running quantised score as CatBoost ``baseline``, so
    a chunk sees -- and corrects -- the rounding error of every chunk before it.  The
    chunks are summed into one model whose (unquantised) leaves the exporter
    quantises with the same grid, reproducing the scores fitted here.
    """
    from catboost import Pool, sum_models  # noqa: PLC0415 -- optional heavy import

    bits = model_cfg.leaf_bits
    if bits is None:  # guarded by ModelConfig, restated for the type checker
        msg = "quantisation-aware fitting needs leaf_bits"
        raise ValueError(msg)
    chunk = model_cfg.leaf_chunk
    running = np.zeros(len(labels), dtype=np.float64)
    models: list[CatBoostClassifier] = []
    for start in range(0, model_cfg.n_trees, chunk):
        size = min(chunk, model_cfg.n_trees - start)
        fitted = CatBoostClassifier(iterations=size, boost_from_average=False, **params)
        fitted.fit(Pool(features, labels, baseline=running))
        table = quantise_leaves(_leaf_table(fitted, size), bits, chunk)
        indexes = fitted.calc_leaf_indexes(Pool(features))
        scale, bias = fitted.get_scale_and_bias()
        running += scale * table[np.arange(size)[None, :], indexes].sum(axis=1) + bias
        models.append(fitted)
    return sum_models(models)


def fit_booster(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
) -> CatBoostClassifier:
    """Fit the frozen symmetric CatBoost booster and return it.

    With ``quantisation_aware`` the trees are fitted in chunks against the quantised
    running score (see :func:`_fit_quantisation_aware`); otherwise one plain fit.
    """
    params: dict[str, Any] = {
        "loss_function": model_cfg.loss_function,
        "depth": model_cfg.depth,
        "learning_rate": model_cfg.learning_rate,
        "grow_policy": model_cfg.grow_policy,
        "random_seed": model_cfg.random_seed,
        "border_count": model_cfg.border_count,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if model_cfg.quantisation_aware:
        return _fit_quantisation_aware(features, labels, model_cfg, params)
    booster = CatBoostClassifier(iterations=model_cfg.n_trees, **params)
    booster.fit(features, labels)
    return booster


def build_ranking(booster: CatBoostClassifier, feature_names: Sequence[str]) -> dict[str, Any]:
    """Gain-importance ranking of a full-width fit, names in descending order.

    ``feature_names`` are the columns the booster was fitted on, in fitting order
    (CatBoost reports importances by position).  Fit with ``top_k_features=0`` so
    the ranking covers every front-end column; that is what :func:`select_columns`
    later indexes into.
    """
    gains = np.asarray(booster.get_feature_importance(type="FeatureImportance"), dtype=float)
    if len(gains) != len(feature_names):
        msg = f"{len(gains)} importances for {len(feature_names)} feature names"
        raise ValueError(msg)
    order = np.argsort(-gains, kind="stable")
    return {
        "names": [feature_names[int(i)] for i in order],
        "gains": [float(gains[int(i)]) for i in order],
        "width": len(feature_names),
    }
