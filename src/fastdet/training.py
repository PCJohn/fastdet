"""Booster fitting and importance-based column pruning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from catboost import CatBoostClassifier
from numpy.typing import NDArray  # noqa: TC002 -- used in dataclass field annotations at runtime

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .config import ModelConfig

__all__ = [
    "LeafGrid",
    "build_ranking",
    "calibrate_exit_stages",
    "default_feature_ranks_path",
    "fit_booster",
    "leaf_grid",
    "load_ranking",
    "quantise_leaves",
    "select_columns",
    "stage_tree_counts",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_LEAF_TABLE_NDIM = 2  # leaves are (n_trees, n_leaves)
_EMPTY_CHUNK_EXPONENT = -60  # a chunk whose trees are all constant: any step will do


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


@dataclass
class LeafGrid:
    """The low-bit leaf grid: ``leaf = offset[tree] + code * 2**(e_min + shift[chunk])``.

    Steps are powers of two, one per chunk of trees, so the scorer sums integer
    codes shifted by the chunk's exponent -- exactly, in any order.  Chunks restart
    at every stage boundary (``stage_starts``) so that a chunked, quantisation-aware
    fit and the export land on the same grid.
    """

    bits: int
    chunk: int
    stage_starts: tuple[int, ...]
    offsets: NDArray[np.float32]  # (n_trees,)
    exponents: NDArray[np.int64]  # (n_chunks,) step = 2**exponent
    codes: NDArray[np.uint8]  # (n_trees, n_leaves)

    @property
    def n_chunks(self) -> int:
        """Number of quantisation chunks."""
        return len(self.exponents)

    @property
    def e_min(self) -> int:
        """Smallest chunk exponent; every chunk's step is ``2**(e_min + shift)``."""
        return int(self.exponents.min())

    @property
    def shifts(self) -> NDArray[np.uint8]:
        """Per-chunk exponent above ``e_min`` (what the blob stores)."""
        return (self.exponents - self.e_min).astype(np.uint8)

    def chunk_of_tree(self, n_trees: int) -> NDArray[np.int64]:
        """Chunk index of every tree (chunks restart at each stage boundary)."""
        out = np.zeros(n_trees, dtype=np.int64)
        base = 0
        starts = [*self.stage_starts, n_trees]
        for stage in range(len(starts) - 1):
            lo, hi = starts[stage], starts[stage + 1]
            out[lo:hi] = base + (np.arange(hi - lo) // self.chunk)
            base += -(-(hi - lo) // self.chunk)
        return out

    def values(self) -> NDArray[np.float64]:
        """The leaf values on the grid, ``(n_trees, n_leaves)`` float64."""
        n_trees = len(self.offsets)
        step = np.ldexp(1.0, self.exponents[self.chunk_of_tree(n_trees)])[:, None]
        return self.offsets[:, None].astype(np.float64) + self.codes.astype(np.float64) * step

    @property
    def offset_sum(self) -> float:
        """Sum of the per-tree offsets."""
        return float(np.sum(self.offsets.astype(np.float64)))


def leaf_grid(
    leaves: NDArray[np.floating], bits: int, chunk: int, stage_starts: tuple[int, ...] = (0,)
) -> LeafGrid:
    """Quantise ``(n_trees, n_leaves)`` leaves onto the grid :class:`LeafGrid` describes.

    Each tree's offset is its smallest leaf; each chunk's step is the smallest power
    of two that spans the widest tree of the chunk with ``2**bits - 1`` codes.  Leaf
    magnitudes shrink as boosting proceeds, so one step for the whole model would
    turn the later trees into noise; a step per chunk keeps them resolved.
    """
    values = np.asarray(leaves, dtype=np.float64)
    if values.ndim != _LEAF_TABLE_NDIM:
        msg = "leaves must be (n_trees, n_leaves)"
        raise ValueError(msg)
    n_trees = len(values)
    levels = float((1 << bits) - 1)
    offsets = values.min(axis=1)
    ranges = values.max(axis=1) - offsets
    starts = [*stage_starts, n_trees]
    exponents: list[int] = []
    chunk_of: list[int] = []
    for stage in range(len(starts) - 1):
        for lo in range(starts[stage], starts[stage + 1], chunk):
            hi = min(lo + chunk, starts[stage + 1])
            widest = float(ranges[lo:hi].max())
            exponent = (
                math.ceil(math.log2(widest / levels)) if widest > 0.0 else _EMPTY_CHUNK_EXPONENT
            )
            chunk_of.extend([len(exponents)] * (hi - lo))
            exponents.append(exponent)
    exp_arr = np.asarray(exponents, dtype=np.int64)
    steps = np.ldexp(1.0, exp_arr[np.asarray(chunk_of, dtype=np.int64)])
    codes = np.rint((values - offsets[:, None]) / steps[:, None])
    codes = np.clip(codes, 0, levels).astype(np.uint8)
    return LeafGrid(
        bits=bits,
        chunk=chunk,
        stage_starts=tuple(stage_starts),
        offsets=offsets.astype(np.float32),
        exponents=exp_arr,
        codes=codes,
    )


def quantise_leaves(
    leaves: NDArray[np.floating], bits: int, chunk: int, stage_starts: tuple[int, ...] = (0,)
) -> NDArray[np.float64]:
    """Leaf values on the low-bit grid (see :func:`leaf_grid`), same shape as ``leaves``."""
    grid = leaf_grid(leaves, bits, chunk, stage_starts)
    # the float32 offset is what the blob stores, so it is what training fits against
    return grid.values()


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


def _fit_stage(  # noqa: PLR0913 -- a private helper with one call site per stage
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    params: Mapping[str, Any],
    *,
    n_trees: int,
    chunk: int,
    bits: int | None,
    running: NDArray[np.float64],
    ignored: list[int] | None,
) -> tuple[list[CatBoostClassifier], NDArray[np.float64]]:
    """Fit ``n_trees`` on top of ``running``; return the models and the new score.

    With ``bits`` the stage is fitted in chunks of ``chunk`` trees and each chunk's
    leaves are put on the grid before the next chunk starts, so later trees see
    -- and correct -- the rounding of earlier ones.  The grid is the one
    :func:`leaf_grid` builds for this stage on export (chunks counted from the
    stage's first tree), so the fitted model is the shipped model.  Without
    ``bits`` the stage is one fit.  ``ignored`` hides feature columns from this
    stage (resolution tiering).
    """
    from catboost import Pool  # noqa: PLC0415 -- optional heavy import

    stage = dict(params)
    if ignored:
        stage["ignored_features"] = list(ignored)
    step = chunk if bits is not None else n_trees
    models: list[CatBoostClassifier] = []
    score = running
    for start in range(0, n_trees, step):
        size = min(step, n_trees - start)
        fitted = CatBoostClassifier(iterations=size, boost_from_average=False, **stage)
        fitted.fit(Pool(features, labels, baseline=score))
        table = _leaf_table(fitted, size)
        if bits is not None:
            table = quantise_leaves(table, bits, chunk)
        indexes = fitted.calc_leaf_indexes(Pool(features))
        scale, bias = fitted.get_scale_and_bias()
        score = score + scale * table[np.arange(size)[None, :], indexes].sum(axis=1) + bias
        models.append(fitted)
    return models, score


def fit_booster(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
    feature_sides: list[int] | None = None,
) -> CatBoostClassifier:
    """Fit the frozen symmetric CatBoost booster and return it.

    Three knobs change how the trees are grown, and they compose:

    * ``quantisation_aware`` fits in chunks against the quantised running score,
      so the model is trained for the leaf grid the exporter writes;
    * ``coarse_trees`` grows the first trees on tile-constant features only
      (``feature_sides`` says which), leaving the rest to the fine stage;
    * neither set, a single ordinary fit -- unchanged from before.
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
    bits = model_cfg.leaf_bits if model_cfg.quantisation_aware else None
    coarse_trees = model_cfg.coarse_trees if model_cfg.coarse_trees < model_cfg.n_trees else 0
    if bits is None and not coarse_trees:
        booster = CatBoostClassifier(iterations=model_cfg.n_trees, **params)
        booster.fit(features, labels)
        return booster

    from catboost import sum_models  # noqa: PLC0415 -- optional heavy import

    models: list[CatBoostClassifier] = []
    score = np.zeros(len(labels), dtype=np.float64)
    if coarse_trees:
        if feature_sides is None:
            msg = "coarse_trees needs feature_sides (the pyramid side of each column)"
            raise ValueError(msg)
        ignored = [i for i, side in enumerate(feature_sides) if side > model_cfg.coarse_max_side]
        if len(ignored) == len(feature_sides):
            msg = f"no column has side <= coarse_max_side={model_cfg.coarse_max_side}"
            raise ValueError(msg)
        coarse, score = _fit_stage(
            features,
            labels,
            params,
            n_trees=coarse_trees,
            chunk=model_cfg.leaf_chunk,
            bits=bits,
            running=score,
            ignored=ignored,
        )
        models += coarse
    fine, _score = _fit_stage(
        features,
        labels,
        params,
        n_trees=model_cfg.n_trees - coarse_trees,
        chunk=model_cfg.leaf_chunk,
        bits=bits,
        running=score,
        ignored=None,
    )
    models += fine
    return models[0] if len(models) == 1 else sum_models(models)


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


def stage_tree_counts(model_cfg: ModelConfig) -> list[int]:
    """Tree counts at which the scorer may stop tiles.

    The coarse/fine boundary and ``exit_stage_fractions`` of the fine tier, each
    rounded to a chunk boundary of the fine stage so a chunked scorer can stop there.
    """
    coarse, n_trees, chunk = model_cfg.coarse_trees, model_cfg.n_trees, model_cfg.leaf_chunk
    counts: list[int] = []
    for frac in model_cfg.exit_stage_fractions:
        raw = coarse + frac * (n_trees - coarse)
        # stay on a chunk boundary of the fine stage so a chunked scorer can stop there
        trees = coarse + round((raw - coarse) / chunk) * chunk
        if 0 < trees < n_trees and trees not in counts:
            counts.append(trees)
    return sorted(counts)


def calibrate_exit_stages(
    partials: NDArray[np.float64],
    finals: NDArray[np.float64],
    stages: Sequence[int],
    *,
    keep_prob: float,
    margin: float,
) -> list[tuple[int, float]]:
    """Per-stage thresholds from the training cells.

    ``partials`` is ``(n_stages, n_cells)`` raw scores after each stage's trees and
    ``finals`` the ``(n_cells,)`` raw scores after every tree.  A stage's threshold is
    the lowest partial score of any cell whose final probability reaches
    ``keep_prob``, minus ``margin``: on the calibration cells no such cell can be
    stopped, and the margin covers what unseen images move.
    """
    keep = finals >= float(np.log(keep_prob / (1.0 - keep_prob)))
    out: list[tuple[int, float]] = []
    for i, trees in enumerate(stages):
        theta = float(partials[i][keep].min()) - margin if keep.any() else -1e30
        out.append((int(trees), theta))
    return out
