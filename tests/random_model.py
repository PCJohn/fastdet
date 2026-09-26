"""A random model of the shipped shape, written by the real exporter.

Splits, thresholds and leaves are drawn, so no dataset or fit is needed, and the file is
real: the C++ scorer and the NumPy runtime both read it.  It serves the latency benchmark
(``test_latency.py``) and the thread gate (``test_native.py``), where what matters is the
shape -- 1000 trees of depth 5, a coarse tier, exit stages that drop some packs and keep
others -- rather than what the trees mean.

**A random model is only an approximation of a trained one**, and traversal cost depends
on its shape, so two properties are calibrated against a real trained model:
``THRESHOLDS_PER_FEATURE`` and ``LEVEL_SHARE``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fastdet import Config
from fastdet.exporter import build_blob
from fastdet.features import GRID, FeatureExtractor, feature_level_bits
from fastdet.runtime import parse_blob

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_TREES = 1000  # the shipped shape (ModelConfig defaults)
DEPTH = 5
LEAF_CHUNK = 16
# Calibration against a model trained on the real dataset (1178 columns): it held
# 6906 thresholds, i.e. ~6 per column, and binning cost is proportional to them.
THRESHOLDS_PER_FEATURE = 6
# Share of splits per feature level (side 64 .. 2, then image-wide).  Traversal cost
# depends on how many of a tree's splits vary inside a 4x4 tile (the coarser levels
# do not), so this drives the traversal number; it comes from a fitted model and is
# the least certain input here -- the benchmark's "splits per tree that vary" line is
# what to compare against a real model's.
LEVEL_SHARE = {64: 0.38, 32: 0.20, 16: 0.14, 8: 0.13, 4: 0.07, 2: 0.05, 1: 0.04}
COARSE_MAX_SIDE = 16  # features constant inside a 4x4 tile: what the coarse tier may split on


def _tile_max(cell_scores: NDArray[np.floating]) -> NDArray[np.floating]:
    """The highest score in each 4x4 tile of a flat 64x64 cell array."""
    return cell_scores.reshape(GRID // 4, 4, GRID // 4, 4).max(axis=(1, 3)).ravel()


def random_model(
    n_features: int,
    *,
    leaf_bits: int = 8,
    coarse_fraction: float = 2.0 / 3.0,
    keep_tiles: float = 0.5,
    fine_stage_keep: float | None = None,
) -> tuple[bytes, NDArray[np.float32], NDArray[np.float32]]:
    """``(blob, native fixture, dense fixture)`` for a random model.

    The model is over ``n_features`` of the default front-end's columns.  The first ``coarse_fraction`` of the trees split only on tile-constant features
    (side <= 16), like a fitted model's coarse tier.  Random leaves cannot be
    calibrated, so the exit stages are synthetic: the one after the coarse tier keeps
    about ``keep_tiles`` of the tiles alive on the fixture, and, when
    ``fine_stage_keep`` is given, a second one halfway through the fine tier keeps
    about that share of them -- a stage inside the fine tier that drops some packs but
    not all, which is what a threaded scorer has to get right.
    """
    rng = np.random.default_rng(0)
    names = FeatureExtractor(Config().train).base_names
    step = max(1, len(names) // n_features)
    columns = [names[i * step] for i in range(n_features)]
    shifts = [feature_level_bits(name)[1] for name in columns]
    sides = [GRID >> (shift >> 1) for shift in shifts]

    by_side: dict[int, list[int]] = {}
    for index, side in enumerate(sides):
        by_side.setdefault(side, []).append(index)
    pools = [s for s in by_side if LEVEL_SHARE.get(s, 0) > 0]
    weights = np.array([LEVEL_SHARE[s] for s in pools], dtype=float)
    weights /= weights.sum()

    counts = rng.integers(2, 2 * THRESHOLDS_PER_FEATURE, n_features)  # mean ~= the real model's
    borders = [np.sort(rng.uniform(0.05, 0.95, int(c))).tolist() for c in counts]
    coarse_trees = round(N_TREES * coarse_fraction)
    coarse_pools = [s for s in pools if s <= COARSE_MAX_SIDE]
    coarse_weights = np.array([LEVEL_SHARE[s] for s in coarse_pools], dtype=float)
    coarse_weights /= coarse_weights.sum()
    trees = []
    for tree in range(N_TREES):
        splits = []
        chosen = (
            rng.choice(coarse_pools, size=DEPTH, p=coarse_weights)
            if tree < coarse_trees
            else rng.choice(pools, size=DEPTH, p=weights)
        )
        for side in chosen:
            feature = int(rng.choice(by_side[int(side)]))
            cut = borders[feature][int(rng.integers(len(borders[feature])))]
            splits.append({"float_feature_index": feature, "border": cut})
        trees.append({"splits": splits, "leaf_values": rng.normal(0.0, 0.05, 1 << DEPTH).tolist()})
    model_json = {
        "features_info": {
            "float_features": [
                {"feature_index": f, "borders": borders[f]} for f in range(n_features)
            ]
        },
        "oblivious_trees": trees,
    }

    native = np.concatenate(
        [rng.uniform(0.0, 1.0, side * side).astype(np.float32) for side in sides]
    )
    dense = np.empty((GRID * GRID, n_features), dtype=np.float32)
    position = 0
    for column, side in enumerate(sides):
        block = native[position : position + side * side].reshape(side, side)
        position += side * side
        factor = GRID // side
        dense[:, column] = np.repeat(np.repeat(block, factor, 0), factor, 1).ravel()

    # the synthetic exit stages, from the partial scores of the unstaged model
    def export(exit_stages: list[tuple[int, float]]) -> bytes:
        blob, _info = build_blob(
            model_json,
            level_shift=shifts,
            leaf_bits=leaf_bits,
            leaf_chunk=LEAF_CHUNK,
            coarse_trees=coarse_trees,
            exit_stages=exit_stages,
        )
        return blob

    stage_trees = [coarse_trees]
    keeps = [keep_tiles]
    if fine_stage_keep is not None:
        fine_chunks = (N_TREES - coarse_trees) // LEAF_CHUNK
        stage_trees.append(coarse_trees + max(1, fine_chunks // 2) * LEAF_CHUNK)
        keeps.append(fine_stage_keep)
    runtime = parse_blob(export([]))
    partials = runtime.partial_scores(runtime.bins(dense), stage_trees)
    stages = [
        (trees, float(np.quantile(_tile_max(partial), 1.0 - keep)))
        for trees, partial, keep in zip(stage_trees, partials, keeps, strict=True)
    ]
    return export(stages), native, dense
