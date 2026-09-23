"""Validation scoring and pooled ranking metrics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .features import GRID, FeatureCache

if TYPE_CHECKING:
    from collections.abc import Callable

    from catboost import CatBoostClassifier

__all__ = ["paired_image_bootstrap", "pooled_pr_auc", "score_validation_per_image"]

BoolArray = NDArray[np.bool_]
Float64Array = NDArray[np.float64]

BOOTSTRAP_ALPHA = 0.05
MIN_BOOTSTRAP_IMAGES = 3


def pooled_pr_auc(scores: Float64Array, labels: BoolArray) -> float:
    """Precision-recall AUC pooled over all cells, via the step integral of PR."""
    target = np.asarray(labels).astype(np.float64)
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(target[order])
    fp = np.cumsum(1.0 - target[order])
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(target.sum()), 1e-12)
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def paired_image_bootstrap(
    scores_a_per_image: list[Float64Array],
    scores_b_per_image: list[Float64Array],
    labels_per_image: list[BoolArray],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, float, int] | None:
    """Paired image-level cluster bootstrap on the pooled PR-AUC difference (B - A).

    Each replicate resamples validation images with replacement, re-pools their
    cells and computes ``PR-AUC(B) - PR-AUC(A)`` on the same pool.  Returns
    ``(delta_mean, ci_lo, ci_hi, delta_observed, n_images)`` or ``None`` for
    fewer than :data:`MIN_BOOTSTRAP_IMAGES` images.
    """
    n = len(scores_a_per_image)
    if n < MIN_BOOTSTRAP_IMAGES or n != len(scores_b_per_image) or n != len(labels_per_image):
        return None
    scores_a = np.vstack(scores_a_per_image)
    scores_b = np.vstack(scores_b_per_image)
    labels_full = np.vstack(labels_per_image)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.randint(0, n, n)
        deltas[b] = pooled_pr_auc(scores_b[picks].ravel(), labels_full[picks].ravel()) - (
            pooled_pr_auc(scores_a[picks].ravel(), labels_full[picks].ravel())
        )
    lo, hi = np.percentile(deltas, [100.0 * BOOTSTRAP_ALPHA / 2, 100.0 * (1 - BOOTSTRAP_ALPHA / 2)])
    delta_observed = pooled_pr_auc(
        np.concatenate(scores_b_per_image), np.concatenate(labels_per_image)
    ) - pooled_pr_auc(np.concatenate(scores_a_per_image), np.concatenate(labels_per_image))
    return float(deltas.mean()), float(lo), float(hi), float(delta_observed), n


def score_validation_per_image(  # noqa: PLR0913 -- one optional override on a settled signature
    model: CatBoostClassifier,
    cache: FeatureCache,
    col_keep: NDArray[np.integer[Any]] | None = None,
    desc: str = "val",
    *,
    keep_grids: bool = False,
    scorer: Callable[[NDArray[np.floating[Any]]], Float64Array] | None = None,
) -> tuple[list[Float64Array], list[BoolArray], dict[int, Float64Array]]:
    """Score every image separately (the granularity the image bootstrap needs).

    Returns ``(scores_per_image, labels_per_image, grids)``, each per-image
    array holding ``GRID*GRID`` cells.  ``scorer`` overrides the booster, e.g. with
    the exported runtime's ``predict_proba`` when the shipped model differs from the
    fitted one (quantised leaves).
    """
    predict = scorer if scorer is not None else (lambda d: model.predict_proba(d)[:, 1])
    full = np.arange(GRID * GRID)
    scores_per_image: list[Float64Array] = []
    labels_per_image: list[BoolArray] = []
    grids: dict[int, Float64Array] = {}

    start = time.time()
    for i in range(cache.n):
        design = cache.gather(i, full, col_keep=col_keep)
        scores: Float64Array = np.asarray(predict(design), dtype=np.float64)
        labels_per_image.append(cache.gt_coverage_list[i] >= cache.cfg.gt_cell_thresh)
        scores_per_image.append(scores)
        if keep_grids:
            grids[i] = scores.reshape(GRID, GRID)
        if (i + 1) % 100 == 0:
            print(f"  [{desc}] scored {i + 1}/{cache.n} ({time.time() - start:.1f}s)")
    return scores_per_image, labels_per_image, grids
