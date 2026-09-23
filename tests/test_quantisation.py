"""Leaf quantisation: the grid, the export, the quantisation-aware fit and the tiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Detector
from fastdet.artifact import ModelArtifact
from fastdet.runtime import parse_blob
from fastdet.training import leaf_grid, quantise_leaves, stage_tree_counts

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config


def _leaves(n_trees: int, n_leaves: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = np.geomspace(1.0, 1e-3, n_trees)[:, None]  # leaf magnitudes shrink as boosting proceeds
    return rng.normal(size=(n_trees, n_leaves)) * scale


def test_leaf_grid_is_power_of_two_per_chunk() -> None:
    leaves = _leaves(40, 8)
    grid = leaf_grid(leaves, bits=4, chunk=16, stage_starts=(0, 24))
    assert grid.n_chunks == 3  # 16 + 8 coarse, 16 fine: chunks restart at the stage boundary
    assert grid.chunk_of_tree(40).tolist() == [0] * 16 + [1] * 8 + [2] * 16
    assert grid.codes.max() <= 15
    values = grid.values()
    for tree in range(40):
        step = float(np.ldexp(1.0, int(grid.exponents[grid.chunk_of_tree(40)[tree]])))
        assert step == 2.0 ** round(np.log2(step))
        # every value is offset + code * step, and the rounding error is at most half a step
        assert np.all(np.abs(values[tree] - leaves[tree]) <= step / 2 + 1e-9)
    quantised = quantise_leaves(leaves, 4, 16, (0, 24))
    np.testing.assert_allclose(quantised, values, atol=1e-12)
    np.testing.assert_allclose(quantise_leaves(quantised, 4, 16, (0, 24)), quantised, atol=1e-6)


def test_leaf_grid_keeps_small_trees_resolved() -> None:
    leaves = _leaves(32, 8)
    per_chunk = quantise_leaves(leaves, 4, chunk=16)
    one_step = quantise_leaves(leaves, 4, chunk=32)
    assert len(np.unique(per_chunk[16:])) > len(np.unique(one_step[16:]))


def test_blob_stores_codes_that_reproduce_the_grid(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    """The blob's codes, offsets and shifts give exactly the leaves the exporter quantised to."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.leaf_bits = 4
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.booster is not None
    model = parse_blob(ModelArtifact.load(detector.export(tmp_path / "q4.fdt")).blob)
    assert model.leaf_bits == 4
    assert model.codes.max() <= 15
    raw = np.asarray(detector.booster.get_leaf_values(), dtype=np.float64).reshape(
        model.n_trees, -1
    )
    coarse = model.coarse_trees
    expected = quantise_leaves(
        raw, 4, small_config.model.leaf_chunk, (0, coarse) if coarse else (0,)
    )
    np.testing.assert_allclose(
        model.leaf_values.reshape(model.n_trees, -1), expected, rtol=0, atol=1e-6
    )
    sample = next(iter(sorted(images_dir.glob("*.png"))))
    reloaded = Detector.load(tmp_path / "q4.fdt")
    np.testing.assert_array_equal(reloaded.predict_proba(sample), detector.predict_proba(sample))


def test_quantisation_aware_fit_matches_its_export(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    """A chunked fit's running score is the exported runtime's raw score, chunk by chunk."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.n_trees = 24
    small_config.model.leaf_bits = 8
    small_config.model.leaf_chunk = 8
    small_config.model.coarse_fraction = 0.5
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.booster is not None
    assert detector.booster.tree_count_ == 24
    assert detector.runtime is not None
    assert detector.runtime.coarse_trees == 12
    # the exported model scores a training image exactly as the quantised booster does
    sample = next(iter(sorted(images_dir.glob("*.png"))))
    design = detector.design_matrix(sample)
    booster_logit = detector.booster.predict(design, prediction_type="RawFormulaVal")
    grid_leaves = detector.runtime.leaf_values.reshape(24, -1)
    raw_leaves = np.asarray(detector.booster.get_leaf_values(), dtype=np.float64).reshape(24, -1)
    # the booster still holds float leaves; the runtime holds the grid the fit was steered to
    assert np.abs(grid_leaves - raw_leaves).max() <= np.ldexp(
        1.0, detector.runtime.e_min + int(detector.runtime.shifts.max())
    )
    runtime_logit = np.log(
        detector.runtime.predict_proba(design) / (1 - detector.runtime.predict_proba(design))
    )
    assert (
        np.abs(runtime_logit - booster_logit).max() < 0.5
    )  # same trees, leaves rounded to the grid


def test_tiered_fit_keeps_the_coarse_stage_coarse(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    images_dir, masks_dir = tiny_dataset
    small_config.model.n_trees = 16
    small_config.model.coarse_fraction = 6 / 16
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.booster is not None
    assert detector.booster.tree_count_ == 16
    assert detector.feature_names is not None
    assert detector.runtime is not None
    assert detector.runtime.coarse_trees == 6
    from fastdet.features import feature_level_bits  # noqa: PLC0415

    sides = [feature_level_bits(n)[0] for n in detector.feature_names]
    splits = detector.runtime.splits["feat"].reshape(16, -1)
    coarse = [sides[int(f)] for f in splits[:6].reshape(-1)]
    fine = [sides[int(f)] for f in splits[6:].reshape(-1)]
    assert max(coarse) <= small_config.model.coarse_max_side
    assert max(fine) > small_config.model.coarse_max_side


def test_exit_stages_are_calibrated_and_applied(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    """Stages land on chunk boundaries, the blob carries them, and exit never lifts a score."""
    images_dir, masks_dir = tiny_dataset
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.runtime is not None
    stages = detector.runtime.exit_stages
    assert [t for t, _ in stages] == stage_tree_counts(small_config.model)
    coarse = small_config.model.coarse_trees
    assert all(
        (t - coarse) % small_config.model.leaf_chunk == 0 for t, _ in stages
    )  # chunk ends of the fine tier
    sample = next(iter(sorted(images_dir.glob("*.png"))))
    design = detector.design_matrix(sample)
    full = detector.runtime.predict_proba(design)
    exited = detector.runtime.predict_proba(design, use_exit=True)
    assert exited.shape == full.shape
    # a stopped cell keeps a partial score; every cell that reaches the keep probability is untouched
    keep = full >= small_config.model.exit_keep_prob
    np.testing.assert_array_equal(exited[keep], full[keep])


@pytest.mark.parametrize("bits", [4, 8])
def test_config_accepts_supported_leaf_bits(small_config: Config, bits: int) -> None:
    small_config.model.leaf_bits = bits
    small_config.model.__post_init__()
