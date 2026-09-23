"""Leaf quantisation and quantisation-aware training.

The property that makes quantisation-aware fitting coherent: training fits each
chunk against the *quantised* score of the chunks before it, and the exporter puts
the same grid on the same leaves, so the blob holds exactly the values that were
fitted against.  Both call :func:`fastdet.training.quantise_leaves`; these tests pin
that they agree, and that the grid is what it claims to be.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Config, Detector
from fastdet.runtime import parse_blob
from fastdet.training import quantise_leaves

if TYPE_CHECKING:
    from pathlib import Path


def test_quantise_leaves_grid() -> None:
    """Per chunk: at most 2**bits distinct codes, each tree offset by its own minimum."""
    rng = np.random.default_rng(0)
    leaves = rng.normal(0.0, 1.0, (10, 8)) * np.array([[2.0]] * 5 + [[0.05]] * 5)  # shrinking
    quantised = quantise_leaves(leaves, bits=4, chunk=5)
    assert quantised.shape == leaves.shape
    for start in (0, 5):
        block, source = quantised[start : start + 5], leaves[start : start + 5]
        offsets = source.min(axis=1, keepdims=True)
        step = max(float((source.max(axis=1) - source.min(axis=1)).max()) / 15.0, 1e-12)
        codes = np.round((block - offsets) / step)
        np.testing.assert_allclose(block, offsets + codes * step, atol=1e-12)
        assert codes.min() >= 0
        assert codes.max() <= 15  # 2**4 - 1 levels
    # A second pass moves nothing: the values are already on their own grid.
    np.testing.assert_allclose(quantise_leaves(quantised, bits=4, chunk=5), quantised, atol=1e-12)


def test_quantise_leaves_keeps_small_trees_resolved() -> None:
    """A per-chunk step, not one global step, is what keeps late trees resolved."""
    big = np.linspace(-1.0, 1.0, 8)[None, :].repeat(16, axis=0)
    small = (big * 0.01)[:16]
    leaves = np.concatenate([big, small])
    per_chunk = quantise_leaves(leaves, bits=4, chunk=16)
    one_step = quantise_leaves(leaves, bits=4, chunk=32)
    assert len(np.unique(per_chunk[16:])) > len(np.unique(one_step[16:]))
    assert len(np.unique(one_step[16:])) <= 3  # the small trees collapse


@pytest.mark.parametrize("bits", [8, 4])
def test_export_quantises_leaves(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path, bits: int
) -> None:
    """The exported blob holds the quantised leaves, and both runtimes read them."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.leaf_bits = bits
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    blob, _info = detector._build_blob()
    model = parse_blob(blob)

    leaves = model.leaf_values.reshape(model.n_trees, -1)
    expected = quantise_leaves(leaves, bits, detector.config.model.leaf_chunk)
    np.testing.assert_allclose(leaves, expected, rtol=0, atol=1e-6)
    for start in range(0, model.n_trees, detector.config.model.leaf_chunk):
        block = leaves[start : start + detector.config.model.leaf_chunk]
        assert len(np.unique(np.round((block - block.min(axis=1, keepdims=True)), 9))) <= (
            1 << bits
        ) * len(block)

    reloaded = Detector.load(detector.export(tmp_path / f"q{bits}.fdt"))
    sample = images_dir / "img_00.png"
    np.testing.assert_array_equal(reloaded.predict_proba(sample), detector.predict_proba(sample))


def test_quantisation_aware_fit_matches_its_export(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    """A quantisation-aware fit exports the grid it was fitted against."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.leaf_bits = 4
    small_config.model.leaf_chunk = 8
    small_config.model.quantisation_aware = True
    small_config.model.n_trees = 20  # not a multiple of the chunk: exercises the tail

    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.booster is not None
    assert detector.booster.tree_count_ == 20

    json_path = tmp_path / "booster.json"
    detector.booster.save_model(str(json_path), format="json")
    model_json = json.loads(json_path.read_text(encoding="utf-8"))
    raw = np.asarray(
        [tree["leaf_values"] for tree in model_json["oblivious_trees"]], dtype=np.float64
    )
    blob, _info = detector._build_blob()
    stored = parse_blob(blob).leaf_values.reshape(20, -1)
    np.testing.assert_allclose(stored, quantise_leaves(raw, 4, 8), rtol=0, atol=1e-6)

    reloaded = Detector.load(detector.export(tmp_path / "qat.fdt"))
    sample = images_dir / "img_00.png"
    np.testing.assert_array_equal(reloaded.predict_proba(sample), detector.predict_proba(sample))
