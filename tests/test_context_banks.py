"""The context banks in one C++ call per level, bit for bit the OpenCV route's.

``compute_context_values`` / ``compute_context2_values`` stay as the reference: 7 cv2 calls
per level on maps of at most 64x64 cells, whose cost was the calls, not the arithmetic.  The
extractor now fills a ``(7, grid, grid)`` buffer per level with ``context_banks`` -- the same
bytes (the box filter reproduces cv2's double sums and single rounding, the morphology its
max / min) -- and reuses that buffer while nothing else holds the banks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Detector
from fastdet.features import (
    CONTEXT2_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    GRID,
    N_BANK_PLANES,
    FeatureExtractor,
    compute_context2_values,
    compute_context_values,
    context_banks,
)
from fastdet.native import _extension

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config
    from fastdet.features import LevelBanks

N_CTX = len(CONTEXT_FEATURE_NAMES)


def _reference(pooled: np.ndarray, grid: int) -> np.ndarray:
    return np.concatenate(
        [
            compute_context_values(pooled, grid).transpose(2, 0, 1),
            compute_context2_values(pooled, grid).transpose(2, 0, 1),
        ]
    )


def _pooled(rng: np.random.Generator, grid: int, kind: int) -> np.ndarray:
    if kind == 0:  # any value in the cell-mean range
        return (rng.random((grid, grid)) * 255).astype(np.float32)
    if kind == 1:  # means of 16x9 cells: multiples of 1/144, not exact in float32
        return (rng.integers(0, 256 * 144, (grid, grid)) / 144.0).astype(np.float32)
    if kind == 2:  # whole grey levels
        return rng.integers(0, 256, (grid, grid)).astype(np.float32)
    if kind == 3:  # nearly flat: the differences are tiny
        base = np.full((grid, grid), 173.25, np.float32)
        return base + rng.integers(0, 2, (grid, grid)).astype(np.float32) * 0.5
    return (rng.random((grid, grid)) ** 6 * 255).astype(np.float32)  # skewed towards black


@pytest.mark.parametrize("grid", [64, 32, 16, 8, 4, 2, 9, 7, 5, 3, 1])
def test_context_banks_match_the_cv2_route(grid: int) -> None:
    rng = np.random.default_rng(grid)
    out = np.zeros((7, grid, grid), np.float32)
    for kind in range(5):
        for _ in range(8):
            pooled = _pooled(rng, grid, kind)
            got = context_banks(pooled, out)
            assert got is out
            np.testing.assert_array_equal(got, _reference(pooled, grid), err_msg=f"kind {kind}")


def test_context_banks_on_real_maps(tiny_dataset: tuple[Path, Path], small_config: Config) -> None:
    """The pooled means of real images, at every level of the pyramid."""
    images_dir, _masks_dir = tiny_dataset
    extractor = FeatureExtractor(small_config.train, threads=1)
    for sample in sorted(images_dir.glob("*.png"))[:3]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        result, extra = extractor.run_imfeat(frame)
        level_maps, _bvecs = extractor.compose(result, frame.shape[:2], extra)
        for size, banks in level_maps.items():
            pooled = extractor._lum_cell_mean(banks[0])
            want = _reference(pooled, size)
            ctx, ctx2 = banks[1], banks[2]
            assert ctx.shape == (size, size, N_CTX)
            assert ctx2.shape == (size, size, 2)
            np.testing.assert_array_equal(ctx.transpose(2, 0, 1), want[:N_CTX])
            np.testing.assert_array_equal(ctx2.transpose(2, 0, 1), want[N_CTX:])
            assert ctx.transpose(2, 0, 1).flags.c_contiguous  # each column a plane
    extractor.close()


def test_bank_buffers_are_reused_but_never_overwritten_under_a_holder(
    small_config: Config,
) -> None:
    extractor = FeatureExtractor(small_config.train, threads=1)
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (160, 160, 3), np.uint8)
    first, _ = extractor.extract(frame)
    kept = {size: banks[1].copy() for size, banks in first.items()}
    holder = first[GRID][1]  # still referenced: the next frame must not write over it
    buf_a = holder.base
    second, _ = extractor.extract(rng.integers(0, 256, (160, 160, 3), np.uint8))
    assert second[GRID][1].base is not buf_a  # a fresh buffer was made for this level
    np.testing.assert_array_equal(holder, kept[GRID])  # the held banks are intact
    del holder, first
    # the buffer's identity, not a reference to it: holding one is what the guard notices
    # (the extractor's own dict entry keeps it alive)
    buf_b = id(second[GRID][1].base)
    del second
    third, _ = extractor.extract(frame)  # nothing held now: the buffer is reused in place
    assert id(third[GRID][1].base) == buf_b
    np.testing.assert_array_equal(third[GRID][1], kept[GRID])
    fourth, _ = extractor.extract(frame)  # ... and the banks of a frame held do stay put
    assert id(fourth[GRID][1].base) != buf_b
    extractor.close()


def test_scores_unchanged_with_the_banks_in_cpp(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """A model's scores through the C++ banks equal the scores through the cv2 reference."""
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    ex = det.extractor
    for sample in sorted(images_dir.glob("*.png"))[:2]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        result, extra = ex.run_imfeat(frame)
        level_maps, bvecs = ex.compose(result, frame.shape[:2], extra)
        by_reference: dict[int, LevelBanks] = {}
        for size, banks in level_maps.items():
            pooled = ex._lum_cell_mean(banks[0])
            by_reference[size] = (
                banks[0],
                compute_context_values(pooled, size),
                compute_context2_values(pooled, size),
            )
        np.testing.assert_array_equal(
            ex.native(level_maps, bvecs, det.col_keep), ex.native(by_reference, bvecs, det.col_keep)
        )
        np.testing.assert_array_equal(
            det._predict_from_maps(level_maps, bvecs), det.predict_proba(frame)
        )
    det.close()


def test_coordinate_planes_are_written_once(small_config: Config) -> None:
    """A fresh buffer gets its coordinate planes; ``coords`` off leaves them alone.

    The extractor writes them when it makes a buffer and keeps them across frames.
    """
    rng = np.random.default_rng(11)
    out = np.zeros((7, 16, 16), np.float32)
    pooled = (rng.random((16, 16)) * 255).astype(np.float32)
    want = _reference(pooled, 16)
    np.testing.assert_array_equal(context_banks(pooled, out), want)
    out[:2] = 7.0
    np.testing.assert_array_equal(context_banks(pooled, out, coords=False)[2:], want[2:])
    assert (out[:2] == 7.0).all()  # untouched
    np.testing.assert_array_equal(context_banks(pooled, out)[:2], want[:2])  # restored
    # the extractor: a buffer's planes are written when it is made and kept after
    extractor = FeatureExtractor(small_config.train, threads=1)
    frame = rng.integers(0, 256, (160, 160, 3), np.uint8)
    for _ in range(3):
        level_maps, _bv = extractor.extract(frame)
        for size, banks in level_maps.items():
            np.testing.assert_array_equal(
                banks[1].transpose(2, 0, 1)[:2],
                _reference(np.zeros((size, size), np.float32), size)[:2],
            )
        del level_maps
    extractor.close()


def test_context_banks_arguments() -> None:
    with pytest.raises(ValueError, match="1 to 64"):
        context_banks(np.zeros((65, 65), np.float32), np.zeros((7, 65, 65), np.float32))
    with pytest.raises(TypeError):
        context_banks(np.zeros((8, 8), np.float64), np.zeros((7, 8, 8), np.float32))
    with pytest.raises(ValueError, match=r"\(7, g, g\)"):
        context_banks(np.zeros((8, 8), np.float32), np.zeros((6, 8, 8), np.float32))
    assert len(CONTEXT2_FEATURE_NAMES) + N_CTX == N_BANK_PLANES  # the buffer's planes


def test_all_levels_in_one_call_read_the_column_where_it_lies(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    """``context_banks_all`` on the raw maps == ``context_banks`` on a copy of each column."""
    ext = _extension()
    images_dir, _masks_dir = tiny_dataset
    extractor = FeatureExtractor(small_config.train, threads=1)
    frame = np.asarray(cv2.imread(str(min(images_dir.glob("*.png")))), dtype=np.uint8)
    result, _extra = extractor.run_imfeat(frame)
    maps = extractor._load_level_maps(result, "imfeat")
    column = extractor._lum_column
    # the raw maps as imfeat lays them out, and the same maps padded to twice the width
    # with the column at an odd stride, so the strides are what is read, not the shape
    wide = [np.zeros((m.shape[0], m.shape[1], 2 * m.shape[2] + 1), np.float32) for m in maps]
    for w, m in zip(wide, maps, strict=True):
        w[..., 1::2] = m
    for arrays, col in ((maps, column), (wide, 2 * column + 1)):
        outs = [np.zeros((N_BANK_PLANES, m.shape[0], m.shape[0]), np.float32) for m in maps]
        ext.context_banks_all(arrays, col, outs)
        for m, out in zip(maps, outs, strict=True):
            want = np.zeros_like(out)
            context_banks(extractor._lum_cell_mean(m), want, coords=False)
            np.testing.assert_array_equal(out[2:], want[2:])
            assert bool(out[2:].any()) == (m.shape[0] >= 3)  # the small levels are left zero
    # the extractor's compose goes through the one call: the same banks as the reference
    level_maps, _bvecs = extractor.compose(result, frame.shape[:2])
    for size, banks in level_maps.items():
        np.testing.assert_array_equal(
            np.concatenate([banks[1].transpose(2, 0, 1), banks[2].transpose(2, 0, 1)]),
            _reference(extractor._lum_cell_mean(banks[0]), size),
        )
    extractor.close()


def test_context_banks_all_arguments() -> None:
    ext = _extension()
    maps = [np.zeros((8, 8, 3), np.float32), np.zeros((4, 4, 3), np.float32)]
    outs = [np.zeros((7, 8, 8), np.float32), np.zeros((7, 4, 4), np.float32)]
    ext.context_banks_all(maps, 1, outs)  # fine
    with pytest.raises(ValueError, match="one entry per level"):
        ext.context_banks_all(maps, 1, outs[:1])
    with pytest.raises(ValueError, match="column"):
        ext.context_banks_all(maps, 3, outs)
    with pytest.raises(ValueError, match="float32"):
        ext.context_banks_all([maps[0].astype(np.float64), maps[1]], 1, outs)
    with pytest.raises(ValueError, match="C-contiguous"):
        ext.context_banks_all(maps, 1, [outs[0][:, ::-1], outs[1]])
    with pytest.raises(ValueError, match=r"\(7, g, g\)"):
        ext.context_banks_all(maps, 1, [outs[0], np.zeros((7, 8, 8), np.float32)])
    with pytest.raises(ValueError, match="1 to 64"):
        ext.context_banks_all(
            [np.zeros((65, 65, 3), np.float32)], 1, [np.zeros((7, 65, 65), np.float32)]
        )
