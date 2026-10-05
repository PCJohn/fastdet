"""Front-end invariants: imfeat never emits NaN or inf, and the cache owns its data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet.features import FeatureCache, FeatureExtractor, compute_global_stats

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from fastdet import Config

_SIDE = 96


def _frames() -> dict[str, NDArray[np.uint8]]:
    rng = np.random.RandomState(0)
    half = np.zeros((_SIDE, _SIDE, 3), dtype=np.uint8)
    half[:, _SIDE // 2 :] = 255
    ramp = np.tile(np.arange(_SIDE, dtype=np.uint8)[None, :, None], (_SIDE, 1, 3))
    return {
        "black": np.zeros((_SIDE, _SIDE, 3), dtype=np.uint8),
        "white": np.full((_SIDE, _SIDE, 3), 255, dtype=np.uint8),
        "flat": np.full((_SIDE, _SIDE, 3), 7, dtype=np.uint8),
        "noise": np.asarray(rng.randint(0, 256, size=(_SIDE, _SIDE, 3)), dtype=np.uint8),
        "binary": np.asarray(rng.randint(0, 2, size=(_SIDE, _SIDE, 3)) * 255, dtype=np.uint8),
        "step": half,
        "ramp": np.ascontiguousarray(ramp),
    }


@pytest.mark.parametrize("name", sorted(_frames()))
def test_front_end_is_finite_on_degenerate_frames(small_config: Config, name: str) -> None:
    extractor = FeatureExtractor(small_config.train)
    level_maps, broadcast_vecs = extractor.extract(_frames()[name])
    for size, banks in level_maps.items():
        for index, bank in enumerate(banks):
            assert np.isfinite(bank).all(), f"{name}: level {size} bank {index}"
    for size, vec in broadcast_vecs.items():
        assert np.isfinite(vec).all(), f"{name}: broadcast {size}"


def test_cache_does_not_hold_imfeat_views(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    """The training cache copies imfeat's maps instead of pinning its pooled blocks."""
    images_dir, masks_dir = tiny_dataset
    pairs = [
        (str(p), str(masks_dir / f"{p.stem}_mask.png")) for p in sorted(images_dir.glob("*.png"))
    ]
    cache = FeatureCache(pairs[:2], small_config.train, "test")
    for banks in cache.level_maps_list[0].values():
        for bank in banks:
            assert bank.base is None, "cached bank is a view into a larger buffer"


def test_global_stats_vector_is_the_numpy_formula() -> None:
    """The extension's vector == the NumPy formula it replaces, NaN and inf zeroed."""
    rng = np.random.default_rng(0)
    block = (rng.standard_normal(135) * 300).astype(np.float32)
    block[[3, 17, 40]] = [np.nan, np.inf, -np.inf]
    for h, w in ((1080, 1920), (0, 7), (3, 3), (4096, 1)):
        want = np.concatenate(
            [
                np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0),
                np.asarray([float(w / max(h, 1)), float(np.log1p(h * w))]),
            ]
        ).astype(np.float32)
        got = compute_global_stats(block, h, w)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, want)
    # a strided block is read where it lies
    np.testing.assert_array_equal(
        compute_global_stats(block[::3], 9, 16), compute_global_stats(block[::3].copy(), 9, 16)
    )
