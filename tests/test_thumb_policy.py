"""The thumbnail policy: ``TrainConfig.thumb`` as a size in pixels or an imfeat rule.

``"pow2"`` (the default) sizes the thumbnail from the frame -- the largest power of two the
shorter side holds, square, never an upscale -- so the front-end's cost follows the frame
instead of every frame being squashed to one fixed size.  A model records the rule, a host
resolves it per frame with ``imfeat.thumb_size``, and the fused and the cv2 paths still
agree byte for byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import imfeat
import numpy as np
import pytest

from conftest import make_small_config
from fastdet import Config, Detector, TrainConfig
from fastdet.demo import front_end_note
from fastdet.features import GRID, FeatureCache, FeatureExtractor, stride_for, thumb_hw
from fastdet.tune import _parse_value

if TYPE_CHECKING:
    from pathlib import Path

# (frame shape, the "pow2" thumbnail): the rule, then the 64 px floor of the grid
POW2 = [
    ((720, 1280, 3), (512, 512)),
    ((1080, 1920, 3), (1024, 1024)),
    ((2160, 3840, 3), (2048, 2048)),
    ((300, 500, 3), (256, 256)),
    ((160, 160, 3), (128, 128)),
    ((96, 200, 3), (64, 64)),
    ((40, 50, 3), (64, 64)),  # below the grid: floored, so cv2 upscales as for a fixed size
    ((1000, 30, 3), (64, 64)),
]


def test_config_thumb_is_a_size_or_a_policy() -> None:
    assert TrainConfig().thumb == "pow2"
    assert TrainConfig(thumb=512).thumb == 512
    assert Config.from_dict(Config().to_dict()).train.thumb == "pow2"  # survives the JSON header
    assert Config.from_dict({"train": {"thumb": 1024}}).train.thumb == 1024  # an older model's
    for bad in ("pow4", "", 0, True, 2.5):
        with pytest.raises(ValueError, match="thumb must be"):
            TrainConfig(thumb=bad)  # type: ignore[arg-type]


def test_thumb_hw_and_stride() -> None:
    for shape, size in POW2:
        assert thumb_hw("pow2", shape) == size, shape
        assert thumb_hw("pow2", shape[:2]) == size
    # a size in pixels is that square whatever the frame, and is not floored
    assert thumb_hw(1024, (720, 1280, 3)) == (1024, 1024)
    assert thumb_hw(32, (40, 50, 3)) == (32, 32)
    # the stride is capped at the cell side, so no cell is left without a sample
    assert stride_for(4, (1024, 1024)) == 4
    assert stride_for(4, (256, 256)) == 4
    assert stride_for(4, (128, 128)) == 2
    assert stride_for(4, (64, 64)) == 1
    assert stride_for(1, (64, 64)) == 1
    assert stride_for(3, (512, 1024)) == 3


def test_policy_fused_and_cv2_paths_agree() -> None:
    """The fused and the cv2 paths agree under the policy.

    The thumbnail follows the frame, made inside imfeat's pass for every frame the rule
    keeps at or under the frame's size and by cv2 for the rest; the features are the same
    either way, and so is the size the host would resolve.
    """
    cfg = make_small_config().train
    cfg.thumb = "pow2"
    cfg.stride = 2
    fused = FeatureExtractor(cfg, threads=2)
    plain = FeatureExtractor(cfg, threads=2, fuse_resize=False)
    rng = np.random.default_rng(7)
    for shape, size in POW2[3:]:
        frame = rng.integers(0, 256, shape, np.uint8)
        assert fused.thumb_hw(frame.shape) == size
        assert fused.fuses_resize(frame) == (min(shape[:2]) >= GRID)
        a, b = fused.extract(frame), plain.extract(frame)
        for level in fused.levels:
            for x, y in zip(a[0][level], b[0][level], strict=True):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(a[1][level], b[1][level])
        if fused.fuses_resize(frame):
            assert fused._fused is not None
            assert fused._fused[1].thumb == size  # the computer was built for that size
    # one computer per thumbnail size on the cv2 side, no more
    assert set(plain._computers) == {size for shape, size in POW2[3:]}
    fused.close()
    plain.close()


@pytest.fixture(scope="module")
def policy_model(tiny_dataset: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model fitted with ``thumb="pow2"`` on the tiny dataset (160 px images: 128 px)."""
    images_dir, masks_dir = tiny_dataset
    cfg = make_small_config()
    cfg.train.thumb = "pow2"
    det = Detector(cfg).fit(images_dir, masks_dir, evaluate=False)
    path = det.export(tmp_path_factory.mktemp("model") / "pow2.fdt")
    det.close()
    return path


def test_policy_model_round_trip_and_host_recipe(
    tiny_dataset: tuple[Path, Path], policy_model: Path
) -> None:
    """The rule is in the exported model, and a host resolving it per frame scores as fastdet.

    With the frame handed to imfeat (the policy name passed straight through) or with the
    host's own cv2 thumbnail of the resolved size.
    """
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(policy_model, threads=2)
    assert det.config.train.thumb == "pow2"
    spec = det.front_end_spec
    assert spec["thumb"] == "pow2"
    grid = [(int(np.log2(n)),) * 2 for n in spec["levels"]]
    sample = np.asarray(cv2.imread(str(min(images_dir.glob("*.png")))), dtype=np.uint8)
    rng = np.random.default_rng(3)
    for frame in (sample, rng.integers(0, 256, (300, 500, 3), np.uint8)):
        rows, cols = imfeat.thumb_size(frame.shape, spec["thumb"])
        assert (rows, cols) == det.extractor.thumb_hw(frame.shape)
        want = det.predict_proba(frame)
        # the frame itself, the policy name passed straight to imfeat
        on_frame = imfeat.FeatureComputer(
            shape=frame.shape,
            grid=grid,
            stride=stride_for(spec["stride"], (rows, cols)),
            threads=2,
            feature_space=spec["space"],
            thumb=spec["thumb"],
        )
        assert on_frame.thumb == (rows, cols)
        np.testing.assert_array_equal(
            det.predict_from_imfeat(on_frame.features(frame), frame.shape[:2]), want
        )
        # the host's own cv2 thumbnail of the resolved size
        on_thumb = imfeat.FeatureComputer(
            shape=(rows, cols, 3),
            grid=grid,
            stride=stride_for(spec["stride"], (rows, cols)),
            threads=2,
            feature_space=spec["space"],
        )
        thumb = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(
            det.predict_from_imfeat(on_thumb.features(thumb), frame.shape[:2]), want
        )
        assert want.shape == (GRID, GRID)
    note = front_end_note(det, rng.integers(0, 256, (720, 1280, 3), np.uint8))
    assert "1280x720 frames -> 512x512 thumbnail (thumb='pow2') made inside imfeat's pass" in note
    note = front_end_note(det, rng.integers(0, 256, (40, 50, 3), np.uint8))
    assert "50x40 frames -> 64x64 thumbnail (thumb='pow2') by cv2.resize" in note
    det.close()


def test_feature_cache_under_policy(tiny_dataset: tuple[Path, Path]) -> None:
    """Training extraction resizes each image to its own thumbnail.

    The labels come from the mask on the 64x64 grid directly, so they do not depend on
    that size.
    """
    images_dir, masks_dir = tiny_dataset
    pairs = [
        (str(p), str(masks_dir / f"{p.stem}_mask.png")) for p in sorted(images_dir.glob("*.png"))
    ]
    cfg = make_small_config().train
    cfg.thumb = "pow2"
    cache = FeatureCache(pairs[:3], cfg, "test")
    assert cache.n == 3
    for level_maps, coverage in zip(cache.level_maps_list, cache.gt_coverage_list, strict=True):
        assert level_maps[GRID][0].shape[:2] == (GRID, GRID)
        assert coverage.shape == (GRID * GRID,)
        assert 0.0 <= coverage.min() <= coverage.max() <= 1.0
    fixed = make_small_config().train  # the same images at a fixed 256 px: different features
    cache_fixed = FeatureCache(pairs[:1], fixed, "test")
    assert not np.array_equal(
        cache.level_maps_list[0][GRID][0], cache_fixed.level_maps_list[0][GRID][0]
    )
    np.testing.assert_array_equal(cache.gt_coverage_list[0], cache_fixed.gt_coverage_list[0])


def test_tune_parses_the_thumb_knob() -> None:
    assert _parse_value("pow2", "int | str", "pow2") == "pow2"
    assert _parse_value("512", "int | str", "pow2") == 512
    assert _parse_value(" pow2-fit ", "int | str", "pow2") == "pow2-fit"
