"""Scoring from an imfeat result the host computed (the framegate integration path)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import imfeat
import numpy as np
import pytest

from fastdet import Detector
from fastdet.features import FeatureExtractor, exponent_for_size

if TYPE_CHECKING:
    from pathlib import Path


def _host_computer(spec: dict[str, Any], threads: int = 1) -> imfeat.FeatureComputer:
    """The computer a host builds from ``front_end_spec`` (what framegate does)."""
    return imfeat.FeatureComputer(
        shape=(spec["thumb"], spec["thumb"], 3),
        grid=[(exponent_for_size(n),) * 2 for n in spec["levels"]],
        stride=spec["stride"],
        threads=threads,
        feature_space=spec["space"] if spec["input_space"] == "bgr" else None,
    )


def test_predict_from_imfeat_matches_predict_proba(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    spec = det.front_end_spec
    assert spec["thumb"] == det.config.train.thumb
    assert spec["levels"][0] == 64
    assert (spec["space"], spec["input_space"]) == ("hsv", "bgr")  # imfeat converts
    assert det.extractor._computers is None  # no pass yet: no imfeat pool either
    host = _host_computer(spec)
    for sample in sorted(images_dir.glob("*.png"))[:2]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        # the host's pass, built from the spec alone, on its own BGR thumbnail
        thumb = cv2.resize(frame, (spec["thumb"],) * 2, interpolation=cv2.INTER_AREA)
        from_host = det.predict_from_imfeat(host.features(thumb), frame.shape[:2])
        np.testing.assert_array_equal(from_host, det.predict_proba(frame))
        # and fastdet's own two halves
        result, extra = det.extractor.run_imfeat(frame)
        from_halves = det.predict_from_imfeat(result, frame.shape[:2], extra)
        np.testing.assert_array_equal(from_halves, from_host)
    with pytest.raises(ValueError, match="extra-scale results"):
        det.predict_from_imfeat(result, frame.shape[:2], [result])
    det.close()


def test_fused_conversion_matches_cvtcolor(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """The features (so every model) are those of the cvtColor'd thumbnail, byte for byte.

    imfeat converts the BGR thumbnail inside its pass; a model trained on features of a
    thumbnail converted with OpenCV first must score exactly as before.
    """
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    spec = det.front_end_spec
    fused = _host_computer(spec)
    as_is = imfeat.FeatureComputer(
        shape=(spec["thumb"], spec["thumb"], 3),
        grid=[(exponent_for_size(n),) * 2 for n in spec["levels"]],
        stride=spec["stride"],
        threads=1,
        feature_space=None,
    )
    for sample in sorted(images_dir.glob("*.png"))[:2]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        thumb = cv2.resize(frame, (spec["thumb"],) * 2, interpolation=cv2.INTER_AREA)
        got = fused.features(thumb)
        want = as_is.features(cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV))
        for a, b in zip(got.maps, want.maps, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            det.predict_from_imfeat(got, frame.shape[:2]),
            det.predict_from_imfeat(want, frame.shape[:2]),
        )
    det.close()


def test_fused_resize_matches_cv2_path(tiny_model: Path) -> None:
    """The thumbnail made inside imfeat's pass gives the cv2 thumbnail's features.

    A frame at least the thumbnail's size goes into imfeat whole; the features, and so
    every score, are those of cv2.resize (INTER_AREA) followed by the pass, byte for byte.
    Smaller frames keep the cv2 path, a frame size change rebuilds the computer, and a
    host feeding the frame to its own computer gets the same map.
    """
    det = Detector.load(tiny_model, threads=2)
    cfg = det.config.train
    rng = np.random.default_rng(2)
    frames = [
        rng.integers(0, 256, (cfg.thumb + 120, cfg.thumb * 2, 3), np.uint8),  # fused
        rng.integers(0, 256, (cfg.thumb, cfg.thumb + 1, 3), np.uint8),  # another size
        rng.integers(0, 256, (cfg.thumb // 2, cfg.thumb * 2, 3), np.uint8),  # cv2: an upscale
    ]
    plain = FeatureExtractor(cfg, threads=2, fuse_resize=False)
    for frame in frames:
        fused = det.extractor
        assert fused._fuses(frame) == (frame.shape[0] >= cfg.thumb)
        a, b = fused.extract(frame), plain.extract(frame)
        for size in fused.levels:
            for x, y in zip(a[0][size], b[0][size], strict=True):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(a[1][size], b[1][size])
        np.testing.assert_array_equal(det.predict_proba(frame), _plain_predict(det, plain, frame))
    # the host recipe with the frame: the computer makes the thumbnail
    spec = det.front_end_spec
    frame = frames[0]
    host = imfeat.FeatureComputer(
        shape=frame.shape,
        grid=[(exponent_for_size(n),) * 2 for n in spec["levels"]],
        stride=spec["stride"],
        threads=2,
        feature_space=spec["space"],
        thumb=spec["thumb"],
    )
    np.testing.assert_array_equal(
        det.predict_from_imfeat(host.features(frame), frame.shape[:2]), det.predict_proba(frame)
    )
    det.close()
    plain.close()


def _plain_predict(det: Detector, plain: FeatureExtractor, frame: np.ndarray) -> np.ndarray:
    """``predict_proba`` through an extractor that resizes with cv2."""
    level_maps, broadcast = plain.extract(frame)
    return det._predict_from_maps(level_maps, broadcast)
