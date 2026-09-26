"""Scoring from an imfeat result the host computed (the framegate integration path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Detector

if TYPE_CHECKING:
    from pathlib import Path


def test_predict_from_imfeat_matches_predict_proba(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    spec = det.front_end_spec
    assert spec["thumb"] == det.config.train.thumb
    assert spec["levels"][0] == 64
    assert det.extractor._computers is None  # no pass yet: no imfeat pool either
    for sample in sorted(images_dir.glob("*.png"))[:2]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        # the host's pass: fastdet's own run_imfeat stands in for framegate here
        result, extra = det.extractor.run_imfeat(frame)
        from_host = det.predict_from_imfeat(result, frame.shape[:2], extra)
        np.testing.assert_array_equal(from_host, det.predict_proba(frame))
    with pytest.raises(ValueError, match="extra-scale results"):
        det.predict_from_imfeat(result, frame.shape[:2], [result])
    det.close()
