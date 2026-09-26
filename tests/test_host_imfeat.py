"""Scoring from an imfeat result the host computed (the framegate integration path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Detector

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config


def test_predict_from_imfeat_matches_predict_proba(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    images_dir, masks_dir = tiny_dataset
    det = Detector.load(
        Detector(small_config).fit(images_dir, masks_dir, evaluate=False).export(tmp_path / "m.fdt")
    )
    spec = det.front_end_spec
    assert spec["thumb"] == small_config.train.thumb
    assert spec["levels"][0] == 64
    for sample in sorted(images_dir.glob("*.png"))[:2]:
        frame = cv2.imread(str(sample))
        # the host's pass: fastdet's own run_imfeat stands in for framegate here
        result, extra = det.extractor.run_imfeat(frame)
        from_host = det.predict_from_imfeat(result, frame.shape[:2], extra)
        np.testing.assert_array_equal(from_host, det.predict_proba(frame))
    with pytest.raises(ValueError, match="extra-scale results"):
        det.predict_from_imfeat(result, frame.shape[:2], [result])
