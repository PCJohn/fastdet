"""The scorer's input packed into a buffer reused across frames (``Detector.pack_native``).

The packed features are a few megabytes per frame; a fresh array each time is a fresh set of
pages to fault in, so a detector keeps one buffer and writes it in place.  The bytes are the
same as a fresh array's, so every score is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Detector
from fastdet.demo import score_frame

if TYPE_CHECKING:
    from pathlib import Path


def test_native_into_a_buffer_is_the_same_bytes(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    ex = det.extractor
    frame = np.asarray(cv2.imread(str(min(images_dir.glob("*.png")))), dtype=np.uint8)
    level_maps, broadcast = ex.extract(frame)
    fresh = ex.native(level_maps, broadcast, det.col_keep)
    assert ex.native_size(det.col_keep) == fresh.shape[0] == det.native.native_size
    buf = np.full(fresh.shape[0], np.nan, np.float32)
    out = ex.native(level_maps, broadcast, det.col_keep, out=buf)
    assert out is buf
    np.testing.assert_array_equal(out, fresh)
    for bad in (
        np.empty(fresh.shape[0] + 1, np.float32),
        np.empty(fresh.shape[0], np.float64),
        np.empty((fresh.shape[0], 1), np.float32),
        np.empty(2 * fresh.shape[0], np.float32)[::2],
    ):
        with pytest.raises(ValueError, match="out must be"):
            ex.native(level_maps, broadcast, det.col_keep, out=bad)
    # the detector's own buffer: one array, written in place, the same scores as a fresh one
    packed = det.pack_native(level_maps, broadcast)
    np.testing.assert_array_equal(packed, fresh)
    assert det.pack_native(level_maps, broadcast) is packed
    np.testing.assert_array_equal(
        det.predict_proba(frame),
        det.native.score(fresh, use_exit=det.config.model.use_exit).reshape(64, 64),
    )
    probs, _feat_ms, _model_ms = score_frame(det, frame)  # the demo uses the buffer too
    np.testing.assert_array_equal(probs, det.predict_proba(frame))
    assert det.native_matrix(frame) is not det._native_buf  # native_matrix hands out a fresh array
    det.close()
    assert det._native_buf is None
