"""The demo renders headless for an image and a short video."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from fastdet import Detector
from fastdet.demo import main, overlay, score_frame

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config


def test_demo_image_and_video_headless(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    images_dir, masks_dir = tiny_dataset
    model = (
        Detector(small_config).fit(images_dir, masks_dir, evaluate=False).export(tmp_path / "m.fdt")
    )
    sample = min(images_dir.glob("*.png"))
    out = tmp_path / "image.png"
    assert (
        main(["--model", str(model), "--source", str(sample), "--headless", "--output", str(out)])
        == 0
    )
    composite = cv2.imread(str(out))
    assert composite is not None
    assert composite.shape[1] > composite.shape[0]  # two panels side by side
    # a three-frame video
    frame = cv2.imread(str(sample))
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (frame.shape[1], frame.shape[0])
    )
    for _ in range(3):
        writer.write(frame)
    writer.release()
    out2 = tmp_path / "video.png"
    assert (
        main(
            [
                "--model",
                str(model),
                "--source",
                str(video),
                "--headless",
                "--output",
                str(out2),
                "--max-frames",
                "3",
            ]
        )
        == 0
    )
    assert out2.exists()
    det = Detector.load(model)
    probs, feature_ms, model_ms = score_frame(det, frame)
    assert probs.shape == (64, 64)
    assert feature_ms > 0
    assert model_ms > 0
    rgb = overlay(frame, probs)
    assert rgb.shape == frame.shape
    assert rgb.dtype == np.uint8
