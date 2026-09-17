"""Shared fixtures: a tiny synthetic images/masks dataset and a cheap config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Config, ModelConfig, TrainConfig

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

IMAGE_SIZE = 160
N_IMAGES = 8

Box = tuple[int, int, int, int]


def _make_image(rng: np.random.RandomState, boxes: list[Box]) -> NDArray[np.uint8]:
    img = np.asarray(rng.randint(40, 90, size=(IMAGE_SIZE, IMAGE_SIZE, 3)), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        img[y0:y1, x0:x1] = 235
    return img


def _make_mask(boxes: list[Box]) -> NDArray[np.uint8]:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 255
    return mask


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Write ``N_IMAGES`` textured images each with 1-2 bright masked boxes."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    rng = np.random.RandomState(0)
    for i in range(N_IMAGES):
        boxes: list[Box] = []
        for _ in range(1 + i % 2):
            x0 = int(rng.randint(8, IMAGE_SIZE - 56))
            y0 = int(rng.randint(8, IMAGE_SIZE - 56))
            boxes.append((x0, y0, x0 + 40, y0 + 40))
        cv2.imwrite(str(images_dir / f"img_{i:02d}.png"), _make_image(rng, boxes))
        cv2.imwrite(str(masks_dir / f"img_{i:02d}_mask.png"), _make_mask(boxes))
    return images_dir, masks_dir


@pytest.fixture
def small_config() -> Config:
    """A cheap but structurally complete config for end-to-end tests."""
    return Config(
        model=ModelConfig(depth=3, n_trees=60, learning_rate=0.2, border_count=15),
        train=TrainConfig(
            levels=(64, 32, 16, 8),
            thumb=256,
            top_k_features=0,
            val_frac=0.25,
            max_train_cells=20_000,
        ),
    )
