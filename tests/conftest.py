"""Shared fixtures: a tiny synthetic images/masks dataset and a cheap config."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from fastdet import Config, Detector, ModelConfig, TrainConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        # The default front-end's shape (HSV, one scale, six dyadic levels down to
        # 2x2, so level shifts 8 and 10 reach the exporter and the C++ binner) at a
        # quarter of its resolution: 256 px at stride 1 keeps 4 samples per cell.
        train=TrainConfig(
            levels=(64, 32, 16, 8, 4, 2),
            thumb=256,
            stride=1,
            top_k_features=0,
            val_frac=0.25,
            max_train_cells=20_000,
        ),
    )


CPP_DIR = Path(__file__).resolve().parents[1] / "cpp"


def _build_scorer(build_dir: Path) -> Path:
    """Configure and build the scorer with CMake (Highway, as in imfeat); return it.

    Highway is fetched by CMake; set ``FASTDET_HWY_DIR`` to a local Highway
    checkout to build offline.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    configure = [cmake, "-S", str(CPP_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
    if os.environ.get("FASTDET_HWY_DIR"):
        configure.append(f"-DFETCHCONTENT_SOURCE_DIR_HIGHWAY={os.environ['FASTDET_HWY_DIR']}")
    build = [
        cmake,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "fastdet_score",
        "--target",
        "fastdet_native",
    ]
    for step in (configure, build):
        result = subprocess.run(  # noqa: S603 -- argv is cmake and fixed arguments
            step, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            pytest.skip(
                f"could not build the C++ runtime:\n{result.stdout[-1500:]}{result.stderr[-1500:]}"
            )
    name = "fastdet_score.exe" if sys.platform == "win32" else "fastdet_score"
    found = sorted(build_dir.rglob(name))
    if not found:
        pytest.skip("C++ build produced no fastdet_score binary")
    return found[0]


@pytest.fixture
def scratch(tmp_path: Path) -> Iterator[Path]:
    """A working directory whose contents are removed when the test ends.

    The C++ gates write a model and a feature fixture per image (several MB each),
    and pytest keeps the last three runs' temp trees; these do not need keeping.
    """
    with tempfile.TemporaryDirectory(dir=tmp_path) as work:
        yield Path(work)


@pytest.fixture(scope="session")
def scorer() -> Path:
    """The C++ scorer binary, built once per session and reused across runs.

    The build lives in the repo's gitignored ``build/`` rather than a pytest temp
    directory: it holds the fetched Highway checkout (~26 MB), and pytest keeps the
    last three runs, so a temp build would re-fetch and rebuild Highway every
    session and keep three copies of it.
    """
    return _build_scorer(CPP_DIR.parent / "build" / "pytest-cpp")


@pytest.fixture
def fitted(
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Detector, Path]:
    """A detector fitted once on the tiny dataset, in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir)
    return det, images_dir
