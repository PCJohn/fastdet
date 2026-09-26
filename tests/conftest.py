"""Shared fixtures.

A tiny synthetic dataset, a cheap config, one model fitted on them once per session, and
the C++ harness.  The tests import the *installed* ``fastdet`` (editable or not): that is where the compiled
scorer lives, so ``src/`` is deliberately not on ``sys.path`` (see pyproject.toml).
"""

from __future__ import annotations

import os
import shutil
import signal
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
    from collections.abc import Callable, Iterator

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


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """``N_IMAGES`` textured images each with 1-2 bright masked boxes (read-only)."""
    root = tmp_path_factory.mktemp("dataset")
    images_dir = root / "images"
    masks_dir = root / "masks"
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


def make_small_config() -> Config:
    """A cheap but structurally complete config for end-to-end tests.

    The default front-end's shape (HSV, one scale, six dyadic levels down to 2x2, so
    level shifts 8 and 10 reach the exporter and the C++ binner) at a quarter of its
    resolution: 256 px at stride 1 keeps 4 samples per cell.
    """
    return Config(
        model=ModelConfig(depth=3, n_trees=60, learning_rate=0.2, border_count=15),
        train=TrainConfig(
            levels=(64, 32, 16, 8, 4, 2),
            thumb=256,
            stride=1,
            top_k_features=0,
            val_frac=0.25,
            max_train_cells=20_000,
        ),
    )


@pytest.fixture
def small_config() -> Config:
    """A fresh copy per test: tests tweak it before fitting."""
    return make_small_config()


@pytest.fixture(scope="session")
def tiny_model(tiny_dataset: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An exported model fitted once with :func:`make_small_config` on the tiny dataset.

    For tests that only need a loaded detector (``Detector.load(tiny_model)``); tests
    that need the live fit, or that prune or otherwise mutate it, use ``fitted``.
    """
    images_dir, masks_dir = tiny_dataset
    det = Detector(make_small_config()).fit(images_dir, masks_dir, evaluate=False)
    path = det.export(tmp_path_factory.mktemp("model") / "tiny.fdt")
    det.close()
    return path


@pytest.fixture
def fitted(
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Detector, Path]:
    """A detector fitted (and evaluated) fresh for this test, in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir)
    return det, images_dir


# -- the C++ harness ---------------------------------------------------------------------

CPP_DIR = Path(__file__).resolve().parents[1] / "cpp"
# exit statuses of a process killed by an illegal instruction: POSIX signal, Windows NTSTATUS
_ILLEGAL_INSTRUCTION = {-int(getattr(signal, "SIGILL", 4)), 0xC000001D, 0xC000001D - (1 << 32)}


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
    build = [cmake, "--build", str(build_dir), "--config", "Release", "--target", "fastdet_score"]
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


@pytest.fixture(scope="session")
def run_scorer(scorer: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    """``run_scorer(*args, cwd=None)``: the built scorer on ``args``, output captured.

    Skips the test when the host cannot execute the SIMD target the scorer was built
    for; the caller asserts on the exit status and output.
    """

    def run(*args: object, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(  # noqa: S603 -- argv is the binary this session built
            [str(scorer), *map(str, args)], capture_output=True, text=True, check=False, cwd=cwd
        )
        if result.returncode in _ILLEGAL_INSTRUCTION:
            pytest.skip("host CPU lacks the SIMD target the scorer was built for")
        return result

    return run
