"""The in-process C++ scorer (ctypes) against the NumPy runtime."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Detector

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config


@pytest.fixture
def native_lib(scorer: Path) -> Path:
    """The shared library built next to the test scorer (the CMake build makes both)."""
    build_dir = scorer.parent
    for name in (
        "libfastdet_native.so",
        "libfastdet_native.dylib",
        "fastdet_native.dll",
        "Release/fastdet_native.dll",
    ):
        path = build_dir / name
        if path.exists():
            os.environ["FASTDET_NATIVE_LIB"] = str(path)
            return path
    pytest.skip("fastdet_native shared library not built")


def test_native_scorer_matches_python_runtime(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path, native_lib: Path
) -> None:
    assert native_lib.exists()
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    loaded = Detector.load(det.export(tmp_path / "model.fdt"))
    assert loaded.native is not None, "Detector.load did not pick up the native library"
    assert loaded.runtime is not None
    for sample in sorted(images_dir.glob("*.png"))[:3]:
        design = loaded.design_matrix(sample)
        native = loaded.native_matrix(sample)
        full = loaded.runtime.predict_grid(design, use_exit=False).reshape(-1)
        got = loaded.native.score(native, use_exit=False)
        np.testing.assert_allclose(got, full, rtol=0, atol=1e-6)  # float32 sigmoid vs float64
        # with the exit: a cell the Python runtime keeps (its tile stays above every threshold)
        # is kept by the C++ too, which drops packs of tiles, so it must carry the same score
        exited_py = loaded.runtime.predict_grid(design, use_exit=True).reshape(-1)
        kept = exited_py == full
        exited_native = loaded.native.score(native, use_exit=True)
        np.testing.assert_allclose(exited_native[kept], full[kept], rtol=0, atol=1e-6)
        assert np.all(exited_native <= np.maximum(full, exited_py) + 1e-6)
        # predict_proba goes through the native scorer now
        np.testing.assert_array_equal(loaded.predict_proba(sample).reshape(-1), exited_native)


def test_native_scorer_latency(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path, native_lib: Path
) -> None:
    """End to end, in process: front-end + C++ model, one image."""
    assert native_lib.exists()
    images_dir, masks_dir = tiny_dataset
    det = Detector.load(
        Detector(small_config).fit(images_dir, masks_dir, evaluate=False).export(tmp_path / "m.fdt")
    )
    assert det.native is not None
    sample = min(images_dir.glob("*.png"))
    image = det._decode(sample)
    for _ in range(3):
        det.predict_proba(image)
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        native = det.native_matrix(image)
    t1 = time.perf_counter()
    for _ in range(reps):
        det.native.score(native)
    t2 = time.perf_counter()
    print(
        f"\n[fastdet-lat] IN-PROCESS ({det.native.target}): front-end {1e3 * (t1 - t0) / reps:.3f} ms"
        f" + C++ model {1e3 * (t2 - t1) / reps:.3f} ms per image (tiny test model: {det.runtime.n_trees} trees)"
    )
