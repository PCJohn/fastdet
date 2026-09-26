"""The in-process C++ scorer (the nanobind extension) against the NumPy reference runtime."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Detector
from fastdet.native import NativeScorer, load_scorer
from fastdet.runtime import parse_blob
from random_model import random_model

THREAD_COUNTS = (
    2,
    3,
    5,
    16,
)  # past the vectors of tiles a pass has (4, 8 or 16): some threads own none

if TYPE_CHECKING:
    from pathlib import Path

    from fastdet import Config


def test_native_scorer_matches_python_runtime(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    images_dir, _masks_dir = tiny_dataset
    loaded = Detector.load(tiny_model)
    assert isinstance(loaded.native, NativeScorer)
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
        # predict_proba goes through the scorer
        np.testing.assert_array_equal(loaded.predict_proba(sample).reshape(-1), exited_native)
    loaded.close()


def test_fitted_and_loaded_detectors_agree(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    """The scorer a fit leaves behind is the one a reload builds from the file."""
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    loaded = Detector.load(det.export(tmp_path / "model.fdt"))
    sample = min(images_dir.glob("*.png"))
    np.testing.assert_array_equal(det.predict_proba(sample), loaded.predict_proba(sample))
    det.close()
    loaded.close()


def test_scorer_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="rejected"):
        load_scorer(b"not a model")


def _assert_threads_agree(blob: bytes, natives: list[np.ndarray]) -> None:
    single = load_scorer(blob, threads=1)
    for threads in THREAD_COUNTS:
        scorer = load_scorer(blob, threads=threads)
        assert scorer.threads == threads
        for native in natives:
            for use_exit in (False, True):
                np.testing.assert_array_equal(
                    scorer.score(native, use_exit=use_exit),
                    single.score(native, use_exit=use_exit),
                    err_msg=f"{threads} threads, use_exit={use_exit}",
                )


def test_threaded_scorer_is_bit_identical_on_a_fitted_model(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """Any thread count gives the same bytes as one thread, with and without the exit."""
    images_dir, _masks_dir = tiny_dataset
    blob = tiny_model.read_bytes()
    assert load_scorer(blob, threads=0).threads == 1  # clamped
    assert load_scorer(blob, threads=1000).threads == 16  # clamped
    det = Detector.load(tiny_model)
    natives = [det.native_matrix(sample) for sample in sorted(images_dir.glob("*.png"))[:3]]
    _assert_threads_agree(blob, natives)
    # the detector's own scorer follows its thread setting
    loaded = Detector.load(tiny_model, threads=2)
    assert loaded.native is not None
    assert loaded.native.threads == 2
    first = min(images_dir.glob("*.png"))
    np.testing.assert_array_equal(loaded.predict_proba(first), det.predict_proba(first))
    loaded.close()
    det.close()


@pytest.mark.parametrize("leaf_bits", [4, 8])
def test_threaded_scorer_is_bit_identical_through_every_exit(leaf_bits: int) -> None:
    """Same, on a model whose stages are known to drop some packs and keep others.

    The fitted model's calibrated stages may drop nothing on the tiny dataset, so this one
    has a stage after the coarse tier and one inside the fine tier that each keep about
    half of the tiles: every thread must apply both, including threads that owned no tiles
    in the coarse tier and only receive packs afterwards.
    """
    blob, native, dense = random_model(300, leaf_bits=leaf_bits, fine_stage_keep=0.5)
    runtime = parse_blob(blob)
    assert len(runtime.exit_stages) == 2
    full = runtime.predict_grid(dense, use_exit=False).reshape(-1)
    exited = runtime.predict_grid(dense, use_exit=True).reshape(-1)
    assert 0 < np.count_nonzero(exited != full) < full.size  # the stages bite, and not everywhere
    _assert_threads_agree(blob, [native])


def test_native_scorer_latency(tiny_dataset: tuple[Path, Path], tiny_model: Path) -> None:
    """End to end, in process: front-end + C++ model, one image."""
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    assert det.native is not None
    assert det.runtime is not None
    image = det._decode(min(images_dir.glob("*.png")))
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
        f"\n[fastdet-lat] IN-PROCESS ({det.native.target}, {det.native.threads} thread(s)):"
        f" front-end {1e3 * (t1 - t0) / reps:.3f} ms"
        f" + C++ model {1e3 * (t2 - t1) / reps:.3f} ms per image (tiny test model: {det.runtime.n_trees} trees)"
    )
    det.close()
