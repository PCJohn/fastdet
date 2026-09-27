"""Latency benchmark, printed by ``pytest -s`` (imfeat prints its own the same way).

Two totals are reported separately because they scale with different things:

* **model inference** -- binning the features to 4-bit codes, then walking every tree
  for every cell.  It runs on the fixed 64x64 output grid, so its cost is set by the
  kept column count, the tree count and the depth, and is **the same for any input
  image size**.
* **front-end** -- turning an image into those features: resize, colour conversion,
  imfeat's pass, the context banks, and packing the result for the scorer.  This is
  what the input size changes.

By default the model is random (``random_model.py``: splits, thresholds and leaves are
drawn, then written by the real blob exporter and scored by the real C++ scorer), so
the benchmark needs no dataset and no fit, and it double-checks the scorer against the
Python runtime on the way.  A random model is only an approximation of a trained one;
for the real number, point the benchmark at an exported model instead::

    FASTDET_MODEL=model.fdt FASTDET_FIXTURE=fixture.f32 pytest -s tests/test_latency.py

where the fixture is ``Detector.native_matrix(image).tofile(...)``.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from fastdet import Config
from fastdet.features import GRID, FeatureExtractor
from fastdet.runtime import parse_blob
from random_model import DEPTH, N_TREES, random_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from subprocess import CompletedProcess

    RunScorer = Callable[..., CompletedProcess[str]]

# framegate's pass: HSV, 64x64 finest grid, six levels, ~4 samples per cell per axis.
FRAMEGATE_LEVELS = (64, 32, 16, 8, 4, 2)
SIZES = (256, 512, 1024)
SOURCE_HW = (1080, 1920)  # the frame the front-end thumbnails: 1080p, the target source
KEPT_COLUMNS = (512, 1178)
_ITERS = 10
_REPS = 15


def _config(size: int, stride: int | None = None) -> Config:
    """Framegate's front-end at working resolution ``size``.

    ``stride`` defaults to framegate's ratio of 4 samples per cell per axis.
    """
    cfg = Config()
    cfg.train.thumb = size
    cfg.train.stride = stride if stride is not None else max(1, size // 256)
    cfg.train.levels = FRAMEGATE_LEVELS
    cfg.train.extra_scales = ""
    cfg.train.imfeat_space = "hsv"
    return cfg


def _p50_min(fn: Any, reps: int = _REPS) -> tuple[float, float]:
    """Median and minimum wall time of ``fn`` in milliseconds."""
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return times[len(times) // 2], times[0]


def _timings(run_scorer: RunScorer, model: Path, fixture: Path, expected: Path | None) -> str:
    """The scorer's report for one model (its gates must pass)."""
    run = run_scorer(model, fixture, expected or "-", _ITERS)
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    return run.stdout


def _ms(label: str, text: str) -> float:
    """The milliseconds the scorer printed for one stage."""
    match = re.search(re.escape(label) + r"\s*:?\s*([\d.]+) ms", text)
    assert match, f"{label!r} not in:\n{text}"
    return float(match.group(1))


def _report_model(name: str, out: str) -> None:
    """Print one model-inference row, with the tree shape it was measured on."""
    target = re.search(r"simd target: (\S+?),", out)
    varying = re.search(r"trees by varying splits: (.+)", out)
    shipped = re.search(r"model, as shipped:\s*([\d.]+) ms", out)
    threaded = re.search(r"model, as shipped, (\d+) threads:\s*([\d.]+) ms", out)
    print(
        f"   {name:<24s} bin features {_ms('tile binner', out):6.3f} ms"
        f" + walk trees {_ms('simd traversal', out):6.3f} ms"
        f" = {_ms('model total', out):6.3f} ms   [{target.group(1) if target else '?'}]"
        + (
            f"\n{'':27s} as shipped (coarse tier per tile, exit, lazy binning): {float(shipped.group(1)):6.3f} ms"
            if shipped
            else ""
        )
        + (
            f"\n{'':27s} as shipped, on {threaded.group(1)} threads: {float(threaded.group(2)):6.3f} ms"
            if threaded
            else ""
        )
    )
    if varying:
        print(
            f"   {'':<24s} splits per tree that vary inside a 4x4 cell tile: {varying.group(1).strip()}"
        )


def test_model_inference_latency(run_scorer: RunScorer, scratch: Path) -> None:
    """Binning + tree traversal for the whole 64x64 grid, on one thread and on the default count."""
    print(
        f"\n[fastdet-lat] MODEL INFERENCE -- {N_TREES} trees x depth {DEPTH}, "
        f"{GRID * GRID} cells, low-bit leaf codes, one thread (and the default thread count)."
    )
    print("              'bin features' = float features -> 4-bit bins, once per")
    print("              distinct value; 'walk trees' = every tree on every cell.")
    print("              Independent of the input image size (fixed output grid).")

    model_path = os.environ.get("FASTDET_MODEL")
    if model_path:  # a real exported model: the authoritative number
        fixture = os.environ.get("FASTDET_FIXTURE")
        if not fixture:
            pytest.skip("FASTDET_MODEL needs FASTDET_FIXTURE (Detector.native_matrix output)")
        _report_model("real model", _timings(run_scorer, Path(model_path), Path(fixture), None))
        return

    for n_features, bits in [(n, 8) for n in KEPT_COLUMNS] + [(KEPT_COLUMNS[-1], 4)]:
        blob, native, dense = random_model(n_features, leaf_bits=bits)
        expected = parse_blob(blob).predict_grid(dense).reshape(-1).astype(np.float32)
        paths = [scratch / name for name in ("m.imsy", "f.f32", "e.f32")]
        for path, payload in zip(paths, (blob, native.tobytes(), expected.tobytes()), strict=True):
            path.write_bytes(payload)
        runs = [_timings(run_scorer, *paths) for _ in range(2)]
        for out in runs:
            assert "0 mismatches PASS" in out  # binner agrees with the reference binner
            assert "DIVERGED" not in out  # SIMD vs scalar, and threads vs one thread: bit-identical
        out = min(runs, key=lambda text: _ms("model total", text))  # the less disturbed run
        label = f"random, {n_features} columns, {bits}-bit leaves"
        _report_model(label, out)
    print("   'as shipped' uses a synthetic exit stage that keeps half of the tiles after the")
    print("   coarse tier (random leaves cannot be calibrated); a fitted model's stages are")
    print("   calibrated on its training images and its share of alive tiles depends on the frame.")
    print(
        "   The threaded row is the same pass split by tiles (bit-identical), default thread count."
    )


def _strides_for(size: int) -> tuple[int, ...]:
    """Strides worth measuring: 1 up to the finest cell's width.

    A 64x64 grid makes cells ``size / GRID`` px wide, so a stride equal to the cell
    width leaves one sample per cell per axis -- imfeat's floor.
    """
    cell = size // GRID
    return tuple(stride for stride in (1, 2, 4, 8, 16) if stride <= cell)


@pytest.mark.parametrize(
    ("size", "stride"), [(size, stride) for size in SIZES for stride in _strides_for(size)]
)
def test_front_end_latency(size: int, stride: int) -> None:
    """Resize, imfeat and bank assembly at one thumbnail size and stride, from a 1080p frame.

    The frame goes into imfeat whole and is thumbnailed inside its pass (``INTER_AREA``,
    cv2.resize's bytes), so the pass is timed on the frame and on a ready thumbnail: the
    difference is what the resize costs inside it.  cv2.resize is timed alongside, on
    cv2's default thread count and on one, as what a host that resizes itself would pay;
    the colour conversion is inside the pass too, cvtColor likewise for reference only.
    """
    cfg = _config(size, stride)
    extractor = FeatureExtractor(cfg.train)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (*SOURCE_HW, 3), dtype=np.uint8)
    thumb = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    level_maps, broadcast = extractor.extract(image)
    assert extractor.fuses_resize(image)

    resize = lambda: cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)  # noqa: E731
    cv2_threads = cv2.getNumThreads()
    resize_ms, _ = _p50_min(resize)
    cv2.setNumThreads(1)
    try:
        resize_1_ms, _ = _p50_min(resize)
    finally:
        cv2.setNumThreads(cv2_threads)
    cvt_ms, _ = _p50_min(lambda: cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV))
    on_thumb = extractor.thumb_computer((size, size))
    on_thumb_ms, _ = _p50_min(lambda: on_thumb.features(thumb))
    fused_ms, _ = _p50_min(lambda: extractor.run_imfeat(image))
    extract_ms, _ = _p50_min(lambda: extractor.extract(image))
    native_ms, _ = _p50_min(lambda: extractor.native(level_maps, broadcast))
    buf = np.empty(extractor.native_size(), np.float32)  # what Detector.pack_native keeps
    reused_ms, _ = _p50_min(lambda: extractor.native(level_maps, broadcast, out=buf))
    print(
        f"\n[fastdet-lat] FRONT-END -- {SOURCE_HW[1]}x{SOURCE_HW[0]} frame -> {size}x{size}x3"
        f" thumbnail, HSV, {len(cfg.train.levels)} pyramid levels, stride {cfg.train.stride}"
        f" ({(size // GRID) // cfg.train.stride} samples per cell per axis),"
        f" {extractor.total_width()} feature columns, {on_thumb.threads} imfeat thread(s):"
    )
    print(
        f"   imfeat pass on the frame, resize + BGR->HSV inside {fused_ms:6.3f} ms"
        f" (on a ready thumbnail {on_thumb_ms:6.3f}: the resize inside costs"
        f" {fused_ms - on_thumb_ms:6.3f}; cv2.resize would be {resize_ms:6.3f} on cv2's"
        f" {cv2_threads} thread(s), {resize_1_ms:6.3f} on one; a separate cvtColor {cvt_ms:6.3f})"
    )
    print(
        f"   context banks + level assembly {extract_ms - fused_ms:6.3f} ms | "
        f"pack features for the scorer {native_ms:6.3f} ms into a fresh array,"
        f" {reused_ms:6.3f} into the reused buffer ({buf.nbytes / 1e6:.1f} MB)"
    )
    print(
        f"   {'':<22s} front-end total (frame -> features ready to score)"
        f" {extract_ms + reused_ms:6.3f} ms"
    )
    assert extract_ms > 0.0
