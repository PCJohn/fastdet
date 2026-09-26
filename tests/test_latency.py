"""Latency benchmark, printed by ``pytest -s`` (imfeat prints its own the same way).

Two totals are reported separately because they scale with different things:

* **model inference** -- binning the features to 4-bit codes, then walking every tree
  for every cell.  It runs on the fixed 64x64 output grid, so its cost is set by the
  kept column count, the tree count and the depth, and is **the same for any input
  image size**.
* **front-end** -- turning an image into those features: resize, colour conversion,
  imfeat's pass, the context banks, and packing the result for the scorer.  This is
  what the input size changes.

By default the model is random -- splits, thresholds and leaves are drawn, then
written by the real blob exporter and scored by the real C++ scorer -- so the
benchmark needs no dataset and no fit, and it double-checks the scorer against the
Python runtime on the way.

**A random model is only an approximation of a trained one**, and traversal cost
depends on its shape, so two properties are calibrated against a real trained model
COARSE_MAX_SIDE = 16  # features constant inside a 4x4 tile: the coarse tier's features
(``THRESHOLDS_PER_FEATURE`` and ``LEVEL_SHARE`` below).  For the real number, point
the benchmark at an exported model instead::

    FASTDET_MODEL=model.fdt FASTDET_FIXTURE=fixture.f32 pytest -s tests/test_latency.py

where the fixture is ``Detector.native_matrix(image).tofile(...)``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from fastdet import Config
from fastdet.exporter import build_blob
from fastdet.features import GRID, FeatureExtractor, feature_level_bits
from fastdet.runtime import parse_blob

if TYPE_CHECKING:
    from numpy.typing import NDArray

# framegate's pass: HSV, 64x64 finest grid, six levels, ~4 samples per cell per axis.
FRAMEGATE_LEVELS = (64, 32, 16, 8, 4, 2)
SIZES = (256, 512, 1024)
N_TREES = 1000  # the shipped shape (ModelConfig defaults)
DEPTH = 5
KEPT_COLUMNS = (512, 1178)
# Calibration against a model trained on the real dataset (1178 columns): it held
# 6906 thresholds, i.e. ~6 per column, and binning cost is proportional to them.
THRESHOLDS_PER_FEATURE = 6
# Share of splits per feature level (side 64 .. 2, then image-wide).  Traversal cost
# depends on how many of a tree's splits vary inside a 4x4 tile (the coarser levels
# do not), so this drives the traversal number; it comes from a fitted model and is
# the least certain input here -- the printed "splits per tree that vary" line is
# what to compare against a real model's.
LEVEL_SHARE = {64: 0.38, 32: 0.20, 16: 0.14, 8: 0.13, 4: 0.07, 2: 0.05, 1: 0.04}
COARSE_MAX_SIDE = 16  # features constant inside a 4x4 tile: what the coarse tier may split on
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


def _random_model(
    n_features: int,
    seed: int = 0,
    leaf_bits: int = 8,
    coarse_fraction: float = 2.0 / 3.0,
    keep_tiles: float = 0.5,
) -> tuple[bytes, NDArray[np.float32], NDArray[np.float32]]:
    """A random model of the shipped shape: ``(blob, native fixture, dense fixture)``.

    The first ``coarse_fraction`` of the trees split only on tile-constant features
    (side <= 16), like a fitted model's coarse tier.  Random leaves cannot be
    calibrated, so the exit stage after the coarse tier is synthetic: its threshold is
    set so that about ``keep_tiles`` of the tiles stay alive on the random fixture.
    """
    rng = np.random.default_rng(seed)
    names = FeatureExtractor(_config(512).train).base_names
    step = max(1, len(names) // n_features)
    columns = [names[i * step] for i in range(n_features)]
    shifts = [feature_level_bits(name)[1] for name in columns]
    sides = [GRID >> (shift >> 1) for shift in shifts]

    by_side: dict[int, list[int]] = {}
    for index, side in enumerate(sides):
        by_side.setdefault(side, []).append(index)
    pools = [s for s in by_side if LEVEL_SHARE.get(s, 0) > 0]
    weights = np.array([LEVEL_SHARE[s] for s in pools], dtype=float)
    weights /= weights.sum()

    counts = rng.integers(2, 2 * THRESHOLDS_PER_FEATURE, n_features)  # mean ~= the real model's
    borders = [np.sort(rng.uniform(0.05, 0.95, int(c))).tolist() for c in counts]
    coarse_trees = round(N_TREES * coarse_fraction)
    coarse_pools = [s for s in pools if s <= COARSE_MAX_SIDE]
    coarse_weights = np.array([LEVEL_SHARE[s] for s in coarse_pools], dtype=float)
    coarse_weights /= coarse_weights.sum()
    trees = []
    for tree in range(N_TREES):
        splits = []
        chosen = (
            rng.choice(coarse_pools, size=DEPTH, p=coarse_weights)
            if tree < coarse_trees
            else rng.choice(pools, size=DEPTH, p=weights)
        )
        for side in chosen:
            feature = int(rng.choice(by_side[int(side)]))
            cut = borders[feature][int(rng.integers(len(borders[feature])))]
            splits.append({"float_feature_index": feature, "border": cut})
        trees.append({"splits": splits, "leaf_values": rng.normal(0.0, 0.05, 1 << DEPTH).tolist()})
    model_json = {
        "features_info": {
            "float_features": [
                {"feature_index": f, "borders": borders[f]} for f in range(n_features)
            ]
        },
        "oblivious_trees": trees,
    }
    blob, _info = build_blob(
        model_json, level_shift=shifts, leaf_bits=leaf_bits, coarse_trees=coarse_trees
    )

    native = np.concatenate(
        [rng.uniform(0.0, 1.0, side * side).astype(np.float32) for side in sides]
    )
    dense = np.empty((GRID * GRID, n_features), dtype=np.float32)
    position = 0
    for column, side in enumerate(sides):
        block = native[position : position + side * side].reshape(side, side)
        position += side * side
        factor = GRID // side
        dense[:, column] = np.repeat(np.repeat(block, factor, 0), factor, 1).ravel()
    # synthetic exit stage: keep about keep_tiles of the tiles after the coarse tier
    runtime = parse_blob(blob)
    partial = runtime.partial_scores(runtime.bins(dense), [coarse_trees])[0]
    tile_max = partial.reshape(GRID // 4, 4, GRID // 4, 4).max(axis=(1, 3)).ravel()
    theta = float(np.quantile(tile_max, 1.0 - keep_tiles))
    blob, _info = build_blob(
        model_json,
        level_shift=shifts,
        leaf_bits=leaf_bits,
        coarse_trees=coarse_trees,
        exit_stages=[(coarse_trees, theta)],
    )
    return blob, native, dense


def _run_scorer(scorer: Path, model: Path, fixture: Path, expected: Path | None) -> str:
    """Run the scorer and return its output, skipping if the host cannot run it."""
    argv = [str(scorer), str(model), str(fixture)]
    argv += [str(expected), str(_ITERS)] if expected else [str(_ITERS)]
    run = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
        pytest.skip("host CPU lacks the SIMD target the scorer was built for")
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


def test_model_inference_latency(scorer: Path, scratch: Path) -> None:
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
        _report_model("real model", _run_scorer(scorer, Path(model_path), Path(fixture), None))
        return

    for n_features, bits in [(n, 8) for n in KEPT_COLUMNS] + [(KEPT_COLUMNS[-1], 4)]:
        blob, native, dense = _random_model(n_features, leaf_bits=bits)
        expected = parse_blob(blob).predict_grid(dense).reshape(-1).astype(np.float32)
        paths = [scratch / name for name in ("m.imsy", "f.f32", "e.f32")]
        for path, payload in zip(paths, (blob, native.tobytes(), expected.tobytes()), strict=True):
            path.write_bytes(payload)
        runs = [_run_scorer(scorer, *paths) for _ in range(2)]
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
    """Resize, colour conversion, imfeat and bank assembly at one resolution and stride."""
    cfg = _config(size, stride)
    extractor = FeatureExtractor(cfg.train)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    thumb = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    level_maps, broadcast = extractor.extract(image)

    resize_ms, _ = _p50_min(
        lambda: cv2.cvtColor(
            cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV
        )
    )
    imfeat_ms, _ = _p50_min(lambda: extractor.fc.features(hsv))
    extract_ms, _ = _p50_min(lambda: extractor.extract(image))
    native_ms, _ = _p50_min(lambda: extractor.native(level_maps, broadcast))
    print(
        f"\n[fastdet-lat] FRONT-END -- {size}x{size}x3 image, HSV, {len(cfg.train.levels)} pyramid"
        f" levels, stride {cfg.train.stride}"
        f" ({(size // GRID) // cfg.train.stride} samples per cell per axis),"
        f" {extractor.total_width()} feature columns:"
    )
    print(
        f"   resize + colour convert {resize_ms:6.3f} ms | imfeat pass {imfeat_ms:6.3f} ms | "
        f"context banks + level assembly {extract_ms - resize_ms - imfeat_ms:6.3f} ms | "
        f"pack features for the scorer {native_ms:6.3f} ms"
    )
    print(
        f"   {'':<22s} front-end total (image -> features ready to score) {extract_ms + native_ms:6.3f} ms"
    )
    assert extract_ms > 0.0
