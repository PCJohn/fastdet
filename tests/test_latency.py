"""Latency benchmark, printed by ``pytest -s`` (imfeat prints its own the same way).

Two numbers, kept apart because they scale with different things:

* **model inference** (binning + traversal) runs on the fixed 64x64 output grid, so
  its cost depends on the kept column count, the tree count and the depth -- **not**
  on the input image size;
* **front-end** (resize, colour conversion, imfeat, banks, native buffer) is what
  the image size changes.

The model is random: splits, borders and leaves are drawn, then exported through the
real blob writer and scored by the real C++ scorer, so no dataset or fit is needed.
Splits are drawn over pyramid levels in the proportions a fitted model showed (38%
of splits on the finest level, 20% one level up, ...), because traversal cost depends
on how many of a tree's splits vary inside a 4x4 tile.  Those proportions come from a
model fitted on synthetic images, so compare the "varying splits per tree" line this
test prints against the one the scorer prints for a real model and adjust
``LEVEL_SHARE`` if they differ.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from fastdet import Config
from fastdet.exporter import build_blob
from fastdet.features import GRID, FeatureExtractor, feature_level_bits
from fastdet.runtime import parse_blob

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# framegate's pass: HSV, 64x64 finest grid, six levels, ~4 samples per cell per axis.
FRAMEGATE_LEVELS = (64, 32, 16, 8, 4, 2)
SIZES = (256, 512, 1024)
N_TREES = 2400
DEPTH = 7
KEPT_COLUMNS = (512, 1178)
# Share of splits per feature level (side 64 .. 2, then image-wide), from a fitted
# model; see the module docstring before trusting it for a different model.
LEVEL_SHARE = {64: 0.38, 32: 0.20, 16: 0.14, 8: 0.13, 4: 0.07, 2: 0.05, 1: 0.04}
_ITERS = 10
_REPS = 15


def _config(size: int) -> Config:
    """Framegate's front-end at working resolution ``size``."""
    cfg = Config()
    cfg.train.thumb = size
    cfg.train.stride = max(1, size // 256)  # keeps framegate's 4 samples per cell
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
    n_features: int, seed: int = 0
) -> tuple[bytes, NDArray[np.float32], NDArray[np.float32]]:
    """A random model of the shipped shape: ``(blob, native fixture, dense fixture)``."""
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

    borders = [np.sort(rng.uniform(0.05, 0.95, 15)).tolist() for _ in range(n_features)]
    trees = []
    for _ in range(N_TREES):
        splits = []
        for side in rng.choice(pools, size=DEPTH, p=weights):
            feature = int(rng.choice(by_side[int(side)]))
            splits.append(
                {"float_feature_index": feature, "border": borders[feature][int(rng.integers(15))]}
            )
        trees.append({"splits": splits, "leaf_values": rng.normal(0.0, 0.05, 1 << DEPTH).tolist()})
    model_json = {
        "features_info": {
            "float_features": [
                {"feature_index": f, "borders": borders[f]} for f in range(n_features)
            ]
        },
        "oblivious_trees": trees,
    }
    blob, _info = build_blob(model_json, level_shift=shifts)

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
    return blob, native, dense


def test_model_inference_latency(scorer: Path, tmp_path: Path) -> None:
    """Binning + traversal on the 64x64 grid, for a random model of the shipped shape."""
    print(f"\n[fastdet-lat] model inference: {N_TREES} trees x depth {DEPTH}, {GRID}x{GRID} cells,")
    print("              one thread; independent of the input image size")
    for n_features in KEPT_COLUMNS:
        blob, native, dense = _random_model(n_features)
        model_path, fixture_path, expected_path = (
            tmp_path / "m.imsy",
            tmp_path / "f.f32",
            tmp_path / "e.f32",
        )
        model_path.write_bytes(blob)
        fixture_path.write_bytes(native.tobytes())
        expected = parse_blob(blob).predict_grid(dense).reshape(-1).astype(np.float32)
        expected_path.write_bytes(expected.tobytes())

        run = subprocess.run(  # noqa: S603 -- argv is the binary the session fixture built
            [str(scorer), str(model_path), str(fixture_path), str(expected_path), str(_ITERS)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
            pytest.skip("host CPU lacks the SIMD target the scorer was built for")
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        assert "0 mismatches PASS" in run.stdout
        assert "BIT-IDENTICAL" in run.stdout

        def field(pattern: str, text: str = run.stdout) -> float:
            match = re.search(pattern + r"\s*:?\s*([\d.]+) ms", text)
            assert match, f"{pattern} not in:\n{text}"
            return float(match.group(1))

        target = re.search(r"simd target: (\S+)", run.stdout)
        varying = re.search(r"trees by varying splits: (.+)", run.stdout)
        print(
            f"   {n_features:5d} columns: binner {field('tile binner'):6.3f} ms | "
            f"traversal {field('simd traversal'):6.3f} ms | "
            f"total {field('model total'):6.3f} ms  [{target.group(1) if target else '?'}]"
        )
        if varying:
            print(f"              varying splits per tree: {varying.group(1).strip()}")


@pytest.mark.parametrize("size", SIZES)
def test_front_end_latency(size: int) -> None:
    """Resize, colour conversion, imfeat and bank assembly at one working resolution."""
    cfg = _config(size)
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
        f"\n[fastdet-lat] front-end {size}x{size}x3, HSV, {len(cfg.train.levels)} levels, "
        f"stride {cfg.train.stride}, {extractor.total_width()} columns:"
    )
    print(
        f"   resize+cvtColor {resize_ms:6.3f} ms | imfeat {imfeat_ms:6.3f} ms | "
        f"banks {extract_ms - resize_ms - imfeat_ms:6.3f} ms | native {native_ms:6.3f} ms | "
        f"extract+native {extract_ms + native_ms:6.3f} ms"
    )
    assert extract_ms > 0.0
