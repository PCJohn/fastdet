"""The scorer reading the level maps where they lie (``Scorer.score_maps``).

The packed matrix (``Detector.native_matrix``) copies every kept column's values into one
contiguous run per column; the map path binds the same values through a per-feature base
pointer and strides, so imfeat's cell-major maps, the context banks' planes and the
broadcast vector are read in place and no packed copy is made per frame.  Binning compares
each value with the same cuts whatever its address, so the scores must be the same bytes --
on random models and on a fitted one, with and without the exit, on every thread count --
and the layout handover must be checked, since a wrong stride would misread silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from fastdet import Detector
from fastdet.features import GRID, MapSources
from fastdet.native import load_scorer
from fastdet.runtime import parse_blob
from random_model import random_model

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

THREAD_COUNTS = (1, 2, 3, 16)


def _cell_major(
    blob: bytes, native: NDArray[np.float32], *, planar: bool = False, padded: bool = False
) -> tuple[MapSources, list[NDArray[np.float32]]]:
    """The packed fixture as a host's maps: one array per feature side.

    Each side's features become the last axis of a ``(side, side, n)`` cell-major array
    (imfeat's layout); image-wide features a 1-D vector.  With ``planar`` the array is a
    transposed view of ``(n, side, side)`` planes (the context banks' layout: a column is
    a contiguous plane); with ``padded`` the features sit at every other index of a
    twice-wider array, so the strides are nothing like the shape.
    """
    runtime = parse_blob(blob)
    sides = [GRID >> (int(shift) >> 1) for shift in runtime.level_shift]
    by_side: dict[int, list[int]] = {}
    for f, side in enumerate(sides):
        by_side.setdefault(side, []).append(f)
    offsets = np.concatenate([[0], np.cumsum([s * s for s in sides], dtype=np.int64)])
    slots: list[tuple[int, int]] = []
    slot_side: list[int] = []
    feature_slot = np.zeros(len(sides), np.int32)
    feature_index = np.zeros(len(sides), np.int32)
    arrays: list[NDArray[np.float32]] = []
    for slot, (side, features) in enumerate(sorted(by_side.items(), reverse=True)):
        n = len(features)
        step = 2 if padded else 1
        arr: NDArray[np.float32]
        if side == 1:
            arr = np.zeros(n * step, np.float32)
            for j, f in enumerate(features):
                arr[j * step] = native[offsets[f]]
        elif planar:
            planes = np.zeros((n * step, side, side), np.float32)
            for j, f in enumerate(features):
                planes[j * step] = native[offsets[f] : offsets[f + 1]].reshape(side, side)
            arr = planes.transpose(1, 2, 0)
        else:
            arr = np.zeros((side, side, n * step), np.float32)
            for j, f in enumerate(features):
                arr[..., j * step] = native[offsets[f] : offsets[f + 1]].reshape(side, side)
        for j, f in enumerate(features):
            feature_slot[f] = slot
            feature_index[f] = j * step
        slots.append((side, slot))
        slot_side.append(side)
        arrays.append(arr)
    sources = MapSources(
        slots=tuple(slots),
        slot_side=np.asarray(slot_side, np.int32),
        feature_slot=feature_slot,
        feature_index=feature_index,
    )
    return sources, arrays


@pytest.mark.parametrize("leaf_bits", [4, 8])
@pytest.mark.parametrize("layout", ["cell_major", "planar", "padded"])
def test_maps_score_like_the_packed_fixture(leaf_bits: int, layout: str) -> None:
    """Every layout, every thread count, with and without the exit: the packed bytes."""
    blob, native, _dense = random_model(300, leaf_bits=leaf_bits, fine_stage_keep=0.5)
    sources, arrays = _cell_major(
        blob, native, planar=layout == "planar", padded=layout == "padded"
    )
    if layout == "planar":  # transposed views: a feature is a contiguous plane
        assert all(not a.flags.c_contiguous for a in arrays if a.ndim == 3)
    else:  # a feature's values are one column of each cell's record: strided
        assert any(a.ndim == 3 and a.shape[2] > 1 for a in arrays)
    if layout == "padded":
        assert np.all(sources.feature_index % 2 == 0)
    reference = load_scorer(blob, threads=1)
    want = {use_exit: reference.score(native, use_exit=use_exit) for use_exit in (False, True)}
    assert not np.array_equal(want[True], want[False])  # the stages bite: the lazy path runs
    for threads in THREAD_COUNTS:
        scorer = load_scorer(blob, threads=threads)
        scorer.set_sources(sources)
        assert scorer.sources is sources
        for use_exit in (False, True):
            got = scorer.score_maps(arrays, use_exit=use_exit)
            np.testing.assert_array_equal(got, want[use_exit], err_msg=f"{threads} threads")


def test_map_sources_name_the_values_native_packs(tiny_model: Path) -> None:
    """The slot/index table reads the same (level, bank, column) as the packed matrix."""
    det = Detector.load(tiny_model)
    ex = det.extractor
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    level_maps, broadcast = ex.extract(frame)
    total = ex.total_width()
    for keep in (None, np.sort(rng.choice(total, 200, replace=False))):
        sources = ex.map_sources(keep)
        arrays = sources.arrays(level_maps, broadcast)
        n = total if keep is None else len(keep)
        assert sources.feature_slot.shape == sources.feature_index.shape == (n,)
        packed = ex.native(level_maps, broadcast, keep)
        pos = 0
        for f in range(n):
            arr = arrays[sources.feature_slot[f]]
            side = sources.slot_side[sources.feature_slot[f]]
            values = (
                arr[sources.feature_index[f]]
                if arr.ndim == 1
                else arr[..., sources.feature_index[f]]
            )
            np.testing.assert_array_equal(np.ravel(values), packed[pos : pos + side * side])
            pos += side * side
        assert pos == packed.shape[0]
    det.close()


def test_detector_scores_the_maps_in_place(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """``predict_proba`` reads the maps; packing them first gives the same bytes."""
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model, threads=2)
    assert det.native is not None
    before = det.native.sources
    assert before is None  # set on first use
    for sample in sorted(images_dir.glob("*.png"))[:3]:
        frame = np.asarray(cv2.imread(str(sample)), dtype=np.uint8)
        level_maps, broadcast = det.extractor.extract(frame)
        packed = det.native.score(
            det.pack_native(level_maps, broadcast), use_exit=det.config.model.use_exit
        )
        np.testing.assert_array_equal(det.score_maps(level_maps, broadcast), packed)
        np.testing.assert_array_equal(det.predict_proba(frame).reshape(-1), packed)
        sources = det.native.sources
        assert sources is not None  # one layout, set on first use
        np.testing.assert_array_equal(
            det.native.score_maps(sources.arrays(level_maps, broadcast), use_exit=False),
            det.native.score(det.pack_native(level_maps, broadcast), use_exit=False),
        )
    assert det.native.sources is not None
    assert det.native.sources.slot_side[0] == GRID
    det.close()


def test_score_maps_checks_its_arrays() -> None:
    """A wrong count, dtype, rank, shape or width is refused, never misread."""
    blob, native, _dense = random_model(60, leaf_bits=4)
    sources, arrays = _cell_major(blob, native)
    scorer = load_scorer(blob, threads=1)
    with pytest.raises(RuntimeError, match="set_sources"):
        scorer.score_maps(arrays)
    scorer.set_sources(sources)
    want = scorer.score(native)
    np.testing.assert_array_equal(scorer.score_maps(arrays), want)
    with pytest.raises(ValueError, match="expected"):
        scorer.score_maps(arrays[:-1])
    finest = next(i for i, (side, _bank) in enumerate(sources.slots) if side == GRID)
    bad_arrays: list[tuple[Any, str]] = [
        (arrays[finest].astype(np.float64), "float32"),
        (arrays[finest][0], "3-D"),
        (arrays[finest][:, :-1], "must be"),
        (arrays[finest][..., :-1], "reads index"),
    ]
    for bad, why in bad_arrays:
        wrong: list[Any] = list(arrays)
        wrong[finest] = bad
        with pytest.raises(ValueError, match=why):
            scorer.score_maps(wrong)
    vector = next(i for i, (side, _bank) in enumerate(sources.slots) if side == 1)
    wrong = list(arrays)
    wrong[vector] = arrays[vector][:-1]  # too short
    with pytest.raises(ValueError, match="values"):
        scorer.score_maps(wrong)
    # a feature's side must be its slot's
    bad_side = sources.slot_side.copy()
    bad_side[finest] = GRID // 2
    with pytest.raises(ValueError, match="side"):
        scorer.set_sources(
            MapSources(sources.slots, bad_side, sources.feature_slot, sources.feature_index)
        )
    assert scorer.sources is sources  # a refused layout leaves the old one in place
    np.testing.assert_array_equal(scorer.score_maps(arrays), want)


def test_concurrent_callers_do_not_disturb_each_other() -> None:
    """Two threads scoring different maps on one scorer: each gets its own bytes.

    The GIL is released during a pass and the scorer serialises passes; the sources a
    call builds must stay its own while another call waits its turn.
    """
    import threading  # noqa: PLC0415 -- test-local

    blob, native, _dense = random_model(120, leaf_bits=4, fine_stage_keep=0.5)
    rng = np.random.default_rng(3)
    other = (native * rng.uniform(0.5, 1.5, native.shape)).astype(np.float32)
    sources, arrays = _cell_major(blob, native)
    _sources2, arrays2 = _cell_major(blob, other)
    scorer = load_scorer(blob, threads=2)
    scorer.set_sources(sources)
    want = [scorer.score(x) for x in (native, other)]
    assert not np.array_equal(want[0], want[1])
    got: dict[int, list[NDArray[np.float32]]] = {0: [], 1: []}

    def worker(i: int, maps: list[NDArray[np.float32]]) -> None:
        for _ in range(40):
            got[i].append(scorer.score_maps(maps))

    threads = [
        threading.Thread(target=worker, args=(i, m)) for i, m in enumerate((arrays, arrays2))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in (0, 1):
        for out in got[i]:
            np.testing.assert_array_equal(out, want[i])
