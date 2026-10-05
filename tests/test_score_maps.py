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
    n_raw = 1 + len(ex.extra_scales)
    for keep in (None, np.sort(rng.choice(total, 200, replace=False))):
        for in_scorer in (False, True):
            sources = ex.map_sources(keep, banks_in_scorer=in_scorer)
            arrays = sources.arrays(level_maps, broadcast)
            n = total if keep is None else len(keep)
            assert sources.feature_slot.shape == sources.feature_index.shape == (n,)
            assert bool(sources.bank_slots) == in_scorer
            computed = {slot: raw for slot, raw, _col in sources.bank_slots}
            packed = ex.native(level_maps, broadcast, keep)
            pos = 0
            for f in range(n):
                slot = int(sources.feature_slot[f])
                side = int(sources.slot_side[slot])
                index = int(sources.feature_index[f])
                if slot in computed:  # a plane of the level's bank buffer: compose's, here
                    size = sources.slots[slot][0]
                    assert arrays[slot] is None
                    assert sources.slots[computed[slot]] == (size, 0)  # its raw map's slot
                    planes = level_maps[size][n_raw].base  # the (7, side, side) buffer
                    assert planes is not None
                    values = planes[index]
                else:
                    arr = arrays[slot]
                    assert arr is not None
                    values = arr[index] if arr.ndim == 1 else arr[..., index]
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


def test_raw_path_computes_the_banks_in_the_scorer(
    tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """``score_raw`` (no compose) == ``score_maps(compose)`` == the packed matrix scored.

    The scorer makes the context banks itself, on the calling thread while the others bin,
    and bins them from its own buffers: the same bytes on every thread count, with and
    without the exit, and the buffers hold what compose computes.
    """
    images_dir, _masks_dir = tiny_dataset
    frames = [
        np.asarray(cv2.imread(str(p)), dtype=np.uint8) for p in sorted(images_dir.glob("*.png"))[:3]
    ]
    reference = Detector.load(tiny_model, threads=1)
    assert reference.native is not None
    want = []
    for frame in frames:
        level_maps, broadcast = reference.extractor.extract(frame)
        packed = reference.pack_native(level_maps, broadcast)
        want.append(
            {
                use_exit: reference.native.score(packed, use_exit=use_exit)
                for use_exit in (False, True)
            }
        )
    for threads in (1, 2, 3, 16):
        det = Detector.load(tiny_model, threads=threads)
        assert det.native is not None
        for frame, expect in zip(frames, want, strict=True):
            result, extra = det.extractor.run_imfeat(frame)
            np.testing.assert_array_equal(
                det.score_raw(result, frame.shape[:2], extra), expect[det.config.model.use_exit]
            )
            sources = det.native.sources
            assert sources is not None
            assert sources.bank_slots  # the banks are the scorer's
            raw = det.extractor.raw_maps(result, frame.shape[:2], extra)
            for use_exit in (False, True):
                got = det.native.score_maps(sources.raw_arrays(raw), use_exit=use_exit)
                np.testing.assert_array_equal(got, expect[use_exit], err_msg=f"{threads} threads")
            # the scorer's buffers hold compose's banks, coordinates included
            level_maps, broadcast = det.extractor.compose(result, frame.shape[:2], extra)
            n_raw = 1 + len(det.extractor.extra_scales)
            for slot, _raw_slot, _column in sources.bank_slots:
                size = sources.slots[slot][0]
                np.testing.assert_array_equal(
                    det.native.bank_buffers[slot], level_maps[size][n_raw].base
                )
            # and the composed maps score the same through the same scorer
            np.testing.assert_array_equal(
                det.score_maps(level_maps, broadcast), expect[det.config.model.use_exit]
            )
            np.testing.assert_array_equal(
                det.predict_proba(frame).reshape(-1), expect[det.config.model.use_exit]
            )
        det.close()
    reference.close()


def test_bank_slots_are_checked(tiny_model: Path) -> None:
    det = Detector.load(tiny_model)
    scorer, sources = det.scorer_layout()
    assert sources.bank_slots
    slot, raw_slot, column = sources.bank_slots[0]
    side = int(sources.slot_side[slot])
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    level_maps, broadcast = det.extractor.extract(frame)
    arrays = sources.arrays(level_maps, broadcast)
    assert arrays[slot] is None
    with_array: list[Any] = list(arrays)
    with_array[slot] = np.zeros((side, side, 7), np.float32)
    with pytest.raises(ValueError, match="pass None"):
        scorer.score_maps(with_array)
    native = scorer._scorer
    good = np.zeros((7, side, side), np.float32)
    with pytest.raises(ValueError, match="distinct"):
        native.set_bank_slot(slot, slot, column, good)
    other = next(i for i, s in enumerate(sources.slot_side) if s not in (side, 1))
    with pytest.raises(ValueError, match="same side"):
        native.set_bank_slot(slot, other, column, good)
    with pytest.raises(ValueError, match=r"\(7, side, side\)"):
        native.set_bank_slot(slot, raw_slot, column, np.zeros((7, side + 1, side), np.float32))
    with pytest.raises(TypeError):  # no implicit conversion of the buffer the scorer keeps
        native.set_bank_slot(slot, raw_slot, column, np.zeros((7, side, side), np.float64))
    # the raw map must be wide enough for the column the banks read
    narrow: list[Any] = list(arrays)
    raw = arrays[raw_slot]
    assert raw is not None
    narrow[raw_slot] = raw[..., :column]
    with pytest.raises(ValueError, match="reads index"):
        scorer.score_maps(narrow)
    np.testing.assert_array_equal(  # intact after the refusals
        scorer.score_maps(arrays), det.score_maps(level_maps, broadcast)
    )
    det.close()
