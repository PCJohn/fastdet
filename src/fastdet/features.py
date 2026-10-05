"""Multi-level image features and per-cell design-matrix assembly.

The front-end turns one BGR image into a small pyramid of ``size x size``
per-cell feature maps (size is a power of two from 64 down to 8) plus a
per-image scene-statistics vector that is broadcast to every cell.  A cell is
one square block of the 64x64 output grid, so the coarser maps are aggregated
views of the same scene and :meth:`FeatureExtractor.gather` can assemble a
single row per cell from every level.

Banks, in the order :meth:`FeatureExtractor.gather` emits them per level:

``raw``
    imfeat's per-channel raw block per cell, once per scale.
``global``
    imfeat's whole-image raw block plus the original frame's aspect ratio and
    log area, identical for every cell (broadcast, carried at the finest level
    only since it does not vary by level).
``context``
    small-scale surround/ring/range of imfeat's per-cell luminance mean.
``ctx2``
    the large-scale companion of ``context`` (9x9 surround, 5x5 range).

fastdet computes no image features of its own: everything per-pixel comes from
imfeat in a single pass, and the banks here are cheap reductions of its output.
The bar detector ("bard") used to live in this module in numpy; it now arrives
inside imfeat's raw block, seven columns per channel. The thumbnail too is made
inside that pass when the frame allows it (see :meth:`FeatureExtractor.run_imfeat`),
and its size can follow the frame: ``TrainConfig.thumb`` is a side in pixels or an
imfeat policy name (see :func:`thumb_hw`; the thumbnail need not be square).
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import imfeat
import numpy as np
from numpy.typing import NDArray

from .config import TrainConfig, iter_feature_mode_tags

if TYPE_CHECKING:
    from collections.abc import Sequence

# The front-end hands imfeat the BGR thumbnail and lets it convert to HSV inside its pass
# (FeatureComputer's input_space / feature_space, which came with imfeat.COLOR_SPACES). An
# imfeat without that would take the thumbnail as HSV and every feature would be wrong, so
# it is refused here rather than found in the maps.
if not hasattr(imfeat, "COLOR_SPACES"):
    _msg = (
        "fastdet needs an imfeat that converts BGR to HSV inside its pass "
        "(pip install git+https://github.com/PCJohn/imfeat)"
    )
    raise ImportError(_msg)

__all__ = [
    "CONTEXT2_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "FEATURE_CHANNELS",
    "GRID",
    "IMAGE_STAT_NAMES",
    "RAW_FEATURE_NAMES",
    "SPACE_INFO",
    "STRIDE",
    "THUMB",
    "FeatureCache",
    "FeatureExtractor",
    "MapSources",
    "context_banks",
    "exponent_for_size",
    "feature_level_bits",
    "parse_extra_scales",
    "parse_level_list",
    "stride_for",
    "thumb_hw",
]

# Default thread count of imfeat and of the scorer: for imfeat, 2 gave most of the gain and 8
# measured no better than 4; the scorer's pass is ~10x shorter, so its gain flattens sooner.
_MAX_DEFAULT_THREADS = 4
Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]

GRID = 64  # Finest/output grid resolution; fixed regardless of the feature levels.
RAW_CHANNELS = 3  # imfeat is always fed a 3-channel image.
_COLOUR_NDIM = 3  # (H, W, C): what a colour frame looks like
THUMB: int | str = "pow2-fit"  # Default thumbnail: imfeat's policy, sized from the frame.
STRIDE = 1  # Default imfeat sampling stride at the primary scale.

_EPS = 1e-9  # Guard for the pooled ratios below (never a real denominator).
_SAMPLES_PER_CELL = 4  # imfeat samples four points per cell; see parse_extra_scales.
_CTX1_SIZE = 3  # Small-scale context kernel side.
_CTX2_SIZE = 5  # Large-scale range kernel side.
_CTX2_SURROUND = 9  # Large-scale surround kernel side.
_CTX1_MIN_GRID = 3  # Below this the 3x3 context kernel is undefined.
_CTX2_MIN_GRID = 5  # Below this the 5x5 range kernel is undefined.


@dataclass(frozen=True)
class SpaceInfo:
    """Color-space metadata the front-end needs to drive imfeat and the banks.

    ``fused`` names the space when imfeat converts the BGR thumbnail to it inside its
    pass (``FeatureComputer(feature_space=fused)``: bit for bit ``cv2.cvtColor``'s
    bytes, at a fraction of its cost and with no second image); otherwise ``convert``
    is the OpenCV conversion the extractor runs first, and imfeat takes the channels as
    they are.
    """

    convert: int | None
    letters: tuple[str, str, str]
    lum: int
    chr: int
    fused: str | None = None


# Color spaces the front-end can compute features in.  The raw block layout is
# channel-agnostic; the global scalars and the bar/context banks assume specific
# channel roles, hence `lum` (luminance-like) and `chr` (chroma used by the
# S_mean_global proxy).
SPACE_INFO: dict[str, SpaceInfo] = {
    "hsv": SpaceInfo(None, ("H", "S", "V"), lum=2, chr=1, fused="hsv"),
    "lab": SpaceInfo(cv2.COLOR_BGR2LAB, ("L", "a", "b"), lum=0, chr=1),
    "luv": SpaceInfo(cv2.COLOR_BGR2LUV, ("L", "u", "v"), lum=0, chr=1),
    "yuv": SpaceInfo(cv2.COLOR_BGR2YUV, ("Y", "U", "V"), lum=0, chr=1),
}
CHANNEL_LETTERS = ("H", "S", "V")

# One channel's slice of an imfeat raw block, taken from the library rather than
# copied: imfeat's per-channel width is a compile-time property of its build (it
# grew from 38 to 45 when the bar detector landed), and a stale copy here silently
# mislabels every column.
RAW_FEATURE_NAMES: tuple[str, ...] = tuple(imfeat.FEATURE_NAMES)
RAW_PER_CHANNEL = len(RAW_FEATURE_NAMES)
FEATURE_CHANNELS = RAW_CHANNELS * RAW_PER_CHANNEL  # one scale's raw block width

# Column offsets inside a channel's slice, looked up by name for the same reason.
_MEAN_IDX = RAW_FEATURE_NAMES.index("mean")
_VAR_IDX = RAW_FEATURE_NAMES.index("var")
_COHERENCE_IDX = RAW_FEATURE_NAMES.index("coherence")

# Whole-image scalars broadcast to every cell.  imfeat already computes a global
# (1-cell) level of the full raw block in the same pass, so the bank is that block
# plus the two things imfeat cannot know: the ORIGINAL frame's shape.
IMAGE_STAT_NAMES = ("aspect_ratio", "log_orig_area")


def global_stat_names(space: SpaceInfo) -> tuple[str, ...]:
    """Column names of the broadcast global bank for ``space``."""
    return (
        tuple(f"{letter}_{feature}" for letter in space.letters for feature in RAW_FEATURE_NAMES)
        + IMAGE_STAT_NAMES
    )


CONTEXT_FEATURE_NAMES = ("ctx_x", "ctx_y", "ctx_surr3", "ctx_ring35", "ctx_range3")
CONTEXT2_FEATURE_NAMES = ("ctx_surr9", "ctx_range5")

# Banks whose value is identical for every cell of a level: stored once per
# image per level and tiled at gather time.
BROADCAST_TAGS = frozenset({"global"})

Range = tuple[int, int]
RangeTable = dict[int, dict[str, Range]]


@dataclass(frozen=True)
class ExtraScale:
    """One extra imfeat scale (larger thumbnail and/or coarser stride)."""

    thumb: int
    stride: int
    label: str


# One pyramid level: imfeat's raw map per scale, as imfeat returns it, then one
# array per context bank.  Laid end to end these are the level's canonical columns.
LevelBanks = tuple["FloatArray", ...]


@dataclass(frozen=True)
class MapSources:
    """Where the scorer reads each kept column, straight from the level maps.

    The scorer's map-reading path (``Scorer.score_maps``) takes one array per *slot*: a
    level's bank -- imfeat's ``(side, side, n)`` map or a context bank, any strides -- or
    the finest level's broadcast vector.  Column ``f`` of the model is index
    ``feature_index[f]`` along the last axis of the array in slot ``feature_slot[f]``; a
    slot's ``slot_side`` is its grid side (1 for the broadcast vector).  Built once per
    kept-column set by :meth:`FeatureExtractor.map_sources`; :meth:`arrays` picks a frame's
    arrays in slot order.
    """

    slots: tuple[tuple[int, int], ...]  # per slot: (level size, bank index; -1 = broadcast)
    slot_side: NDArray[np.int32]
    feature_slot: NDArray[np.int32]
    feature_index: NDArray[np.int32]

    def arrays(
        self, level_maps: dict[int, LevelBanks], broadcast_vecs: dict[int, FloatArray]
    ) -> list[FloatArray]:
        """The frame's arrays in slot order: what ``score_maps`` is given."""
        return [
            broadcast_vecs[size] if bank < 0 else level_maps[size][bank]
            for size, bank in self.slots
        ]


@dataclass(frozen=True)
class _Copy:
    """One contiguous run of output columns, copied from one bank of one level.

    ``bank`` indexes the level's :data:`LevelBanks` (ignored for broadcast runs);
    ``src`` indexes that bank's last axis (a slice when the kept columns are
    contiguous, which makes the source a view); the run lands in ``[dst0, dst1)``.
    """

    size: int
    broadcast: bool
    bank: int
    src: slice | NDArray[np.intp]
    dst0: int
    dst1: int


def exponent_for_size(size: int) -> int:
    """Return log2(size), requiring ``size`` to be a power of two."""
    exponent = round(math.log2(size))
    if 2**exponent != size:
        msg = f"level size {size} is not a power of two"
        raise ValueError(msg)
    return exponent


def parse_level_list(spec: str, grid: int = GRID) -> tuple[int, ...]:
    """Parse ``'64,32,16,8'`` into a descending tuple of valid power-of-two levels."""
    out: list[int] = []
    for raw_token in str(spec).split(","):
        token = raw_token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1 or value > grid or (value & (value - 1)) != 0 or grid % value != 0:
            msg = f"invalid level size {value!r}; must divide {grid} and be 2**k"
            raise ValueError(msg)
        out.append(value)
    if not out:
        msg = f"could not parse any levels from {spec!r}"
        raise ValueError(msg)
    return tuple(sorted(set(out), reverse=True))


def parse_extra_scales(spec: str) -> list[ExtraScale]:
    """``'512:2;256:1'`` -> ``[ExtraScale, ...]``.

    An omitted stride follows imfeat's rule (four samples per cell):
    ``stride = thumb // (GRID * 4)``.
    """
    out: list[ExtraScale] = []
    for raw_token in (spec or "").split(";"):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            thumb_text, stride_text = token.split(":", 1)
            stride = int(stride_text)
        else:
            thumb_text, stride = token, 0
        thumb = int(thumb_text)
        if stride <= 0:
            stride = max(1, thumb // (GRID * _SAMPLES_PER_CELL))
        out.append(ExtraScale(thumb=thumb, stride=stride, label=f"{thumb}px/s{stride}"))
    return out


_COORD_PLANES: dict[int, tuple[FloatArray, FloatArray]] = {}
_MORPH_KERNELS: dict[int, NDArray[np.uint8]] = {}


def _coord_planes(grid_size: int) -> tuple[FloatArray, FloatArray]:
    """Normalised cell coordinates in [-0.5, 0.5] as ``(xx, yy)`` planes, built once per size."""
    planes = _COORD_PLANES.get(grid_size)
    if planes is None:
        axis = (np.arange(grid_size, dtype=np.float32) / float(grid_size - 1) - 0.5).astype(
            np.float32
        )
        yy = np.repeat(axis[:, None], grid_size, axis=1)
        xx = np.repeat(axis[None, :], grid_size, axis=0)
        planes = _COORD_PLANES[grid_size] = (np.ascontiguousarray(xx), np.ascontiguousarray(yy))
    return planes


def _morph_kernel(size: int) -> NDArray[np.uint8]:
    kernel = _MORPH_KERNELS.get(size)
    if kernel is None:
        kernel = _MORPH_KERNELS[size] = np.ones((size, size), np.uint8)
    return kernel


def _box(pooled: FloatArray, size: int) -> FloatArray:
    return np.asarray(
        cv2.boxFilter(pooled, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT),
        dtype=np.float32,
    )


def compute_context_values(pooled: FloatArray, grid_size: int) -> FloatArray:
    """Small-scale context from imfeat's per-cell luminance mean: ``(grid, grid, 5)``.

    ``ctx_x``/``ctx_y`` are normalized cell coordinates in [-0.5, 0.5];
    ``ctx_surr3`` is a Laplacian (cell minus 3x3 mean), ``ctx_ring35`` the
    difference between the 5x5 and 3x3 means and ``ctx_range3`` the 3x3
    max-minus-min.  This is the reference, in OpenCV calls; the extractor computes
    the same bytes in one C++ call per level (:func:`context_banks`).
    """
    channels = len(CONTEXT_FEATURE_NAMES)
    out = np.empty((grid_size, grid_size, channels), dtype=np.float32)
    if grid_size < _CTX1_MIN_GRID:
        out.fill(0.0)
        return out
    xx, yy = _coord_planes(grid_size)
    out[..., 0] = xx
    out[..., 1] = yy
    blur3 = _box(pooled, _CTX1_SIZE)
    np.subtract(pooled, blur3, out=out[..., 2])
    np.subtract(_box(pooled, _CTX2_SIZE), blur3, out=out[..., 3])
    kernel = _morph_kernel(_CTX1_SIZE)
    np.subtract(cv2.dilate(pooled, kernel), cv2.erode(pooled, kernel), out=out[..., 4])
    return out


def compute_context2_values(pooled: FloatArray, grid_size: int) -> FloatArray:
    """Large-scale context from imfeat's per-cell luminance mean: ``(grid, grid, 2)``.

    ``ctx_surr9`` is the cell mean minus the 9x9 mean, ``ctx_range5`` the 5x5
    max-minus-min; together they cover wider surroundings than ``context``.  The
    reference, as :func:`compute_context_values`.
    """
    channels = len(CONTEXT2_FEATURE_NAMES)
    out = np.empty((grid_size, grid_size, channels), dtype=np.float32)
    if grid_size < _CTX2_MIN_GRID:
        out.fill(0.0)
        return out
    np.subtract(pooled, _box(pooled, _CTX2_SURROUND), out=out[..., 0])
    kernel = _morph_kernel(_CTX2_SIZE)
    np.subtract(cv2.dilate(pooled, kernel), cv2.erode(pooled, kernel), out=out[..., 1])
    return out


N_BANK_PLANES = len(CONTEXT_FEATURE_NAMES) + len(CONTEXT2_FEATURE_NAMES)  # 7
_N_CTX = len(CONTEXT_FEATURE_NAMES)


def _refs_when_idle() -> int:
    """What ``sys.getrefcount`` reports for a bank buffer nobody but its owner holds.

    Measured the way :meth:`FeatureExtractor._bank_planes` asks -- the owner's dict entry,
    a local, and the call's own argument -- so the guard does not depend on how many of
    those a given interpreter counts.
    """
    scratch = {0: np.zeros(1, dtype=np.float32)}
    buf = scratch.get(0)
    return sys.getrefcount(buf)


_IDLE_REFS = _refs_when_idle()


def context_banks(pooled: FloatArray, out: FloatArray, *, coords: bool = True) -> FloatArray:
    """Both context banks of one level into ``out``, a ``(7, grid, grid)`` float32 array.

    Planes 0 and 1 are the cell coordinates, constants written when ``coords`` is set
    (a caller reusing ``out`` across frames writes them once and passes ``False`` after);
    planes 2..6 are computed by the C++ extension in one call, bit for bit what
    :func:`compute_context_values` and :func:`compute_context2_values` give through their
    7 OpenCV calls (the tests pin it).  ``out[:5].transpose(1, 2, 0)`` and
    ``out[5:].transpose(1, 2, 0)`` are the two banks in the ``(grid, grid, n)`` shape the
    levels carry, each column a contiguous plane.
    """
    if coords:
        grid_size = out.shape[1]
        if grid_size < _CTX1_MIN_GRID:
            out[:2] = 0.0
        else:
            out[0], out[1] = _coord_planes(grid_size)
    _banks_extension().context_banks(pooled, out)
    return out


_BANKS_EXTENSION: Any = None


def _banks_extension() -> Any:
    """The C++ extension, resolved once (``native`` imports this module, hence lazily)."""
    global _BANKS_EXTENSION  # noqa: PLW0603 -- a module-level cache
    if _BANKS_EXTENSION is None:
        from .native import _extension  # noqa: PLC0415 -- see above

        _BANKS_EXTENSION = _extension()
    return _BANKS_EXTENSION


def compute_global_stats(global_raw: FloatArray, orig_h: int, orig_w: int) -> FloatArray:
    """Whole-image block from imfeat, plus the original frame's shape.

    Written into one float32 array: the block's values as they are (non-finite ones
    zeroed), then the aspect ratio and the log area rounded to float32 once.
    """
    n = global_raw.shape[0]
    out = np.empty(n + 2, dtype=np.float32)
    out[:n] = global_raw
    finite = np.isfinite(out[:n])
    if not finite.all():  # NaN and +-inf to zero, as nan_to_num would
        out[:n][~finite] = 0.0
    out[n] = orig_w / max(orig_h, 1)
    out[n + 1] = np.log1p(orig_h * orig_w)
    return out


def default_threads() -> int:
    """Threads used when none are given (imfeat and the scorer): the cores, at most 4."""
    return max(1, min(_MAX_DEFAULT_THREADS, os.cpu_count() or 1))


def thumb_hw(thumb: int | str, shape: tuple[int, ...]) -> tuple[int, int]:
    """The ``(rows, cols)`` thumbnail of a frame of ``shape`` under ``thumb``.

    A side in pixels is a square of that size whatever the frame (smaller frames are
    upscaled to it); a policy name is imfeat's rule on the frame's shape
    (``imfeat.thumb_size``: ``"pow2"`` is the square of the shorter side's power of two,
    ``"pow2-fit"`` keeps the frame's shape inside that square, as closely as the grid's 64
    px rounding allows, ``"pow2-cover"`` around it; never an upscale), floored at the
    ``GRID`` px the grid needs, so a frame smaller than that is upscaled to 64 px as a
    fixed size would upscale it.
    """
    rows, cols = imfeat.thumb_size(shape[:2], thumb)
    if isinstance(thumb, str):
        rows, cols = max(rows, GRID), max(cols, GRID)
    return rows, cols


def stride_for(stride: int, size: tuple[int, int]) -> int:
    """The sampling stride on a thumbnail of ``size``: ``stride``, at most the cell side.

    imfeat restarts the stride at every finest-cell edge along the columns but not the
    rows, so a stride wider than a cell leaves some cells without a sample; with a policy
    the cells shrink with the frame, and the stride is capped at their side so every cell
    keeps at least one.  A host that runs imfeat itself applies the same rule.
    """
    return max(1, min(int(stride), min(size) // GRID))


def make_feature_computer(  # noqa: PLR0913 -- the front-end's knobs, all explicit
    available_levels: tuple[int, ...],
    thumb: int | str | tuple[int, int] = THUMB,
    stride: int = STRIDE,
    threads: int | None = None,
    space: str = "hsv",
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> tuple[Any, int]:
    """Build the imfeat computer for ``available_levels``, with features in ``space``.

    The computer takes the BGR thumbnail when imfeat converts to ``space`` itself (HSV:
    inside the pass, ``cv2.cvtColor``'s bytes exactly) and the converted thumbnail
    otherwise (see :class:`SpaceInfo`).  ``thumb`` is the thumbnail's side, or its
    ``(rows, cols)``, or with ``frame_shape`` also a policy name: the computer then takes
    the frame itself and makes the thumbnail inside the pass (``cv2.resize``'s
    ``INTER_AREA`` bytes exactly; the frame must be at least that size in both axes, and
    the size is what :func:`thumb_hw` gives, stride :func:`stride_for`'s).  imfeat
    partitions the work by cell, so its output is bit-identical for any thread count;
    ``threads`` only sets how fast the pass runs (``None`` = :func:`default_threads`).
    """
    exponents = [exponent_for_size(size) for size in available_levels]
    size: tuple[int, int]
    if frame_shape is None:
        if isinstance(thumb, str):
            msg = f"thumb={thumb!r} is a policy: the thumbnail's size needs the frame's shape"
            raise ValueError(msg)
        size = (thumb, thumb) if isinstance(thumb, int) else (int(thumb[0]), int(thumb[1]))
        shape: tuple[int, ...] = (*size, 3)
    else:
        shape = tuple(frame_shape)
        size = (
            thumb_hw(thumb, shape)
            if isinstance(thumb, (int, str))
            else (int(thumb[0]), int(thumb[1]))
        )
    computer = imfeat.FeatureComputer(
        shape=shape,
        grid=[(e, e) for e in exponents],
        stride=stride_for(stride, size),
        threads=threads if threads is not None else default_threads(),
        input_space="bgr",
        feature_space=SPACE_INFO[space].fused,
        thumb=None if frame_shape is None else size,
    )
    return computer, len(exponents)


class FeatureExtractor:
    """Config-driven front-end: an image in, per-level feature maps out.

    The extractor is stateless with respect to images, so one instance serves a
    whole split (via :class:`FeatureCache`) or a single ``predict_proba`` call.
    """

    def __init__(
        self, cfg: TrainConfig, threads: int | None = None, *, fuse_resize: bool = True
    ) -> None:
        """Configure the imfeat pyramid and the enabled feature banks.

        With ``fuse_resize`` (the default) a frame at least its thumbnail's size in both
        axes is handed to imfeat whole and thumbnailed inside its pass, by a computer
        built for that frame size and rebuilt when the size changes -- right for a stream,
        wasteful for a set of images of many sizes (:class:`FeatureCache` turns it off).
        The numbers are the same either way.
        """
        self.cfg = cfg
        self.fuse_resize = fuse_resize
        self.levels: tuple[int, ...] = tuple(sorted(cfg.levels, reverse=True))
        self.extra_scales: list[ExtraScale] = parse_extra_scales(cfg.extra_scales)
        self.extra_labels: tuple[str, ...] = tuple(scale.label for scale in self.extra_scales)
        self.raw_width = FEATURE_CHANNELS * (1 + len(self.extra_scales))

        self.space = SPACE_INFO[cfg.imfeat_space]

        tags = iter_feature_mode_tags(cfg.feature_mode)
        self.compute_global_values = "global" in tags
        self.compute_context_values = "context" in tags
        self.compute_ctx2_values = "ctx2" in tags

        self.global_names = (
            list(global_stat_names(self.space)) if self.compute_global_values else []
        )
        self.context_names = list(CONTEXT_FEATURE_NAMES) if self.compute_context_values else []
        self.ctx2_names = list(CONTEXT2_FEATURE_NAMES) if self.compute_ctx2_values else []

        self.block_ranges, self.broadcast_ranges = self._compute_block_ranges()
        self._plans: dict[bytes, tuple[int, list[_Copy]]] = {}
        self._all_cells = np.arange(GRID * GRID)

        self.threads = threads
        self.n_levels = len(self.levels)
        # The imfeat computers (and their worker pools) are built on the first pass, so a
        # host that runs imfeat itself and feeds compose() never spawns them.  One per
        # thumbnail size for ready thumbnails (a policy makes a few sizes), one per extra
        # scale, and one for the frame size last seen when the resize is fused.
        self._computers: dict[tuple[int, int], Any] = {}
        self._extra: list[tuple[Any, int, str]] | None = None
        self._fused: tuple[tuple[int, ...], Any] | None = None  # (frame shape, computer)
        # the context banks' buffers, one per level, reused while nothing else holds them
        self._bank_scratch: dict[int, FloatArray] = {}
        self._level_plans: dict[int, tuple[bool, bool, bool, int, int, int]] = {}

        self.base_names = self._build_names()

    def thumb_hw(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """The ``(rows, cols)`` thumbnail a frame of ``shape`` gets (see :func:`thumb_hw`)."""
        return thumb_hw(self.cfg.thumb, shape)

    def thumb_computer(self, size: tuple[int, int]) -> Any:
        """The imfeat computer that takes a ready thumbnail of ``size`` (built on first use)."""
        fc = self._computers.get(size)
        if fc is None:
            cfg = self.cfg
            fc, _ = make_feature_computer(
                self.levels, size, cfg.stride, self.threads, cfg.imfeat_space
            )
            self._computers[size] = fc
        return fc

    @property
    def extra_computers(self) -> list[tuple[Any, int, str]]:
        """``(computer, thumb, label)`` per extra scale (built on first use)."""
        if self._extra is None:
            cfg = self.cfg
            self._extra = [
                (
                    make_feature_computer(
                        self.levels, scale.thumb, scale.stride, self.threads, cfg.imfeat_space
                    )[0],
                    scale.thumb,
                    scale.label,
                )
                for scale in self.extra_scales
            ]
        return self._extra

    def fuses_resize(self, img_bgr: Image) -> bool:
        """Whether ``img_bgr`` goes into imfeat whole, which then makes the thumbnail.

        Fusing on, the colour conversion inside the pass (HSV), the box filter, and a BGR
        frame at least the thumbnail's size in both axes (an upscale is bilinear in OpenCV,
        not ``INTER_AREA``; under a policy the thumbnail never outgrows the frame except
        below the 64 px floor); otherwise cv2 resizes first, which for a frame smaller
        than the thumbnail is the cheap bilinear path anyway.
        """
        cfg = self.cfg
        if not (
            self.fuse_resize
            and SPACE_INFO[cfg.imfeat_space].fused is not None
            and cfg.resize_interp == "area"
            and img_bgr.ndim == _COLOUR_NDIM
            and img_bgr.shape[2] == RAW_CHANNELS
        ):
            return False
        rows, cols = self.thumb_hw(img_bgr.shape)
        return bool(img_bgr.shape[0] >= rows and img_bgr.shape[1] >= cols)

    def _fused_computer(self, shape: tuple[int, ...]) -> Any:
        """The computer that takes frames of ``shape``, kept until the shape changes."""
        if self._fused is None or self._fused[0] != shape:
            cfg = self.cfg
            fc, _ = make_feature_computer(
                self.levels,
                self.thumb_hw(shape),
                cfg.stride,
                self.threads,
                cfg.imfeat_space,
                frame_shape=shape,
            )
            self._fused = (shape, fc)
        return self._fused[1]

    def _compute_block_ranges(self) -> tuple[RangeTable, RangeTable]:
        """Offset tables indexing the per-cell array and the broadcast vector."""
        ranges_by_size: RangeTable = {}
        broadcast_by_size: RangeTable = {}
        for size in self.levels:
            offset = 0
            ranges: dict[str, Range] = {}
            ranges["raw"] = (offset, offset + self.raw_width)
            offset += self.raw_width
            for tag, names in (
                ("context", self.context_names),
                ("ctx2", self.ctx2_names),
            ):
                ranges[tag] = (offset, offset + len(names))
                offset += len(names)
            ranges_by_size[size] = ranges

            # The global bank is identical at every level, so carry it once, at the
            # finest one, instead of repeating it len(levels) times in every row.
            width = len(self.global_names) if size == self.levels[0] else 0
            broadcast_by_size[size] = {"global": (0, width)}
        return ranges_by_size, broadcast_by_size

    def _build_names(self) -> list[str]:
        """Flat base-feature names in :meth:`gather` column order."""
        letters = SPACE_INFO[self.cfg.imfeat_space].letters
        scale_tags: list[str | None] = [None, *list(self.extra_labels)]
        names: list[str] = []
        banks = {"context": self.context_names, "ctx2": self.ctx2_names}
        for size in self.levels:
            for tag in iter_feature_mode_tags(self.cfg.feature_mode):
                if tag == "raw":
                    for scale_tag in scale_tags:
                        prefix = f"{size}" + ("" if scale_tag is None else f"@{scale_tag}")
                        names.extend(
                            f"{prefix}/{channel}/{feature}"
                            for channel in letters
                            for feature in RAW_FEATURE_NAMES
                        )
                elif tag == "global":
                    low, high = self.broadcast_ranges[size]["global"]
                    names.extend(f"{size}/global/{name}" for name in self.global_names[low:high])
                else:
                    names.extend(f"{size}/{tag}/{name}" for name in banks[tag])
        return names

    def tag_width(self, size: int, tag: str) -> int:
        """Number of columns ``tag`` contributes at level ``size``."""
        ranges = self.broadcast_ranges[size] if tag in BROADCAST_TAGS else self.block_ranges[size]
        low, high = ranges.get(tag, (0, 0))
        return max(0, high - low)

    def level_width(self, size: int) -> int:
        """Columns per cell at ``size`` for the active feature mode."""
        return sum(
            self.tag_width(size, tag) for tag in iter_feature_mode_tags(self.cfg.feature_mode)
        )

    def total_width(self) -> int:
        """Columns per cell summed over every level (the full design width)."""
        return sum(self.level_width(size) for size in self.levels)

    def _resize_bgr(
        self, img: Image, size: int | tuple[int, int], interpolation: int = cv2.INTER_AREA
    ) -> Image:
        rows, cols = (size, size) if isinstance(size, int) else size
        resized = cv2.resize(img, (cols, rows), interpolation=interpolation)
        return np.asarray(resized, dtype=np.uint8)

    def _convert(self, thumb_bgr: Image, space: SpaceInfo) -> Image:
        """What the computer is given: the BGR thumbnail itself when imfeat converts."""
        if space.convert is None:
            return thumb_bgr
        converted = cv2.cvtColor(thumb_bgr, space.convert)
        return np.asarray(converted, dtype=np.uint8)

    def _load_level_maps(self, result: Any, source: str) -> list[FloatArray]:
        # imfeat's result.maps appends a trailing whole-image "global" entry
        # beyond the requested levels, hence the slice to n_levels.
        maps = list(result.maps)[: self.n_levels]
        if len(maps) != len(self.levels):
            msg = f"{source} returned {len(maps)} levels, expected {len(self.levels)}"
            raise RuntimeError(msg)
        return [np.asarray(m, dtype=np.float32) for m in maps]

    def _lum_cell_mean(self, primary_raw: FloatArray) -> FloatArray:
        """Per-cell mean of the luminance channel of one level's map, from imfeat.

        A contiguous copy of the one column of the map's per-cell feature vector.
        """
        column = primary_raw[..., self.space.lum * RAW_PER_CHANNEL + _MEAN_IDX]
        return np.ascontiguousarray(column, dtype=np.float32)

    def _compose_level(
        self, size: int, raw_arrays: list[FloatArray], global_vec: FloatArray
    ) -> tuple[LevelBanks, FloatArray]:
        """One level's banks and broadcast vector.

        imfeat's maps are used as they come -- no copy into a wider array, and no
        sanitising: they are always finite (every ratio is guarded; the tests pin
        it on degenerate frames).  They are views of imfeat's pooled block, which
        is released when the caller drops them; :class:`FeatureCache` copies.
        """
        expected = (size, size, FEATURE_CHANNELS)
        for arr in raw_arrays:
            if arr.shape != expected:
                msg = f"unexpected map shape for level {size}: {arr.shape}"
                raise RuntimeError(msg)
        banks: list[FloatArray] = list(raw_arrays)
        has_ctx, has_ctx2, whole_global, broadcast_total, low, high = self._level_plan(size)
        if has_ctx or has_ctx2:
            planes, fresh = self._bank_planes(size)
            context_banks(self._lum_cell_mean(raw_arrays[0]), planes, coords=fresh)
            if has_ctx:
                banks.append(planes[:_N_CTX].transpose(1, 2, 0))
            if has_ctx2:
                banks.append(planes[_N_CTX:].transpose(1, 2, 0))
        if whole_global:  # the global block is the whole broadcast vector: no copy
            return tuple(banks), global_vec[:broadcast_total]
        bvec = np.zeros(broadcast_total, dtype=np.float32)
        if high > low:
            bvec[low:high] = global_vec[low:high]
        return tuple(banks), bvec

    def _level_plan(self, size: int) -> tuple[bool, bool, bool, int, int, int]:
        """What :meth:`_compose_level` does at ``size``, settled once per level.

        Whether the level has the ``context`` and ``ctx2`` banks, whether its broadcast
        vector is the global block whole, and that vector's width and the block's range.
        """
        plan = self._level_plans.get(size)
        if plan is None:
            ranges = self.block_ranges[size]
            ctx = ranges.get("context", (0, 0))
            ctx2 = ranges.get("ctx2", (0, 0))
            broadcast_ranges = self.broadcast_ranges[size]
            total = max((hi for (_, hi) in broadcast_ranges.values()), default=0)
            low, high = broadcast_ranges.get("global", (0, 0))
            plan = (
                ctx[1] > ctx[0],
                ctx2[1] > ctx2[0],
                low == 0 and high == total,
                total,
                low,
                high,
            )
            self._level_plans[size] = plan
        return plan

    def _bank_planes(self, size: int) -> tuple[FloatArray, bool]:
        """The ``(7, size, size)`` buffer the level's context banks are written into.

        Kept per level and reused frame after frame -- unless something still holds the
        last frame's banks (they are views of it, and a view holds a reference), in which
        case a fresh one is made, so results a caller keeps are never overwritten.  The
        reference count of a buffer nobody else holds was measured at import, by the same
        statements (``_IDLE_REFS``).  The flag says the buffer is new (its constant planes
        are not written yet).
        """
        buf = self._bank_scratch.get(size)
        if buf is None or sys.getrefcount(buf) > _IDLE_REFS:
            buf = np.zeros((N_BANK_PLANES, size, size), dtype=np.float32)
            self._bank_scratch[size] = buf
            return buf, True
        return buf, False

    def close(self) -> None:
        """Drop the imfeat computers (their worker threads are joined by the destructor).

        The next pass builds them again; a host feeding :meth:`compose` never needs them.
        """
        self._computers = {}
        self._extra = None
        self._fused = None
        self._bank_scratch = {}

    @property
    def front_end_spec(self) -> dict[str, Any]:
        """The imfeat call this extractor makes, for hosts that run imfeat themselves.

        A host (framegate) that computes the same thumbnail and ``FeatureComputer``
        checks its settings against this and feeds :meth:`compose` instead of paying
        for a second pass.  ``thumb`` is the thumbnail's side in pixels (square) or the
        imfeat policy that sizes it from the frame (``"pow2-fit"`` by default; a host
        resolves it with ``imfeat.thumb_size(frame.shape, spec["thumb"])``, floored at 64 px
        as :func:`thumb_hw` does), ``stride`` the imfeat sampling stride (capped at the
        thumbnail's cell side, :func:`stride_for`), ``levels`` the grid sizes requested
        (imfeat appends its whole-image level), ``space`` the colour space the features
        are computed in and ``input_space`` that of the array the computer is given:
        ``"bgr"`` when imfeat converts inside its pass (``FeatureComputer(feature_space=
        space)``, the HSV default), otherwise ``space`` itself, converted with OpenCV
        first and taken as it is (``feature_space=None``); ``extra_scales`` are further
        ``(thumb, stride)`` passes on downsized copies of the thumbnail, in the order
        :meth:`compose` expects them.  A host with BGR frames at least the thumbnail's
        size in both axes can let imfeat make it too (``FeatureComputer(shape=frame.shape,
        thumb=spec["thumb"], ...)`` on the frame, a policy name included): the bytes are
        ``cv2.resize``'s ``INTER_AREA`` ones, so the result is the same.
        """
        cfg = self.cfg
        space = SPACE_INFO[cfg.imfeat_space]
        return {
            "thumb": cfg.thumb,
            "stride": cfg.stride,
            "resize_interp": cfg.resize_interp,
            "space": cfg.imfeat_space,
            "input_space": "bgr" if space.fused is not None else cfg.imfeat_space,
            "levels": list(self.levels),
            "raw_channels_per_level": RAW_CHANNELS * RAW_PER_CHANNEL,
            "extra_scales": [(e.thumb, e.stride) for e in self.extra_scales],
        }

    def run_imfeat(self, img_bgr: Image) -> tuple[Any, list[Any]]:
        """Resize and run imfeat: ``(result, extra_results)`` for :meth:`compose`.

        The thumbnail resize and the colour conversion happen inside imfeat's pass when
        they can (see :meth:`fuses_resize`; the frame goes in whole and the pass makes the
        thumbnail, ``cv2.resize``'s bytes exactly), or here first otherwise -- a frame
        smaller than the thumbnail, the nearest-neighbour filter, a colour space imfeat
        cannot convert to (see :class:`SpaceInfo`).  Same numbers either way.
        """
        cfg = self.cfg
        space = SPACE_INFO[cfg.imfeat_space]
        size = self.thumb_hw(img_bgr.shape)
        if self.fuses_resize(img_bgr):
            fc = self._fused_computer(img_bgr.shape)
            if not self.extra_scales:
                return fc.features(img_bgr), []
            # the extra scales are made from the thumbnail: have the pass write it out
            thumb_bgr = np.empty((*size, 3), np.uint8)
            result = fc.features(img_bgr, thumb_out=thumb_bgr)
        else:
            interpolation = cv2.INTER_AREA if cfg.resize_interp == "area" else cv2.INTER_NEAREST
            thumb_bgr = self._resize_bgr(img_bgr, size, interpolation)
            result = self.thumb_computer(size).features(self._convert(thumb_bgr, space))
        extra_results = []
        for computer, extra_thumb, _label in self.extra_computers:
            extra_bgr = self._resize_bgr(thumb_bgr, extra_thumb)
            extra_results.append(computer.features(self._convert(extra_bgr, space)))
        return result, extra_results

    def compose(
        self,
        result: Any,
        image_hw: tuple[int, int],
        extra_results: Sequence[Any] = (),
    ) -> tuple[dict[int, LevelBanks], dict[int, FloatArray]]:
        """Imfeat results -> ``(level_maps, broadcast_vecs)`` (see module docstring).

        ``result`` is what ``imfeat.FeatureComputer.features`` returned for the
        thumbnail :attr:`front_end_spec` describes, ``image_hw`` the original frame's
        ``(height, width)`` (the global block records its aspect and area) and
        ``extra_results`` the extra scales' results in spec order.  This is the second
        half of :meth:`extract`; a host with its own imfeat pass calls it directly.
        """
        maps = self._load_level_maps(result, "imfeat")
        if len(extra_results) != len(self.extra_scales):
            msg = f"expected {len(self.extra_scales)} extra-scale results, got {len(extra_results)}"
            raise ValueError(msg)
        extra_maps = [self._load_level_maps(r, "extra scale") for r in extra_results]
        for level_maps_of_scale in (maps, *extra_maps):
            for size, arr in zip(self.levels, level_maps_of_scale, strict=True):
                want = (size, size, RAW_CHANNELS * RAW_PER_CHANNEL)
                if tuple(arr.shape) != want:
                    msg = f"imfeat map has shape {tuple(arr.shape)}, this model expects {want}"
                    raise ValueError(msg)
        orig_h, orig_w = image_hw
        # result.maps[-1] is imfeat's whole-image level: the same raw block, one cell.
        global_vec = compute_global_stats(
            np.asarray(result.maps[-1], dtype=np.float32).ravel(), orig_h, orig_w
        )
        level_maps: dict[int, LevelBanks] = {}
        broadcast_vecs: dict[int, FloatArray] = {}
        for i, size in enumerate(self.levels):
            raw_arrays = [maps[i], *(scale_maps[i] for scale_maps in extra_maps)]
            out, bvec = self._compose_level(size, raw_arrays, global_vec)
            level_maps[size] = out
            broadcast_vecs[size] = bvec
        return level_maps, broadcast_vecs

    def extract(self, img_bgr: Image) -> tuple[dict[int, LevelBanks], dict[int, FloatArray]]:
        """Image -> ``(level_maps, broadcast_vecs)`` (see module docstring)."""
        result, extra_results = self.run_imfeat(img_bgr)
        return self.compose(result, img_bgr.shape[:2], extra_results)

    def gather(
        self,
        level_maps: dict[int, LevelBanks],
        broadcast_vecs: dict[int, FloatArray],
        flat_indices: NDArray[np.integer],
        col_keep: NDArray[np.integer] | None = None,
    ) -> FloatArray:
        """``(n_cells, width)`` design matrix for flat 64x64 cell indices.

        ``col_keep`` selects a strictly ascending subset of the canonical columns
        (e.g. the pruned set), so kept columns stay in canonical order.

        Each output column is a coarse level value repeated over that level's
        block, so the matrix is written in one pass straight from the level maps:
        no per-level temporaries, no concatenation, no column re-selection.  For the
        whole 64x64 grid the repetition is a broadcast into a ``(L, f, L, f, k)``
        view of the output; for sampled cells it is an index by parent cell.
        """
        width, copies = self._gather_plan(col_keep)
        flat = np.asarray(flat_indices)
        out = np.empty((len(flat), width), dtype=np.float32)
        whole_grid = len(flat) == GRID * GRID and np.array_equal(flat, self._all_cells)
        rows, cols = (None, None) if whole_grid else np.divmod(flat, GRID)
        for copy in copies:
            if copy.broadcast:
                out[:, copy.dst0 : copy.dst1] = broadcast_vecs[copy.size][copy.src]
                continue
            bank = level_maps[copy.size][copy.bank][..., copy.src]
            factor = GRID // copy.size
            if rows is None or cols is None:
                blocks = out.reshape(copy.size, factor, copy.size, factor, width)
                blocks[..., copy.dst0 : copy.dst1] = bank[:, None, :, None, :]
            else:
                out[:, copy.dst0 : copy.dst1] = bank[rows // factor, cols // factor]
        return out

    def native_size(self, col_keep: NDArray[np.integer] | None = None) -> int:
        """Floats in :meth:`native`'s output for ``col_keep``: what a reusable buffer holds."""
        _width, copies = self._gather_plan(col_keep)
        return sum((c.dst1 - c.dst0) * (1 if c.broadcast else c.size * c.size) for c in copies)

    def native(
        self,
        level_maps: dict[int, LevelBanks],
        broadcast_vecs: dict[int, FloatArray],
        col_keep: NDArray[np.integer] | None = None,
        out: FloatArray | None = None,
    ) -> FloatArray:
        """Kept columns at native resolution, flat: the scorer's input, no dense table.

        Column ``j`` contributes its ``side x side`` values row-major, in column
        order, where ``side`` is the column's level (1 for globals); this is
        :meth:`gather` over the full grid with the repetition left out.

        ``out`` is written and returned in place of a new array: a flat, C-contiguous
        float32 array of :meth:`native_size` floats.  The packed features are a few
        megabytes, and a fresh array per frame is a fresh set of pages to fault in, so a
        stream reuses one buffer (:meth:`Detector.pack_native` keeps it); the bytes are the
        same either way.
        """
        _width, copies = self._gather_plan(col_keep)
        sides = [1 if copy.broadcast else copy.size for copy in copies]
        size = sum((c.dst1 - c.dst0) * side * side for c, side in zip(copies, sides, strict=True))
        if out is None:
            out = np.empty(size, dtype=np.float32)
        elif (
            out.dtype != np.float32
            or out.ndim != 1
            or out.shape[0] != size
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            msg = f"out must be a writeable C-contiguous float32 array of {size} values"
            raise ValueError(msg)
        pos = 0
        for copy, side in zip(copies, sides, strict=True):
            n_cols = copy.dst1 - copy.dst0
            run = out[pos : pos + n_cols * side * side]
            if copy.broadcast:
                run[:] = broadcast_vecs[copy.size][copy.src]
            else:
                whole = level_maps[copy.size][copy.bank]
                planes = whole.transpose(2, 0, 1)  # (n, side, side): the bank column-major
                if planes.flags.c_contiguous:
                    # each column already a contiguous plane (the context banks): a straight
                    # copy of the kept planes, no transpose and no temporary
                    dst = run.reshape(n_cols, side, side)
                    if isinstance(copy.src, slice):
                        dst[...] = planes[copy.src]
                    else:
                        np.take(planes, copy.src, axis=0, out=dst)
                elif isinstance(copy.src, slice) and copy.src == slice(0, whole.shape[-1]):
                    # the whole bank: one 2-D transpose in OpenCV (blocked; ~1.5x NumPy's)
                    cv2.transpose(
                        whole.reshape(-1, whole.shape[-1]), run.reshape(n_cols, side * side)
                    )
                else:
                    bank = whole[..., copy.src]  # (side, side, n)
                    run.reshape(n_cols, side, side)[...] = bank.transpose(2, 0, 1)
            pos += run.size
        return out

    def map_sources(self, col_keep: NDArray[np.integer] | None = None) -> MapSources:
        """Where the scorer finds each kept column in the level maps (see :class:`MapSources`).

        One slot per bank the kept columns touch, in :meth:`native`'s order, and per kept
        column its slot and its index along the bank's last axis -- the same
        ``(level, bank, column)`` :meth:`native` copies from, so the scorer reads the same
        values it would be given packed.
        """
        _width, copies = self._gather_plan(col_keep)
        slots: list[tuple[int, int]] = []
        slot_side: list[int] = []
        feature_slot: list[NDArray[np.int32]] = []
        feature_index: list[NDArray[np.int32]] = []
        for copy in copies:
            n_cols = copy.dst1 - copy.dst0
            slots.append((copy.size, -1 if copy.broadcast else copy.bank))
            slot_side.append(1 if copy.broadcast else copy.size)
            feature_slot.append(np.full(n_cols, len(slots) - 1, dtype=np.int32))
            index = (
                np.arange(copy.src.start, copy.src.stop)
                if isinstance(copy.src, slice)
                else np.asarray(copy.src)
            )
            feature_index.append(index.astype(np.int32))
        empty = np.zeros(0, dtype=np.int32)
        return MapSources(
            slots=tuple(slots),
            slot_side=np.asarray(slot_side, dtype=np.int32),
            feature_slot=np.concatenate(feature_slot) if feature_slot else empty,
            feature_index=np.concatenate(feature_index) if feature_index else empty,
        )

    def _segments(self) -> list[tuple[int, bool, int, int, int]]:
        """Canonical column layout: ``(size, broadcast, bank, base, width)`` runs.

        Laid end to end these are the columns of :meth:`gather`, in the order
        :attr:`base_names` is built in.  ``bank`` indexes the level's
        :data:`LevelBanks` and ``base`` is the first column inside it (broadcast
        runs read the level's broadcast vector instead).
        """
        segments: list[tuple[int, bool, int, int, int]] = []
        for size in self.levels:
            bank = 0
            for tag in iter_feature_mode_tags(self.cfg.feature_mode):
                if tag in BROADCAST_TAGS:
                    low, high = self.broadcast_ranges[size].get(tag, (0, 0))
                    if high > low:
                        segments.append((size, True, -1, low, high - low))
                    continue
                low, high = self.block_ranges[size].get(tag, (0, 0))
                if high <= low:
                    continue
                # The raw block holds one imfeat map per scale, each its own bank.
                widths = (
                    [FEATURE_CHANNELS] * ((high - low) // FEATURE_CHANNELS)
                    if tag == "raw"
                    else [high - low]
                )
                for width in widths:
                    segments.append((size, False, bank, 0, width))
                    bank += 1
        return segments

    def _gather_plan(self, col_keep: NDArray[np.integer] | None) -> tuple[int, list[_Copy]]:
        """Output width and the column runs :meth:`gather` copies, cached per ``col_keep``."""
        keep = None if col_keep is None else np.asarray(col_keep, dtype=np.intp)
        key = b"" if keep is None else keep.tobytes()
        if key in self._plans:
            return self._plans[key]
        total = self.total_width()
        if keep is not None and (
            np.any(np.diff(keep) <= 0) or (keep.size and (keep[0] < 0 or keep[-1] >= total))
        ):
            msg = "col_keep must be strictly ascending canonical column indices"
            raise ValueError(msg)
        mask = np.ones(total, dtype=bool) if keep is None else np.isin(np.arange(total), keep)
        copies: list[_Copy] = []
        column = dst = 0
        for size, broadcast, bank, base, width in self._segments():
            local = np.flatnonzero(mask[column : column + width])
            column += width
            if not local.size:
                continue
            contiguous = local[-1] - local[0] + 1 == local.size
            src: slice | NDArray[np.intp] = (
                slice(base + int(local[0]), base + int(local[-1]) + 1)
                if contiguous
                else (local + base).astype(np.intp)
            )
            copies.append(_Copy(size, broadcast, bank, src, dst, dst + local.size))
            dst += local.size
        self._plans[key] = (dst, copies)
        return dst, copies


class FeatureCache:
    """Extract and store per-level feature maps for a whole image split once."""

    def __init__(
        self, pairs: Sequence[tuple[str, str]], cfg: TrainConfig, desc: str = "cache"
    ) -> None:
        """Extract and cache feature maps for every readable pair in ``pairs``."""
        from .images import (  # noqa: PLC0415 -- lazy to break the images<->features cycle
            mask_to_grid_coverage,
            read_image,
            read_mask,
        )

        self.cfg = cfg
        # images of many sizes: a computer per frame size would be rebuilt for each
        self.extractor = FeatureExtractor(cfg, fuse_resize=False)
        self.base_names = self.extractor.base_names
        self.levels = self.extractor.levels

        self.level_maps_list: list[dict[int, LevelBanks]] = []
        self.broadcast_list: list[dict[int, FloatArray]] = []
        self.gt_coverage_list: list[FloatArray] = []
        self.img_paths: list[str] = []

        start = time.time()
        skipped = 0
        for i, (img_path, mask_path) in enumerate(pairs):
            img = read_image(img_path)
            mask = read_mask(mask_path)
            if img is None or mask is None:
                skipped += 1
                continue
            level_maps, broadcast_vecs = self.extractor.extract(img)
            # Copy: imfeat's maps are views of a per-frame pooled block, and holding
            # them for the whole split would keep every frame's block alive.
            self.level_maps_list.append(
                {size: tuple(bank.copy() for bank in banks) for size, banks in level_maps.items()}
            )
            self.broadcast_list.append(broadcast_vecs)
            self.gt_coverage_list.append(
                mask_to_grid_coverage(mask, GRID).ravel().astype(np.float32)
            )
            self.img_paths.append(img_path)
            if (i + 1) % 100 == 0:
                print(f"  [{desc}] extracted {i + 1}/{len(pairs)} ({time.time() - start:.1f}s)")

        if not self.level_maps_list:
            msg = f"no usable image/mask pairs in {desc} split (skipped {skipped})"
            raise RuntimeError(msg)
        self.n = len(self.level_maps_list)

    def gather(
        self,
        img_index: int,
        flat_indices: NDArray[np.integer],
        col_keep: NDArray[np.integer] | None = None,
    ) -> FloatArray:
        """Design-matrix rows for one cached image (see :meth:`FeatureExtractor.gather`)."""
        return self.extractor.gather(
            self.level_maps_list[img_index],
            self.broadcast_list[img_index],
            flat_indices,
            col_keep=col_keep,
        )

    def total_width(self) -> int:
        """Full per-cell column count across every level."""
        return self.extractor.total_width()


def feature_level_bits(name: str) -> tuple[int, int]:
    """``'8/a/hog0'`` -> ``(level, coarse_shift)`` where shift is ``2*log2(64/level)``.

    A level-``L`` column is constant on ``L x L`` blocks of the output grid, and the
    C++ binner bins one value per block.  Global columns are constant over the
    whole image, so they are a 1x1 level (shift 12): binned once, not 4096 times.
    """
    prefix, tag = name.split("/", 2)[:2]
    size = 1 if tag in BROADCAST_TAGS else int(prefix.split("@", 1)[0])
    if size < 1 or GRID % size or size & (size - 1):
        msg = f"{name!r}: level {size} does not tile the {GRID}x{GRID} grid dyadically"
        raise ValueError(msg)
    return size, 2 * ((GRID // size).bit_length() - 1)
