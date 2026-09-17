"""Multi-level image features and per-cell design-matrix assembly.

The front-end turns one BGR image into a small pyramid of ``size x size``
per-cell feature maps (size is a power of two from 64 down to 8) plus a
per-image scene-statistics vector that is broadcast to every cell.  A cell is
one square block of the 64x64 output grid, so the coarser maps are aggregated
views of the same scene and :meth:`FeatureExtractor.gather` can assemble a
single row per cell from every level.

Banks, in the order :meth:`FeatureExtractor.gather` emits them per level:

``raw``
    imfeat's 3x38 raw block per cell (plus a 114-wide block per extra scale).
``global``
    seven whole-image scalars, identical for every cell (broadcast).
``bard``
    multi-lag bar-detector stroke statistics (cover / spectrum / peak / balance).
``context``
    small-scale surround/ring/range of the pooled grey mean.
``ctx2``
    the large-scale companion of ``context`` (9x9 surround, 5x5 range).

The numeric behaviour here is frozen: it reproduces the design matrices the
shipped model was trained on bit-for-bit.
"""

from __future__ import annotations

import math
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

__all__ = [
    "BARD_LAGS",
    "BARD_TAU",
    "CONTEXT2_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "FEATURE_CHANNELS",
    "GLOBAL_STAT_NAMES",
    "GRID",
    "RAW_FEATURE_NAMES",
    "SPACE_INFO",
    "STRIDE",
    "THUMB",
    "FeatureCache",
    "FeatureExtractor",
    "exponent_for_size",
    "feature_level_bits",
    "parse_extra_scales",
    "parse_level_list",
]

Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]

GRID = 64  # Finest/output grid resolution; fixed regardless of the feature levels.
FEATURE_CHANNELS = 3 * 38  # 114: three channels x imfeat's 38-D raw block.
THUMB = 512  # Default square resize target for the feature pyramid.
STRIDE = 2  # Default imfeat sampling stride at the primary scale.
BARD_LAGS = (1, 2, 4)  # Default bar-detector neighbor distances in pixels.
BARD_TAU = 8.0  # Default per-pixel contrast floor (grey levels).

_EPS = 1e-9  # Guard for the pooled ratios below (never a real denominator).
_SAMPLES_PER_CELL = 4  # imfeat samples four points per cell; see parse_extra_scales.
_BAR_SPAN = 2  # A lag-d neighbor test reaches +/- d, i.e. spans 2*d.
_CTX1_SIZE = 3  # Small-scale context kernel side.
_CTX2_SIZE = 5  # Large-scale range kernel side.
_CTX2_SURROUND = 9  # Large-scale surround kernel side.
_CTX1_MIN_GRID = 3  # Below this the 3x3 context kernel is undefined.
_CTX2_MIN_GRID = 5  # Below this the 5x5 range kernel is undefined.
_COHERENCE_PERCENTILE = 75  # Percentile of the finest-level coherence used globally.
_LEVEL_SHIFT = {64: 0, 32: 2, 16: 4, 8: 6}  # 2*log2(64/level) per level size.


@dataclass(frozen=True)
class SpaceInfo:
    """Color-space metadata the front-end needs to drive imfeat and the banks."""

    convert: int
    letters: tuple[str, str, str]
    lum: int
    chr: int


# Color spaces the front-end can feed imfeat.  The 38-D block layout is
# channel-agnostic; the global scalars and the bar/context banks assume specific
# channel roles, hence `lum` (luminance-like) and `chr` (chroma used by the
# S_mean_global proxy).
SPACE_INFO: dict[str, SpaceInfo] = {
    "hsv": SpaceInfo(cv2.COLOR_BGR2HSV, ("H", "S", "V"), lum=2, chr=1),
    "lab": SpaceInfo(cv2.COLOR_BGR2LAB, ("L", "a", "b"), lum=0, chr=1),
    "luv": SpaceInfo(cv2.COLOR_BGR2LUV, ("L", "u", "v"), lum=0, chr=1),
    "yuv": SpaceInfo(cv2.COLOR_BGR2YUV, ("Y", "U", "V"), lum=0, chr=1),
}
CHANNEL_LETTERS = ("H", "S", "V")

# Per-channel 38-feature layout of a raw block, in exact imfeat order:
# f[0..4] structure tensor, f[5..13] HOG, f[14..15] extrema, f[16..25] LBP,
# f[26..33] derived nonlinear descriptors, f[34..37] moments.
RAW_FEATURE_NAMES = (
    "energy",
    "coherence",
    "anisotropy",
    "shear",
    "corner",
    "hog0",
    "hog1",
    "hog2",
    "hog3",
    "hog4",
    "hog5",
    "hog6",
    "hog7",
    "hog8",
    "vmax",
    "vmin",
    "lbp0",
    "lbp1",
    "lbp2",
    "lbp3",
    "lbp4",
    "lbp5",
    "lbp6",
    "lbp7",
    "lbp8",
    "lbp9",
    "std_skew",
    "excess_kurt",
    "edge_sharpness",
    "detail",
    "concentration",
    "cardinality",
    "grad_sparsity",
    "rms_contrast",
    "mean",
    "var",
    "m3",
    "m4",
)

GLOBAL_STAT_NAMES = (
    "texture_level",
    "S_mean_global",
    "V_mean_global",
    "V_var_global",
    "coherence_p75",
    "aspect_ratio",
    "log_orig_area",
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


def bank_channel(thumb_bgr: Image, bank_gray: str) -> Image:
    """The single grey channel driving the bar-detector and context banks.

    ``v`` is framegate's saturation-blind choice; ``y`` (BT.601 luma) and
    ``lstar`` (Lab L*) are alternatives derived from the same resized thumb.
    """
    if bank_gray == "v":
        return np.asarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2HSV)[:, :, 2], dtype=np.uint8)
    if bank_gray == "y":
        return np.asarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    return np.asarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2LAB)[:, :, 0], dtype=np.uint8)


def resolve_bard_channels(space_name: str, spec: str) -> list[tuple[str, int]]:
    """Resolve a bard-channel spec against the active space.

    Tokens: ``lum`` (luminance channel), ``chr1``/``chr2`` (the two chroma
    channels), a literal space letter, or ``all``.
    """
    space = SPACE_INFO[space_name]
    letters = space.letters
    role = {"lum": space.lum, "chr1": space.chr}
    role["chr2"] = next(i for i in range(len(letters)) if i not in (space.lum, space.chr))

    order: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            order.append(index)

    for raw_token in (spec or "lum").replace(" ", "").split(","):
        part = raw_token.strip().lower()
        if part == "all":
            add(role["lum"])
            add(role["chr1"])
            add(role["chr2"])
        elif part in role:
            add(role[part])
        elif len(part) == 1 and part.upper() in letters:
            add(letters.index(part.upper()))
    return [(letters[i], i) for i in order]


def _bar_detector_maps(gray: Image, lags: tuple[int, ...] = BARD_LAGS) -> FloatArray:
    """``(H, W)`` grey -> ``(2, L, H, W)`` response maps (dark, light) per lag.

    A pixel responds at lag ``d`` when it is darker (resp. lighter) than both
    neighbors at distance ``d`` in x and y; a straight stroke of width ``w``
    therefore fires only once ``d > w/2``, which makes the per-lag spectrum a
    width cue.  Pixels within ``d`` of an edge get zero.
    """
    grey: NDArray[np.int16] = gray.astype(np.int16)
    h, w = grey.shape
    lag_list = tuple(int(d) for d in lags)
    dark = np.zeros((len(lag_list), h, w), np.uint8)
    light = np.zeros((len(lag_list), h, w), np.uint8)
    for j, d in enumerate(lag_list):
        if _BAR_SPAN * d >= min(h, w):
            continue
        centre = grey[:, d : w - d]
        left = grey[:, : w - _BAR_SPAN * d]
        right = grey[:, _BAR_SPAN * d :]
        dark_h = np.minimum(np.maximum(left - centre, 0), np.maximum(right - centre, 0))
        light_h = np.minimum(np.maximum(centre - left, 0), np.maximum(centre - right, 0))
        dark[j][:, d : w - d] = np.maximum(dark[j][:, d : w - d], dark_h).astype(np.uint8)
        light[j][:, d : w - d] = np.maximum(light[j][:, d : w - d], light_h).astype(np.uint8)

        centre = grey[d : h - d, :]
        up = grey[: h - _BAR_SPAN * d, :]
        down = grey[_BAR_SPAN * d :, :]
        dark_v = np.minimum(np.maximum(up - centre, 0), np.maximum(down - centre, 0))
        light_v = np.minimum(np.maximum(centre - up, 0), np.maximum(centre - down, 0))
        dark[j][d : h - d, :] = np.maximum(dark[j][d : h - d, :], dark_v).astype(np.uint8)
        light[j][d : h - d, :] = np.maximum(light[j][d : h - d, :], light_v).astype(np.uint8)
    return np.stack([dark, light], axis=0).astype(np.float32)


def pool_bard_block(
    response_maps: FloatArray, grid_size: int, stride: int = 4, tau: float = BARD_TAU
) -> FloatArray:
    """``(2, L, H, W)`` responses -> ``(grid, grid, 8)`` per-cell stat block.

    Every aggregation pools only the pixels a stride-``stride`` imfeat mesh
    samples (``R[..., ::s, ::s]``) and only pixels whose peak response over all
    lags/polarities clears ``tau``.  Output columns: ``cover``, one ``spec`` per
    lag, ``peak`` (mass-weighted mean lag), ``peaked`` (spectrum dispersion) and
    ``bal`` (signed light-vs-dark balance in [-1, 1]).
    """
    dark_maps = response_maps[0].astype(np.float32)
    light_maps = response_maps[1].astype(np.float32)
    n_lags = dark_maps.shape[0]
    step = max(1, int(stride))

    def sampled_block_mean(source: FloatArray) -> FloatArray:
        sampled = source[::step, ::step]
        return np.asarray(
            cv2.resize(sampled, (grid_size, grid_size), interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        )

    response = np.maximum(dark_maps, light_maps)
    mask = (response.max(axis=0) >= tau).astype(np.float32)
    cover = sampled_block_mean(mask)

    totals = np.stack([sampled_block_mean(response[j] * mask) for j in range(n_lags)], axis=-1)
    all_totals = totals.sum(axis=-1)
    spectrum = np.where(
        all_totals[..., None] > _EPS,
        totals / np.maximum(all_totals[..., None], _EPS),
        0.0,
    )
    weights = np.arange(1, n_lags + 1, dtype=np.float32)
    peak = np.where(
        all_totals > _EPS,
        (totals * weights).sum(axis=-1) / np.maximum(all_totals, _EPS),
        0.0,
    )
    mean_response = all_totals / n_lags
    variance = np.mean((totals - mean_response[..., None]) ** 2, axis=-1)
    std_response = np.sqrt(np.maximum(variance, 0.0))
    peaked = np.where(
        mean_response > _EPS,
        std_response / np.maximum(mean_response, _EPS),
        0.0,
    )
    dark_totals = sampled_block_mean(dark_maps.max(axis=0) * mask)
    light_totals = sampled_block_mean(light_maps.max(axis=0) * mask)
    balance = np.where(
        dark_totals + light_totals > _EPS,
        (light_totals - dark_totals) / np.maximum(dark_totals + light_totals, _EPS),
        0.0,
    )
    return np.stack(
        [cover] + [spectrum[..., j] for j in range(n_lags)] + [peak, peaked, balance], axis=-1
    ).astype(np.float32)


def compute_context_values(v_gray: Image, grid_size: int) -> FloatArray:
    """Pooled grey channel -> ``(grid, grid, 5)`` small-scale context.

    ``ctx_x``/``ctx_y`` are normalized cell coordinates in [-0.5, 0.5];
    ``ctx_surr3`` is a Laplacian (cell minus 3x3 mean), ``ctx_ring35`` the
    difference between the 5x5 and 3x3 means and ``ctx_range3`` the 3x3
    max-minus-min.
    """
    channels = len(CONTEXT_FEATURE_NAMES)
    if grid_size < _CTX1_MIN_GRID:
        return np.zeros((grid_size, grid_size, channels), dtype=np.float32)
    pooled = np.asarray(
        cv2.resize(v_gray.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )
    blur3 = np.asarray(
        cv2.boxFilter(
            pooled, -1, (_CTX1_SIZE, _CTX1_SIZE), normalize=True, borderType=cv2.BORDER_REFLECT
        ),
        dtype=np.float32,
    )
    blur5 = np.asarray(
        cv2.boxFilter(
            pooled, -1, (_CTX2_SIZE, _CTX2_SIZE), normalize=True, borderType=cv2.BORDER_REFLECT
        ),
        dtype=np.float32,
    )
    kernel = np.ones((_CTX1_SIZE, _CTX1_SIZE), np.uint8)
    dilate3 = np.asarray(cv2.dilate(pooled, kernel), dtype=np.float32)
    erode3 = np.asarray(cv2.erode(pooled, kernel), dtype=np.float32)
    axis = np.arange(grid_size, dtype=np.float32)
    denom = float(grid_size - 1)
    yy = np.repeat((axis / denom - 0.5)[:, None], grid_size, axis=1)
    xx = np.repeat((axis / denom - 0.5)[None, :], grid_size, axis=0)
    return np.stack([xx, yy, pooled - blur3, blur5 - blur3, dilate3 - erode3], axis=-1).astype(
        np.float32
    )


def compute_context2_values(v_gray: Image, grid_size: int) -> FloatArray:
    """Pooled grey channel -> ``(grid, grid, 2)`` large-scale context.

    ``ctx_surr9`` is the cell mean minus the 9x9 mean, ``ctx_range5`` the 5x5
    max-minus-min; together they cover wider surroundings than ``context``.
    """
    channels = len(CONTEXT2_FEATURE_NAMES)
    if grid_size < _CTX2_MIN_GRID:
        return np.zeros((grid_size, grid_size, channels), dtype=np.float32)
    pooled = np.asarray(
        cv2.resize(v_gray.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )
    blur9 = np.asarray(
        cv2.boxFilter(
            pooled,
            -1,
            (_CTX2_SURROUND, _CTX2_SURROUND),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        ),
        dtype=np.float32,
    )
    kernel = np.ones((_CTX2_SIZE, _CTX2_SIZE), np.uint8)
    dilate5 = np.asarray(cv2.dilate(pooled, kernel), dtype=np.float32)
    erode5 = np.asarray(cv2.erode(pooled, kernel), dtype=np.float32)
    return np.stack([pooled - blur9, dilate5 - erode5], axis=-1).astype(np.float32)


def compute_global_stats(
    finest_raw: FloatArray, grid_size: int, orig_h: int, orig_w: int, space: SpaceInfo
) -> FloatArray:
    """Seven whole-image scalars from the finest raw block and original size."""
    full = finest_raw[:, :, :FEATURE_CHANNELS].reshape(grid_size, grid_size, 3, 38)
    lum_var = full[:, :, space.lum, 35]
    lum_mean = full[:, :, space.lum, 34]
    chr_mean = full[:, :, space.chr, 34]
    coherence = full[:, :, space.lum, 1]

    texture_level = float(np.sqrt(np.maximum(lum_var, 0.0)).mean())
    values = [
        texture_level,
        float(chr_mean.mean()),
        float(lum_mean.mean()),
        float(lum_var.mean()),
        float(np.percentile(coherence, _COHERENCE_PERCENTILE)),
        float(orig_w / max(orig_h, 1)),
        float(np.log1p(orig_h * orig_w)),
    ]
    return np.asarray(values, dtype=np.float32)


def make_feature_computer(
    available_levels: tuple[int, ...], thumb: int = THUMB, stride: int = STRIDE
) -> tuple[Any, int]:
    """Build the single-threaded imfeat computer for ``available_levels``."""
    exponents = [exponent_for_size(size) for size in available_levels]
    computer = imfeat.FeatureComputer(
        shape=(thumb, thumb, 3),
        grid=[(e, e) for e in exponents],
        stride=stride,
        threads=1,  # single-threaded: keeps extraction deterministic and comparable
    )
    return computer, len(exponents)


class FeatureExtractor:
    """Config-driven front-end: an image in, per-level feature maps out.

    The extractor is stateless with respect to images, so one instance serves a
    whole split (via :class:`FeatureCache`) or a single ``predict_proba`` call.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        """Configure the imfeat pyramid and the enabled feature banks."""
        self.cfg = cfg
        self.levels: tuple[int, ...] = tuple(sorted(cfg.levels, reverse=True))
        self.extra_scales: list[ExtraScale] = parse_extra_scales(cfg.extra_scales)
        self.extra_labels: tuple[str, ...] = tuple(scale.label for scale in self.extra_scales)
        self.raw_width = FEATURE_CHANNELS * (1 + len(self.extra_scales))

        tags = iter_feature_mode_tags(cfg.feature_mode)
        self.compute_global_values = "global" in tags
        self.compute_bard_values = "bard" in tags
        self.compute_context_values = "context" in tags
        self.compute_ctx2_values = "ctx2" in tags

        self.bard_channels = resolve_bard_channels(cfg.imfeat_space, "lum")
        self.bard_lags = tuple(cfg.bard_lags)

        self.global_names = list(GLOBAL_STAT_NAMES) if self.compute_global_values else []
        self.bard_names: list[str] = []
        if self.compute_bard_values:
            suffixes = (
                ["cover"]
                + [f"spec{j + 1}" for j in range(len(self.bard_lags))]
                + ["peak", "peaked", "bal"]
            )
            for letter, _ in self.bard_channels:
                self.bard_names += [f"bard{letter}_{name}" for name in suffixes]
        self.context_names = list(CONTEXT_FEATURE_NAMES) if self.compute_context_values else []
        self.ctx2_names = list(CONTEXT2_FEATURE_NAMES) if self.compute_ctx2_values else []
        self.has_bard = bool(self.bard_names)

        self.block_ranges, self.broadcast_ranges = self._compute_block_ranges()

        self.fc, self.n_levels = make_feature_computer(self.levels, cfg.thumb, cfg.stride)
        self.extra_computers: list[tuple[Any, int, str]] = []
        for scale in self.extra_scales:
            computer, _ = make_feature_computer(self.levels, scale.thumb, scale.stride)
            self.extra_computers.append((computer, scale.thumb, scale.label))

        self.base_names = self._build_names()

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
                ("bard", self.bard_names),
                ("context", self.context_names),
                ("ctx2", self.ctx2_names),
            ):
                ranges[tag] = (offset, offset + len(names))
                offset += len(names)
            ranges_by_size[size] = ranges

            broadcast_offset = 0
            broadcast_ranges: dict[str, Range] = {}
            broadcast_ranges["global"] = (
                broadcast_offset,
                broadcast_offset + len(self.global_names),
            )
            broadcast_offset += len(self.global_names)
            broadcast_by_size[size] = broadcast_ranges
        return ranges_by_size, broadcast_by_size

    def _build_names(self) -> list[str]:
        """Flat base-feature names in :meth:`gather` column order."""
        letters = SPACE_INFO[self.cfg.imfeat_space].letters
        scale_tags: list[str | None] = [None, *list(self.extra_labels)]
        names: list[str] = []
        banks = {
            "bard": self.bard_names,
            "context": self.context_names,
            "ctx2": self.ctx2_names,
        }
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
                    names.extend(f"{size}/global/{name}" for name in GLOBAL_STAT_NAMES)
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

    def _resize_bgr(self, img: Image, size: int, interpolation: int = cv2.INTER_AREA) -> Image:
        resized = cv2.resize(img, (size, size), interpolation=interpolation)
        return np.asarray(resized, dtype=np.uint8)

    def _convert(self, thumb_bgr: Image, space: SpaceInfo) -> Image:
        converted = cv2.cvtColor(thumb_bgr, space.convert)
        return np.asarray(converted, dtype=np.uint8)

    def _bank_image(self, thumb_bgr: Image, feature_img: Image) -> Image:
        if self.cfg.bank_gray == "v" and self.cfg.imfeat_space == "hsv":
            return np.asarray(feature_img[:, :, 2], dtype=np.uint8)
        return bank_channel(thumb_bgr, self.cfg.bank_gray)

    def _bard_scale_entries(
        self, thumb_bgr: Image, feature_img: Image, space: SpaceInfo
    ) -> list[tuple[int, list[tuple[str, FloatArray]]]]:
        scale_feature_imgs: dict[int, Image] = {self.cfg.thumb: feature_img}
        for _computer, extra_thumb, _label in self.extra_computers:
            extra_bgr = self._resize_bgr(thumb_bgr, extra_thumb)
            scale_feature_imgs.setdefault(extra_thumb, self._convert(extra_bgr, space))
        primary = scale_feature_imgs[self.cfg.thumb]
        entries = [
            (
                letter,
                _bar_detector_maps(
                    np.asarray(primary[:, :, channel], dtype=np.uint8), self.bard_lags
                ),
            )
            for letter, channel in self.bard_channels
        ]
        return [(self.cfg.stride, entries)]

    def _load_level_maps(self, result: Any, source: str) -> list[FloatArray]:
        # imfeat's result.maps appends a trailing whole-image "global" entry
        # beyond the requested levels, hence the slice to n_levels.
        maps = list(result.maps)[: self.n_levels]
        if len(maps) != len(self.levels):
            msg = f"{source} returned {len(maps)} levels, expected {len(self.levels)}"
            raise RuntimeError(msg)
        return [np.asarray(m, dtype=np.float32) for m in maps]

    def _extra_level_maps(self, thumb_bgr: Image, space: SpaceInfo) -> list[list[FloatArray]]:
        extra_maps: list[list[FloatArray]] = []
        for computer, extra_thumb, _label in self.extra_computers:
            extra_bgr = self._resize_bgr(thumb_bgr, extra_thumb)
            result = computer.features(self._convert(extra_bgr, space))
            extra_maps.append(self._load_level_maps(result, "extra scale"))
        return extra_maps

    def _compose_level(
        self,
        size: int,
        raw_arrays: list[FloatArray],
        bank: Image,
        bard_entries: list[tuple[int, list[tuple[str, FloatArray]]]],
        global_vec: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        expected = (size, size, FEATURE_CHANNELS)
        if raw_arrays[0].shape != expected:
            msg = f"unexpected map shape for level {size}: {raw_arrays[0].shape}"
            raise RuntimeError(msg)
        raw_clean = np.nan_to_num(raw_arrays[0], nan=0.0, posinf=0.0, neginf=0.0)
        for extra_arr in raw_arrays[1:]:
            if extra_arr.shape != expected:
                msg = f"unexpected extra-scale map shape for level {size}"
                raise RuntimeError(msg)
            clean_extra = np.nan_to_num(extra_arr, nan=0.0, posinf=0.0, neginf=0.0)
            raw_clean = np.concatenate([raw_clean, clean_extra], axis=-1)

        ranges = self.block_ranges[size]
        total_width = max((high for (_, high) in ranges.values()), default=raw_clean.shape[-1])
        out = np.empty((size, size, total_width), dtype=np.float32)

        low, high = ranges["raw"]
        out[:, :, low:high] = raw_clean

        low, high = ranges.get("bard", (0, 0))
        if high > low:
            parts = [
                pool_bard_block(entry_maps, size, stride=entry_stride, tau=self.cfg.bard_tau)
                for entry_stride, entries in bard_entries
                for _letter, entry_maps in entries
            ]
            out[:, :, low:high] = np.concatenate(parts, axis=-1)

        low, high = ranges.get("context", (0, 0))
        if high > low:
            out[:, :, low:high] = compute_context_values(bank, size)

        low, high = ranges.get("ctx2", (0, 0))
        if high > low:
            out[:, :, low:high] = compute_context2_values(bank, size)

        broadcast_ranges = self.broadcast_ranges[size]
        broadcast_total = max((high for (_, high) in broadcast_ranges.values()), default=0)
        bvec = np.zeros(broadcast_total, dtype=np.float32)
        low, high = broadcast_ranges.get("global", (0, 0))
        if high > low:
            bvec[low:high] = global_vec
        return out, bvec

    def extract(self, img_bgr: Image) -> tuple[dict[int, FloatArray], dict[int, FloatArray]]:
        """Image -> ``(level_maps, broadcast_vecs)`` (see module docstring)."""
        cfg = self.cfg
        interpolation = cv2.INTER_AREA if cfg.resize_interp == "area" else cv2.INTER_NEAREST
        thumb_bgr = self._resize_bgr(img_bgr, cfg.thumb, interpolation)
        space = SPACE_INFO[cfg.imfeat_space]
        feature_img = self._convert(thumb_bgr, space)
        result = self.fc.features(feature_img)

        bank = self._bank_image(thumb_bgr, feature_img)
        bard_entries = (
            self._bard_scale_entries(thumb_bgr, feature_img, space) if self.has_bard else []
        )
        maps = self._load_level_maps(result, "imfeat")
        extra_maps = self._extra_level_maps(thumb_bgr, space)

        orig_h, orig_w = img_bgr.shape[:2]
        finest_raw = np.nan_to_num(maps[0], nan=0.0, posinf=0.0, neginf=0.0)
        global_vec = compute_global_stats(finest_raw, self.levels[0], orig_h, orig_w, space)

        level_maps: dict[int, FloatArray] = {}
        broadcast_vecs: dict[int, FloatArray] = {}
        for i, size in enumerate(self.levels):
            raw_arrays = [maps[i], *(scale_maps[i] for scale_maps in extra_maps)]
            out, bvec = self._compose_level(size, raw_arrays, bank, bard_entries, global_vec)
            level_maps[size] = out
            broadcast_vecs[size] = bvec
        return level_maps, broadcast_vecs

    def gather(
        self,
        level_maps: dict[int, FloatArray],
        broadcast_vecs: dict[int, FloatArray],
        flat_indices: NDArray[np.integer],
        col_keep: NDArray[np.integer] | None = None,
    ) -> FloatArray:
        """``(n_cells, width)`` design matrix for flat 64x64 cell indices.

        ``col_keep`` selects a fixed subset of the canonical columns (e.g. the
        pruned set); kept columns stay in canonical order so every row lines up.
        """
        tags = iter_feature_mode_tags(self.cfg.feature_mode)
        rows, cols = np.unravel_index(flat_indices, (GRID, GRID))
        n_rows = len(flat_indices)
        blocks: list[FloatArray] = []
        for size in self.levels:
            arr = level_maps[size]
            factor = GRID // size
            coarse_rows = rows // factor
            coarse_cols = cols // factor
            for tag in tags:
                if tag in BROADCAST_TAGS:
                    low, high = self.broadcast_ranges[size].get(tag, (0, 0))
                    if high <= low:
                        continue
                    blocks.append(
                        np.broadcast_to(broadcast_vecs[size][low:high], (n_rows, high - low))
                    )
                    continue
                low, high = self.block_ranges[size].get(tag, (0, 0))
                if high <= low:
                    continue
                blocks.append(arr[coarse_rows, coarse_cols, low:high])
        if not blocks:
            return np.zeros((n_rows, 0), dtype=np.float32)
        design: FloatArray = np.concatenate(blocks, axis=1).astype(np.float32)
        if col_keep is not None:
            design = design[:, col_keep]
        return design


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
        self.extractor = FeatureExtractor(cfg)
        self.base_names = self.extractor.base_names
        self.levels = self.extractor.levels

        self.level_maps_list: list[dict[int, FloatArray]] = []
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
            self.level_maps_list.append(level_maps)
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
    """``'8/a/hog0'`` -> ``(level, coarse_shift)`` where shift is ``2*log2(64/level)``."""
    prefix = name.split("/", 1)[0]
    size = int(prefix.split("@", 1)[0])
    return size, _LEVEL_SHIFT[size]
