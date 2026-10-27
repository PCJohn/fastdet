"""Booster fitting and importance-based column pruning."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from catboost import CatBoostClassifier
from numpy.typing import NDArray  # noqa: TC002 -- used in dataclass field annotations at runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from catboost import Pool

    from .config import ModelConfig

__all__ = [
    "LeafGrid",
    "auto_block_gib",
    "bin_bits",
    "build_ranking",
    "calibrate_exit_stages",
    "cuda_device_id",
    "default_feature_ranks_path",
    "fit_booster",
    "fit_booster_blocks",
    "gpu_free_gib",
    "host_available_gib",
    "leaf_grid",
    "load_ranking",
    "plan_blocks",
    "quantise_leaves",
    "resolve_task_type",
    "select_columns",
    "stage_tree_counts",
    "unused_blocks",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_LEAF_TABLE_NDIM = 2  # leaves are (n_trees, n_leaves)
_EMPTY_CHUNK_EXPONENT = -60  # a chunk whose trees are all constant: any step will do


def default_feature_ranks_path() -> Path:
    """Path of the bundled gain-importance ranking for the frozen front-end."""
    return _DATA_DIR / "feature_ranks.json"


def load_ranking(path: str | Path | None = None) -> dict[str, Any]:
    """Load a gain-importance ranking dict with a ``names`` list (descending)."""
    resolved = Path(path) if path is not None else default_feature_ranks_path()
    with resolved.open(encoding="utf-8") as fh:
        ranking: dict[str, Any] = json.load(fh)
    if not ranking.get("names"):
        msg = f"{resolved}: ranking has no 'names'"
        raise ValueError(msg)
    return ranking


def select_columns(
    base_names: list[str], ranking: Mapping[str, Any], top_k: int
) -> NDArray[np.int64]:
    """Indices (canonical order) of the top-``top_k`` ranked base columns.

    ``top_k <= 0`` keeps every column.  Missing ranked names are an error: a
    ranking for a different front-end silently drops real columns otherwise.
    """
    if top_k <= 0:
        return np.arange(len(base_names), dtype=np.int64)
    ranked = list(ranking["names"][:top_k])
    keep = set(ranked)
    missing = [n for n in ranked if n not in set(base_names)]
    if missing:
        msg = f"top-{top_k} ranked names not in this front-end: {missing[:5]}"
        raise ValueError(msg)
    columns: NDArray[np.int64] = np.array(
        [i for i, n in enumerate(base_names) if n in keep], dtype=np.int64
    )
    return columns


@dataclass
class LeafGrid:
    """The low-bit leaf grid: ``leaf = offset[tree] + code * 2**(e_min + shift[chunk])``.

    Steps are powers of two, one per chunk of trees, so the scorer sums integer
    codes shifted by the chunk's exponent -- exactly, in any order.  Chunks restart
    at every stage boundary (``stage_starts``) so that a chunked, quantisation-aware
    fit and the export land on the same grid.
    """

    bits: int
    chunk: int
    stage_starts: tuple[int, ...]
    offsets: NDArray[np.float32]  # (n_trees,)
    exponents: NDArray[np.int64]  # (n_chunks,) step = 2**exponent
    codes: NDArray[np.uint8]  # (n_trees, n_leaves)

    @property
    def n_chunks(self) -> int:
        """Number of quantisation chunks."""
        return len(self.exponents)

    @property
    def e_min(self) -> int:
        """Smallest chunk exponent; every chunk's step is ``2**(e_min + shift)``."""
        return int(self.exponents.min())

    @property
    def shifts(self) -> NDArray[np.uint8]:
        """Per-chunk exponent above ``e_min`` (what the blob stores)."""
        return (self.exponents - self.e_min).astype(np.uint8)

    def chunk_of_tree(self, n_trees: int) -> NDArray[np.int64]:
        """Chunk index of every tree (chunks restart at each stage boundary)."""
        out = np.zeros(n_trees, dtype=np.int64)
        base = 0
        starts = [*self.stage_starts, n_trees]
        for stage in range(len(starts) - 1):
            lo, hi = starts[stage], starts[stage + 1]
            out[lo:hi] = base + (np.arange(hi - lo) // self.chunk)
            base += -(-(hi - lo) // self.chunk)
        return out

    def values(self) -> NDArray[np.float64]:
        """The leaf values on the grid, ``(n_trees, n_leaves)`` float64."""
        n_trees = len(self.offsets)
        step = np.ldexp(1.0, self.exponents[self.chunk_of_tree(n_trees)])[:, None]
        return self.offsets[:, None].astype(np.float64) + self.codes.astype(np.float64) * step

    @property
    def offset_sum(self) -> float:
        """Sum of the per-tree offsets."""
        return float(np.sum(self.offsets.astype(np.float64)))


def leaf_grid(
    leaves: NDArray[np.floating], bits: int, chunk: int, stage_starts: tuple[int, ...] = (0,)
) -> LeafGrid:
    """Quantise ``(n_trees, n_leaves)`` leaves onto the grid :class:`LeafGrid` describes.

    Each tree's offset is its smallest leaf; each chunk's step is the smallest power
    of two that spans the widest tree of the chunk with ``2**bits - 1`` codes.  Leaf
    magnitudes shrink as boosting proceeds, so one step for the whole model would
    turn the later trees into noise; a step per chunk keeps them resolved.
    """
    values = np.asarray(leaves, dtype=np.float64)
    if values.ndim != _LEAF_TABLE_NDIM:
        msg = "leaves must be (n_trees, n_leaves)"
        raise ValueError(msg)
    n_trees = len(values)
    levels = float((1 << bits) - 1)
    offsets = values.min(axis=1)
    ranges = values.max(axis=1) - offsets
    starts = [*stage_starts, n_trees]
    exponents: list[int] = []
    chunk_of: list[int] = []
    for stage in range(len(starts) - 1):
        for lo in range(starts[stage], starts[stage + 1], chunk):
            hi = min(lo + chunk, starts[stage + 1])
            widest = float(ranges[lo:hi].max())
            exponent = (
                math.ceil(math.log2(widest / levels)) if widest > 0.0 else _EMPTY_CHUNK_EXPONENT
            )
            chunk_of.extend([len(exponents)] * (hi - lo))
            exponents.append(exponent)
    exp_arr = np.asarray(exponents, dtype=np.int64)
    steps = np.ldexp(1.0, exp_arr[np.asarray(chunk_of, dtype=np.int64)])
    codes = np.rint((values - offsets[:, None]) / steps[:, None])
    codes = np.clip(codes, 0, levels).astype(np.uint8)
    return LeafGrid(
        bits=bits,
        chunk=chunk,
        stage_starts=tuple(stage_starts),
        offsets=offsets.astype(np.float32),
        exponents=exp_arr,
        codes=codes,
    )


def quantise_leaves(
    leaves: NDArray[np.floating], bits: int, chunk: int, stage_starts: tuple[int, ...] = (0,)
) -> NDArray[np.float64]:
    """Leaf values on the low-bit grid (see :func:`leaf_grid`), same shape as ``leaves``."""
    grid = leaf_grid(leaves, bits, chunk, stage_starts)
    # the float32 offset is what the blob stores, so it is what training fits against
    return grid.values()


def _leaf_table(booster: CatBoostClassifier, n_trees: int) -> NDArray[np.float64]:
    """Leaves as ``(n_trees, max_leaves)``; short trees padded with their last leaf.

    Padding only fills positions no leaf index can reach, and repeats a real value,
    so it changes neither predictions nor a tree's leaf range.
    """
    flat = np.asarray(booster.get_leaf_values(), dtype=np.float64)
    counts = list(booster.get_tree_leaf_counts())
    table = np.zeros((n_trees, max(counts)), dtype=np.float64)
    at = 0
    for tree, count in enumerate(counts):
        table[tree, :count] = flat[at : at + count]
        table[tree, count:] = flat[at + count - 1]
        at += count
    return table


def resolve_task_type(task_type: str) -> str:
    """``AUTO`` -> ``GPU`` when CatBoost can see a CUDA device, else ``CPU``."""
    if task_type != "AUTO":
        return task_type
    try:
        from catboost.utils import get_gpu_device_count  # noqa: PLC0415 -- optional heavy import

        return "GPU" if get_gpu_device_count() > 0 else "CPU"
    except Exception:  # noqa: BLE001 -- any failure to probe means no usable GPU
        return "CPU"


def make_pool(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
    input_borders: str | Path | None = None,
) -> Pool:
    """The training set as a CatBoost ``Pool`` quantised once (``border_count`` cuts per column).

    Quantising up front turns the ``n_cells x n_features`` float32 matrix into 4-bit
    bins, so the float matrix can be released and every stage and chunk of the fit
    skips its own border search; the borders are the same the fit would find.
    ``input_borders`` (a file from ``Pool.save_quantization_borders``) quantises with
    those cut points instead of searching for new ones.
    """
    from catboost import Pool  # noqa: PLC0415 -- optional heavy import

    pool = Pool(features, labels)
    if input_borders is None:
        pool.quantize(border_count=int(model_cfg.border_count))
    else:
        pool.quantize(border_count=int(model_cfg.border_count), input_borders=str(input_borders))
    return pool


def plan_blocks(n_cells: int, width: int, block_gib: float | None) -> int:
    """Blocks needed to keep each block's float32 matrix within ``block_gib`` GiB (1 = no blocking)."""
    if block_gib is None or block_gib <= 0.0 or n_cells <= 0:
        return 1
    gib = n_cells * width * 4 / 2**30
    return max(1, math.ceil(gib / block_gib))


# -- sizing blocks from the memory that is free right now ---------------------------------------
# Peak host RAM while a pool is built, in float32 matrices: the matrix, CatBoost's Pool copy of
# it (+1.0x) and the bins (+0.3x), measured in README "Training time and memory".  If the
# Fortran-order layout docs/OPTIMIZATION.md describes lands, halve this.
_HOST_PEAK = 2.3
_HOST_USE = 0.8  # share of the free host RAM a fit may plan on
_GPU_USE = 0.4  # share of the free GPU memory the quantised pool may take (rest: histograms etc.)
_FLOAT32_BITS = 32
_NVIDIA_SMI_WINDOWS = (  # where the driver puts nvidia-smi.exe when it is not on PATH
    ("SystemRoot", r"System32\nvidia-smi.exe"),
    ("ProgramFiles", r"NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
)


def bin_bits(border_count: int) -> int:
    """Bits CatBoost's quantised pool spends per value for ``border_count`` cuts (15 -> 4)."""
    return max(1, math.ceil(math.log2(border_count + 1)))


def _cgroup_headroom_gib(limit_file: str, used_file: str) -> float | None:
    """Headroom under one cgroup memory limit, or ``None`` if there is none (or no cgroup)."""
    try:
        limit = Path(limit_file).read_text(encoding="ascii").strip()
        used = int(Path(used_file).read_text(encoding="ascii").strip())
        if limit == "max" or int(limit) >= 1 << 60:
            return None
        return max(0.0, (int(limit) - used) / 2**30)
    except (OSError, ValueError):
        return None


def _cgroup_available_gib() -> float | None:
    """Headroom under a container's memory limit (cgroup v2, then v1), or ``None`` if unlimited."""
    v2 = _cgroup_headroom_gib("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")
    v1 = _cgroup_headroom_gib(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )
    found = [v for v in (v2, v1) if v is not None]
    return min(found) if found else None


def _meminfo_available_gib() -> float | None:
    """Linux: the kernel's ``MemAvailable`` estimate from ``/proc/meminfo``."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20  # kB -> GiB
    except (OSError, ValueError, IndexError):
        return None
    return None


def _sysconf_available_gib() -> float | None:
    """POSIX: free physical pages (Linux without ``/proc``, the BSDs; not macOS)."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, ValueError, OSError):
        return None


def _vm_stat_available_gib() -> float | None:
    """macOS: free + inactive + speculative pages from ``vm_stat`` (there is no MemAvailable)."""
    exe = shutil.which("vm_stat")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 -- fixed arguments, no shell
            [exe], capture_output=True, text=True, timeout=15, check=False
        ).stdout
        header, *lines = out.splitlines()
        page = int(header.split("page size of")[1].split()[0])
        pages = 0
        for line in lines:
            key, _, value = line.partition(":")
            if key.strip() in {"Pages free", "Pages inactive", "Pages speculative"}:
                pages += int(value.strip().rstrip("."))
        return pages * page / 2**30
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _windows_available_gib() -> float | None:  # pragma: no cover -- Windows only
    """Windows: ``ullAvailPhys`` from ``GlobalMemoryStatusEx``."""
    import ctypes  # noqa: PLC0415 -- Windows only

    class _MemStatus(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        )

    status = _MemStatus()
    status.length = ctypes.sizeof(_MemStatus)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.avail_phys) / 2**30


def host_available_gib() -> float | None:
    """GiB of host RAM a process can still take right now, or ``None`` if it cannot be read.

    The smaller of the OS's "available" figure (Linux ``/proc/meminfo``, else free pages from
    ``sysconf``, macOS ``vm_stat``, Windows ``GlobalMemoryStatusEx``) and the headroom under a
    container (cgroup) limit.
    """
    available: float | None = None
    if sys.platform == "win32":  # pragma: no cover -- Windows only
        available = _windows_available_gib()
    elif sys.platform == "darwin":  # pragma: no cover -- macOS only
        available = _vm_stat_available_gib()
    else:
        available = _meminfo_available_gib()
        if available is None:
            available = _sysconf_available_gib()
    capped = _cgroup_available_gib()
    if available is None:
        return capped
    return available if capped is None else min(available, capped)


def _nvidia_smi() -> str | None:
    """Path of ``nvidia-smi``: on PATH, else where the Windows driver installs it."""
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":  # pragma: no cover -- Windows only
        for var, tail in _NVIDIA_SMI_WINDOWS:
            root = os.environ.get(var)
            if root and (Path(root) / tail).is_file():
                return str(Path(root) / tail)
    return exe


def cuda_device_id(device: int = 0, visible: str | None = None) -> str:
    """What ``nvidia-smi -i`` should be given for CUDA device ``device``.

    CatBoost numbers devices the way CUDA does, after ``CUDA_VISIBLE_DEVICES`` (``visible``,
    read from the environment when ``None``) has been applied; ``nvidia-smi`` numbers the
    physical GPUs.  The entry is an index or a ``GPU-``/``MIG-`` UUID, both of which
    ``nvidia-smi -i`` accepts.
    """
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return str(device)
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if device >= len(entries):
        return str(device)  # CUDA would see no such device; let nvidia-smi say so
    return entries[device]


def gpu_free_gib(device: int = 0) -> float | None:
    """GiB of free memory on CUDA device ``device`` right now (``nvidia-smi``), or ``None``."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 -- fixed arguments, no shell
            [
                exe,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                cuda_device_id(device),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return float(out.stdout.strip().splitlines()[0]) / 1024  # MiB -> GiB
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def auto_block_gib(
    host_gib: float | None, gpu_gib: float | None, *, gpu_bits: int = 4
) -> tuple[float | None, str]:
    """Largest float32 block (GiB) the free memory allows, and why; ``None`` = nothing to go on.

    Host RAM: building a pool peaks at ``_HOST_PEAK`` times its float32 matrix, so a block may be
    ``_HOST_USE * host_gib / _HOST_PEAK``.  GPU (when training on it): the quantised pool,
    ``gpu_bits`` a value (:func:`bin_bits` of ``border_count``), may take ``_GPU_USE`` of the
    free GPU memory.  The smaller bound wins.
    """
    bounds: list[tuple[float, str]] = []
    if host_gib is not None:
        limit = _HOST_USE * host_gib / _HOST_PEAK
        bounds.append((limit, f"{host_gib:.1f} GiB host RAM free -> blocks <= {limit:.1f} GiB"))
    if gpu_gib is not None:
        limit = _GPU_USE * gpu_gib * _FLOAT32_BITS / gpu_bits
        bounds.append((limit, f"{gpu_gib:.1f} GiB GPU memory free -> blocks <= {limit:.1f} GiB"))
    if not bounds:
        return None, "free memory could not be read"
    best = min(bounds, key=lambda b: b[0])
    return best[0], "; ".join(why for _, why in bounds)


def unused_blocks(model_cfg: ModelConfig, n_blocks: int) -> int:
    """Blocks of ``n_blocks`` that no chunk of the fit would start in (their cells go unused).

    The trees are laid out evenly over the blocks and a chunk is fitted on the block its
    first tree falls in (see :class:`_PoolFeed`), so a block narrower than a chunk can be
    skipped.  Chunks are ``leaf_chunk`` trees, restarting at the coarse/fine boundary, when
    the fit is quantisation-aware; otherwise one per block, so every block is used as long as
    it holds a tree.
    """
    n_trees = model_cfg.n_trees
    if n_blocks <= 1 or n_trees <= 0:
        return 0
    layout = _PoolFeed(n_blocks=n_blocks, total_trees=n_trees)
    if not model_cfg.quantisation_aware:
        return max(0, n_blocks - n_trees)
    coarse = model_cfg.coarse_trees if model_cfg.coarse_trees < n_trees else 0
    starts = {
        *range(0, coarse, model_cfg.leaf_chunk),
        *range(coarse, n_trees, model_cfg.leaf_chunk),
    }
    return n_blocks - len({layout.block_of(s) for s in starts})


class _PoolFeed:
    """The pool boosting trains on, and (optionally) the blocks of cells it moves through.

    *Fixed* (``_PoolFeed(pool)``): one quantised pool for the whole fit, as ``fit_booster``
    has always done.

    *Blocked* (``_PoolFeed(provider=..., ...)``): the cells arrive in ``n_blocks`` blocks and
    block ``b`` serves the chunks that start at tree ``b * total_trees // n_blocks`` on, so the
    float matrix of only one block exists at a time and the whole training set is never in
    memory.  Every block is quantised with the cut points of the first, so all the trees share
    one set of ``border_count`` borders per column (the export format has a 4-bit bin).  A
    block is a fresh sample, so it starts from the score of every tree fitted so far (the
    quantised leaves, when ``bits`` is set) as its CatBoost baseline: the trees of the next
    block fit what the earlier ones have not explained, on cells they never saw.
    """

    def __init__(  # noqa: PLR0913 -- one fixed-pool argument plus the blocked-mode settings
        self,
        pool: Pool | None = None,
        *,
        provider: Callable[[int], tuple[NDArray[np.floating], NDArray[np.bool_]]] | None = None,
        n_blocks: int = 1,
        total_trees: int = 0,
        model_cfg: ModelConfig | None = None,
        borders: Path | None = None,
    ) -> None:
        """Hold ``pool`` (fixed), or build blocks on demand from ``provider`` (blocked)."""
        self._pool = pool
        self.provider = provider
        self.n_blocks = n_blocks
        self.total_trees = total_trees
        self.model_cfg = model_cfg
        self.borders = borders
        self.block = -1
        self.history: list[tuple[CatBoostClassifier, NDArray[np.float64]]] = []

    @property
    def blocked(self) -> bool:
        """Whether the cells arrive block by block."""
        return self.provider is not None

    @property
    def pool(self) -> Pool:
        """The pool of the current block (the only pool, when fixed)."""
        if self._pool is None:
            msg = "no block is loaded yet"
            raise RuntimeError(msg)
        return self._pool

    def block_of(self, tree: int) -> int:
        """The block that serves a chunk starting at global tree index ``tree``."""
        return min(self.n_blocks - 1, tree * self.n_blocks // max(self.total_trees, 1))

    def trees_left(self, tree: int) -> int:
        """Trees from ``tree`` to the end of its block (unbounded when fixed)."""
        if not self.blocked:
            return 1 << 62
        following = self.block_of(tree) + 1
        if following >= self.n_blocks:
            return self.total_trees - tree
        return -(-following * self.total_trees // self.n_blocks) - tree

    def start_stage(self, score: NDArray[np.float64]) -> None:
        """Put the running score on the pool as its baseline (a blocked feed does so on load)."""
        if not self.blocked:
            self.pool.set_baseline(np.asarray(score, dtype=np.float64).reshape(-1, 1))

    def advance(self, tree: int, score: NDArray[np.float64]) -> NDArray[np.float64]:
        """Make the pool current for a chunk starting at ``tree``; return the score on its cells."""
        block = self.block_of(tree)
        if not self.blocked or block == self.block:
            return score
        assert self.provider is not None  # noqa: S101 -- narrows the type; blocked implies it
        assert self.model_cfg is not None  # noqa: S101
        self._pool = None  # free the last block (float copy and bins) before the next is built
        features, labels = self.provider(block)
        first = self.block < 0
        pool = make_pool(
            features, labels, self.model_cfg, input_borders=None if first else self.borders
        )
        del features
        if first and self.borders is not None:
            pool.save_quantization_borders(str(self.borders))
        self._pool, self.block = pool, block
        score = np.zeros(pool.num_row(), dtype=np.float64)
        for model, table in self.history:
            size = table.shape[0]
            indexes = model.calc_leaf_indexes(pool)
            scale, bias = model.get_scale_and_bias()
            score += scale * table[np.arange(size)[None, :], indexes].sum(axis=1) + bias
        pool.set_baseline(score.reshape(-1, 1))
        return score

    def record(self, model: CatBoostClassifier, table: NDArray[np.float64]) -> None:
        """Remember a fitted chunk and its leaves so the next block can be scored with them."""
        if self.blocked:
            self.history.append((model, table))


def _fit_stage(  # noqa: PLR0913 -- a private helper with one call site per stage
    feed: _PoolFeed,
    params: Mapping[str, Any],
    *,
    n_trees: int,
    chunk: int,
    bits: int | None,
    running: NDArray[np.float64],
    ignored: list[int] | None,
    first_tree: int = 0,
) -> tuple[list[CatBoostClassifier], NDArray[np.float64]]:
    """Fit ``n_trees`` on top of ``running``; return the models and the new score.

    With ``bits`` the stage is fitted in chunks of ``chunk`` trees and each chunk's
    leaves are put on the grid before the next chunk starts, so later trees see
    -- and correct -- the rounding of earlier ones.  The grid is the one
    :func:`leaf_grid` builds for this stage on export (chunks counted from the
    stage's first tree), so the fitted model is the shipped model.  Without
    ``bits`` the stage is one fit (one per block, when the feed is blocked).
    ``ignored`` hides feature columns from this stage (resolution tiering).
    ``first_tree`` is the stage's first tree among all the model's, which the blocks
    are laid out over.
    """
    stage = dict(params)
    if ignored:
        stage["ignored_features"] = list(ignored)
    models: list[CatBoostClassifier] = []
    score = running
    feed.start_stage(score)
    start = 0
    while start < n_trees:
        tree = first_tree + start
        score = feed.advance(tree, score)
        size = min(n_trees - start, chunk if bits is not None else feed.trees_left(tree))
        fitted = CatBoostClassifier(iterations=size, boost_from_average=False, **stage)
        fitted.fit(feed.pool)
        table = _leaf_table(fitted, size)
        if bits is not None:
            table = quantise_leaves(table, bits, chunk)
        indexes = fitted.calc_leaf_indexes(feed.pool)
        scale, bias = fitted.get_scale_and_bias()
        score = score + scale * table[np.arange(size)[None, :], indexes].sum(axis=1) + bias
        feed.pool.set_baseline(np.asarray(score, dtype=np.float64).reshape(-1, 1))
        feed.record(fitted, table)
        models.append(fitted)
        start += size
    return models, score


def _booster_params(model_cfg: ModelConfig) -> dict[str, Any]:
    """The CatBoost parameters every fit of the frozen booster shares."""
    params: dict[str, Any] = {
        "loss_function": model_cfg.loss_function,
        "depth": model_cfg.depth,
        "learning_rate": model_cfg.learning_rate,
        "grow_policy": model_cfg.grow_policy,
        "random_seed": model_cfg.random_seed,
        "border_count": model_cfg.border_count,
        **(
            {"scale_pos_weight": model_cfg.scale_pos_weight}
            if model_cfg.scale_pos_weight is not None
            else {}
        ),
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "task_type": resolve_task_type(model_cfg.task_type),
    }
    if params["task_type"] == "GPU":
        params["devices"] = "0"
    return params


def _fit_tiers(
    feed: _PoolFeed,
    params: Mapping[str, Any],
    model_cfg: ModelConfig,
    feature_sides: list[int] | None,
    n_cells: int,
) -> CatBoostClassifier:
    """Fit the coarse then the fine stage on ``feed`` and return the summed booster."""
    from catboost import sum_models  # noqa: PLC0415 -- optional heavy import

    bits = model_cfg.leaf_bits if model_cfg.quantisation_aware else None
    coarse_trees = model_cfg.coarse_trees if model_cfg.coarse_trees < model_cfg.n_trees else 0
    models: list[CatBoostClassifier] = []
    score = np.zeros(n_cells, dtype=np.float64)
    if coarse_trees:
        if feature_sides is None:
            msg = "coarse_trees needs feature_sides (the pyramid side of each column)"
            raise ValueError(msg)
        ignored = [i for i, side in enumerate(feature_sides) if side > model_cfg.coarse_max_side]
        if len(ignored) == len(feature_sides):
            msg = f"no column has side <= coarse_max_side={model_cfg.coarse_max_side}"
            raise ValueError(msg)
        coarse, score = _fit_stage(
            feed,
            params,
            n_trees=coarse_trees,
            chunk=model_cfg.leaf_chunk,
            bits=bits,
            running=score,
            ignored=ignored,
        )
        models += coarse
    fine, _score = _fit_stage(
        feed,
        params,
        n_trees=model_cfg.n_trees - coarse_trees,
        chunk=model_cfg.leaf_chunk,
        bits=bits,
        running=score,
        ignored=None,
        first_tree=coarse_trees,
    )
    models += fine
    return models[0] if len(models) == 1 else sum_models(models)


def fit_booster(
    features: NDArray[np.floating],
    labels: NDArray[np.bool_] | NDArray[np.integer[Any]],
    model_cfg: ModelConfig,
    feature_sides: list[int] | None = None,
) -> CatBoostClassifier:
    """Fit the frozen symmetric CatBoost booster and return it.

    Three knobs change how the trees are grown, and they compose:

    * ``quantisation_aware`` fits in chunks against the quantised running score,
      so the model is trained for the leaf grid the exporter writes;
    * ``coarse_trees`` grows the first trees on tile-constant features only
      (``feature_sides`` says which), leaving the rest to the fine stage;
    * neither set, a single ordinary fit -- unchanged from before.
    """
    params = _booster_params(model_cfg)
    bits = model_cfg.leaf_bits if model_cfg.quantisation_aware else None
    coarse_trees = model_cfg.coarse_trees if model_cfg.coarse_trees < model_cfg.n_trees else 0
    n_cells = len(labels)
    pool = make_pool(features, labels, model_cfg)
    del features  # the quantised pool is the training set from here on (4 bits per value)
    if bits is None and not coarse_trees:
        booster = CatBoostClassifier(iterations=model_cfg.n_trees, **params)
        booster.fit(pool)
        return booster
    return _fit_tiers(_PoolFeed(pool), params, model_cfg, feature_sides, n_cells)


def fit_booster_blocks(
    provider: Callable[[int], tuple[NDArray[np.floating], NDArray[np.bool_]]],
    n_blocks: int,
    model_cfg: ModelConfig,
    feature_sides: list[int] | None = None,
) -> CatBoostClassifier:
    """Fit the booster on more cells than fit in memory, one block of them at a time.

    ``provider(b)`` returns the ``(features, labels)`` of block ``b``; it is called once
    per block, in order, and the block is released before the next one is built, so peak
    memory is one block (about 2.3x its float32 size while the pool is built) however many
    blocks there are.  Blocks must be random samples of the same cells, each with both
    classes: block 0 also fixes the cut points every later block is quantised with.

    This is out-of-core boosting, not the same fit as one pool of every cell: the trees
    are laid out evenly over the blocks (block ``b`` serves trees ``b * n_trees // n_blocks``
    on), each chunk is fitted on one block with the score of all earlier trees as its
    baseline, and a block no chunk starts in is never used (``n_trees`` should be at least
    ``n_blocks`` chunks of ``leaf_chunk``).  With ``n_blocks == 1`` it is :func:`fit_booster`.
    """
    if n_blocks < 1:
        msg = "n_blocks must be >= 1"
        raise ValueError(msg)
    params = _booster_params(model_cfg)
    with tempfile.TemporaryDirectory() as tmp:
        feed = _PoolFeed(
            provider=provider,
            n_blocks=n_blocks,
            total_trees=model_cfg.n_trees,
            model_cfg=model_cfg,
            borders=Path(tmp) / "borders.tsv",
        )
        return _fit_tiers(feed, params, model_cfg, feature_sides, 0)


def build_ranking(booster: CatBoostClassifier, feature_names: Sequence[str]) -> dict[str, Any]:
    """Gain-importance ranking of a full-width fit, names in descending order.

    ``feature_names`` are the columns the booster was fitted on, in fitting order
    (CatBoost reports importances by position).  Fit with ``top_k_features=0`` so
    the ranking covers every front-end column; that is what :func:`select_columns`
    later indexes into.
    """
    gains = np.asarray(booster.get_feature_importance(type="FeatureImportance"), dtype=float)
    if len(gains) != len(feature_names):
        msg = f"{len(gains)} importances for {len(feature_names)} feature names"
        raise ValueError(msg)
    order = np.argsort(-gains, kind="stable")
    return {
        "names": [feature_names[int(i)] for i in order],
        "gains": [float(gains[int(i)]) for i in order],
        "width": len(feature_names),
    }


def stage_tree_counts(model_cfg: ModelConfig) -> list[int]:
    """Tree counts at which the scorer may stop tiles.

    The coarse/fine boundary and ``exit_stage_fractions`` of the fine tier, each
    rounded to a chunk boundary of the fine stage so a chunked scorer can stop there.
    """
    coarse, n_trees, chunk = model_cfg.coarse_trees, model_cfg.n_trees, model_cfg.leaf_chunk
    counts: list[int] = []
    for frac in model_cfg.exit_stage_fractions:
        raw = coarse + frac * (n_trees - coarse)
        # stay on a chunk boundary of the fine stage so a chunked scorer can stop there
        trees = coarse + round((raw - coarse) / chunk) * chunk
        if 0 < trees < n_trees and trees not in counts:
            counts.append(trees)
    return sorted(counts)


def calibrate_exit_stages(
    partials: NDArray[np.float64],
    finals: NDArray[np.float64],
    stages: Sequence[int],
    *,
    keep_prob: float,
    margin: float,
) -> list[tuple[int, float]]:
    """Per-stage thresholds from the training cells.

    ``partials`` is ``(n_stages, n_cells)`` raw scores after each stage's trees and
    ``finals`` the ``(n_cells,)`` raw scores after every tree.  A stage's threshold is
    the lowest partial score of any cell whose final probability reaches
    ``keep_prob``, minus ``margin``: on the calibration cells no such cell can be
    stopped, and the margin covers what unseen images move.
    """
    keep = finals >= float(np.log(keep_prob / (1.0 - keep_prob)))
    out: list[tuple[int, float]] = []
    for i, trees in enumerate(stages):
        theta = float(partials[i][keep].min()) - margin if keep.any() else -1e30
        out.append((int(trees), theta))
    return out
