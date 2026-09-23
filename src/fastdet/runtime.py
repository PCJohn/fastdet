"""Python reader/scorer for the ``IMSY`` tree blob (mirror of the C++ runtime).

Blob layout (little-endian), magic ``IMSY``, version 4::

    +0    4 bytes  magic "IMSY"
    +4    uint32   version (4)
    +8    uint32   n_trees
    +12   uint32   n_features
    +16   uint32   n_leafs_total
    +20   uint32   depth
    +24   uint32   leaf_bits (4 or 8)
    +28   uint32   leaf_chunk           trees per quantisation chunk (and scorer pass)
    +32   uint32   coarse_trees         the first coarse_trees trees split only on tile-constant features
    +36   int32    e_min                smallest chunk exponent
    +40   uint32   n_chunks
    +44   uint32   n_stages
    +48   uint32   tree_offsets[n_trees + 1]   leaf-code start per tree
          uint32   tree_base[n_trees + 1]      split-byte offset per tree
          split[n_trees * depth]                u16 feature, u8 bin, u8 pad
          uint8    codes[n_leafs_total]         leaf code, < 2**leaf_bits
          float32  offsets[n_trees]             per tree
          uint8    shifts[n_chunks]             chunk step = 2**(e_min + shift)
          stage[n_stages]                       u32 trees, f32 threshold (raw score)
          uint32   n_borders[n_features]
          float32  borders[sum(n_borders)]
          uint8    level_shift[n_features]      2*log2(64/level); 12 = one value per image

A leaf's value is ``offsets[tree] + codes[leaf] * 2**(e_min + shifts[chunk(tree)])``;
chunks are ``leaf_chunk`` consecutive trees, restarting at ``coarse_trees``.  The raw
score of a cell is ``sum(offsets) + total * 2**e_min`` where ``total`` is the integer
sum of shifted codes, so every runtime reaches the same bits.

A split at level ``d`` (0 = root) sets leaf bit ``d`` when the byte-space bin of
its feature is strictly greater than the split's bin; the root split is the LOWEST
bit (CatBoost oblivious encoding).

Early exit: after ``trees`` trees of a stage, a 4x4 tile of the 64x64 grid whose
cells all have raw scores below the stage's threshold keeps its partial score.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["BLOB_VERSION", "GRID_CELLS", "IMSY_MAGIC", "ImysModel", "parse_blob"]

IMSY_MAGIC = b"IMSY"
BLOB_VERSION = 4
GRID_SIDE = 64
GRID_CELLS = GRID_SIDE * GRID_SIDE
TILE = 4
_MATRIX_NDIM = 2
_HEADER_BYTES = 48

ByteBins = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt32Array = NDArray[np.uint32]
VoidArray = NDArray[np.void]

_SPLIT_DTYPE = np.dtype([("feat", "<u2"), ("bin", "u1"), ("pad", "u1")])


@dataclass
class ImysModel:
    """Parsed ``IMSY`` blob with a NumPy scoring path."""

    version: int
    n_trees: int
    n_features: int
    n_leafs_total: int
    depth: int
    leaf_bits: int
    leaf_chunk: int
    coarse_trees: int
    e_min: int
    tree_offsets: UInt32Array
    tree_base: UInt32Array
    splits: VoidArray
    codes: ByteBins
    offsets: Float32Array
    shifts: ByteBins
    exit_stages: list[tuple[int, float]]
    n_borders: UInt32Array
    borders: list[Float32Array]
    level_shift: ByteBins

    # -- derived -------------------------------------------------------------
    @property
    def chunk_of_tree(self) -> Int64Array:
        """Quantisation chunk of every tree (chunks restart at the coarse/fine boundary)."""
        out = np.zeros(self.n_trees, dtype=np.int64)
        starts = (
            [0, self.coarse_trees, self.n_trees]
            if 0 < self.coarse_trees < self.n_trees
            else [0, self.n_trees]
        )
        base = 0
        for i in range(len(starts) - 1):
            lo, hi = starts[i], starts[i + 1]
            out[lo:hi] = base + np.arange(hi - lo) // self.leaf_chunk
            base += -(-(hi - lo) // self.leaf_chunk)
        return out

    @property
    def offset_sum(self) -> float:
        """Sum of the per-tree offsets: the constant part of every cell's raw score."""
        return float(np.sum(self.offsets.astype(np.float64)))

    @property
    def leaf_values(self) -> Float64Array:
        """Every leaf's value on the grid, ``(n_leafs_total,)`` float64."""
        n_leaves = 1 << self.depth
        steps = np.ldexp(1.0, self.e_min + self.shifts[self.chunk_of_tree].astype(np.int64))
        codes = self.codes.reshape(self.n_trees, n_leaves).astype(np.float64)
        values: Float64Array = (
            self.offsets[:, None].astype(np.float64) + codes * steps[:, None]
        ).reshape(-1)
        return values

    # -- scoring ---------------------------------------------------------------
    def bins(self, x: NDArray[np.floating]) -> ByteBins:
        """Map float features to byte bins: ``searchsorted-left(borders[f], x)``."""
        x_arr: Float32Array = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != _MATRIX_NDIM or x_arr.shape[1] != self.n_features:
            msg = f"expected (n_cells, {self.n_features}) features, got {x_arr.shape}"
            raise ValueError(msg)
        bin_matrix: ByteBins = np.empty((x_arr.shape[0], self.n_features), dtype=np.uint8)
        for f in range(self.n_features):
            bin_matrix[:, f] = np.searchsorted(self.borders[f], x_arr[:, f], side="left")
        return bin_matrix

    def _leaf_indexes(self, bin_matrix: ByteBins, tree: int) -> Int64Array:
        idx = np.zeros(bin_matrix.shape[0], dtype=np.int64)
        off0 = tree * self.depth
        for d in range(self.depth):
            split = self.splits[off0 + d]
            idx |= (bin_matrix[:, split["feat"]] > split["bin"]).astype(np.int64) << d
        return idx

    def raw_scores(self, bin_matrix: ByteBins, *, use_exit: bool = False) -> Float64Array:
        """Raw (pre-sigmoid) scores from precomputed byte bins.

        Integer arithmetic on the codes, so the result is the C++ runtime's to the
        bit.  With ``use_exit`` the blob's stages apply: the cells are the 64x64
        grid in row-major order and a 4x4 tile whose cells all score below a stage's
        threshold stops accumulating (its cells keep their partial score).
        """
        n_cells = bin_matrix.shape[0]
        total = np.zeros(n_cells, dtype=np.int64)
        alive = np.ones(n_cells, dtype=bool)
        chunk_of = self.chunk_of_tree
        stages = dict(self.exit_stages) if use_exit else {}
        if use_exit and n_cells != GRID_CELLS:
            msg = f"early exit needs the full {GRID_SIDE}x{GRID_SIDE} grid ({GRID_CELLS} cells), got {n_cells}"
            raise ValueError(msg)
        offset_done = 0.0
        for t in range(self.n_trees):
            idx = self._leaf_indexes(bin_matrix, t)
            code = self.codes[int(self.tree_offsets[t]) + idx].astype(np.int64)
            shift = int(self.shifts[chunk_of[t]])
            total += np.where(alive, code << shift, 0)
            offset_done += float(self.offsets[t])
            if (t + 1) in stages:
                partial = offset_done + np.ldexp(total.astype(np.float64), self.e_min)
                tiles = partial.reshape(GRID_SIDE // TILE, TILE, GRID_SIDE // TILE, TILE).max(
                    axis=(1, 3)
                )
                keep = np.repeat(
                    np.repeat(tiles >= stages[t + 1], TILE, axis=0), TILE, axis=1
                ).reshape(-1)
                alive &= keep
        raw: Float64Array = self.offset_sum + np.ldexp(total.astype(np.float64), self.e_min)
        return raw

    def partial_scores(self, bin_matrix: ByteBins, stages: list[int]) -> Float64Array:
        """Raw scores after each of ``stages`` trees, ``(len(stages), n_cells)`` -- for calibration."""
        n_cells = bin_matrix.shape[0]
        total = np.zeros(n_cells, dtype=np.int64)
        chunk_of = self.chunk_of_tree
        out = np.zeros((len(stages), n_cells), dtype=np.float64)
        want = {trees: i for i, trees in enumerate(stages)}
        offset_done = 0.0
        for t in range(self.n_trees):
            idx = self._leaf_indexes(bin_matrix, t)
            total += self.codes[int(self.tree_offsets[t]) + idx].astype(np.int64) << int(
                self.shifts[chunk_of[t]]
            )
            offset_done += float(self.offsets[t])
            if (t + 1) in want:
                out[want[t + 1]] = offset_done + np.ldexp(total.astype(np.float64), self.e_min)
        return out

    def predict_proba(self, x: NDArray[np.floating], *, use_exit: bool = False) -> Float64Array:
        """Positive-class probability for each row of ``x``."""
        raw = self.raw_scores(self.bins(x), use_exit=use_exit)
        probabilities: Float64Array = 1.0 / (1.0 + np.exp(-raw))
        return probabilities

    def predict_grid(
        self, x: NDArray[np.floating], grid: int = GRID_SIDE, *, use_exit: bool = False
    ) -> Float32Array:
        """Positive-class probability reshaped to a ``grid x grid`` map."""
        grid_map: Float32Array = (
            self.predict_proba(x, use_exit=use_exit).reshape(grid, grid).astype(np.float32)
        )
        return grid_map


def _u32(blob: bytes, offset: int, count: int) -> UInt32Array:
    """Read a little-endian ``uint32`` array from ``blob`` (zero-copy view)."""
    return np.frombuffer(blob, dtype="<u4", count=count, offset=offset)


def parse_blob(blob: bytes) -> ImysModel:
    """Parse an ``IMSY`` blob into an :class:`ImysModel`."""
    if blob[:4] != IMSY_MAGIC:
        msg = f"not an IMSY blob (magic {blob[:4]!r})"
        raise ValueError(msg)
    fields = struct.unpack_from("<IIIIIIIIiII", blob, 4)
    version, n_trees, n_features, n_leafs_total, depth = (int(v) for v in fields[:5])
    leaf_bits, leaf_chunk, coarse_trees, e_min, n_chunks, n_stages = (int(v) for v in fields[5:])
    if version != BLOB_VERSION:
        msg = f"unsupported IMSY version {version} (this reader is version {BLOB_VERSION})"
        raise ValueError(msg)
    off = _HEADER_BYTES
    tree_offsets = _u32(blob, off, n_trees + 1)
    off += 4 * (n_trees + 1)
    tree_base = _u32(blob, off, n_trees + 1)
    off += 4 * (n_trees + 1)
    n_splits = n_trees * depth
    splits = np.frombuffer(blob, dtype=_SPLIT_DTYPE, count=n_splits, offset=off)
    off += n_splits * 4
    codes: ByteBins = np.frombuffer(blob, dtype="u1", count=n_leafs_total, offset=off).copy()
    off += n_leafs_total
    offsets: Float32Array = np.frombuffer(blob, dtype="<f4", count=n_trees, offset=off).copy()
    off += 4 * n_trees
    shifts: ByteBins = np.frombuffer(blob, dtype="u1", count=n_chunks, offset=off).copy()
    off += n_chunks
    exit_stages: list[tuple[int, float]] = []
    for _ in range(n_stages):
        trees, theta = struct.unpack_from("<If", blob, off)
        exit_stages.append((int(trees), float(theta)))
        off += 8
    n_borders = _u32(blob, off, n_features)
    off += n_features * 4
    borders: list[Float32Array] = []
    for f in range(n_features):
        nb = int(n_borders[f])
        borders.append(np.frombuffer(blob, dtype="<f4", count=nb, offset=off).copy())
        off += nb * 4
    level_shift: ByteBins = np.frombuffer(blob, dtype="u1", count=n_features, offset=off).copy()
    off += n_features
    if off != len(blob):
        msg = f"IMSY blob has {len(blob) - off} trailing bytes"
        raise ValueError(msg)
    return ImysModel(
        version=version,
        n_trees=n_trees,
        n_features=n_features,
        n_leafs_total=n_leafs_total,
        depth=depth,
        leaf_bits=leaf_bits,
        leaf_chunk=leaf_chunk,
        coarse_trees=coarse_trees,
        e_min=e_min,
        tree_offsets=tree_offsets,
        tree_base=tree_base,
        splits=splits,
        codes=codes,
        offsets=offsets,
        shifts=shifts,
        exit_stages=exit_stages,
        n_borders=n_borders,
        borders=borders,
        level_shift=level_shift,
    )
