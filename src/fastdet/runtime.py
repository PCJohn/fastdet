"""Python reader/scorer for the ``IMSY`` tree blob (mirror of the C++ runtime).

Blob layout (little-endian), magic ``IMSY``::

    +0    4 bytes  magic "IMSY"
    +4    uint32   version (2 or 3)
    +8    uint32   n_trees
    +12   uint32   n_features
    +16   uint32   n_leafs_total
    +20   uint32   depth
    +24   uint32   tree_offsets[n_trees + 1]   leaf-value start per tree
          uint32   tree_base[n_trees + 1]      split-byte offset per tree
          split[n_trees * depth]                u16 feature, u8 bin, u8 pad
          float32  leaf_values[n_leafs_total]
          uint32   n_borders[n_features]
          float32  borders[sum(n_borders)]
          uint8    level_shift[n_features]      2*log2(64/level); 12 = one value per image
    v3:   uint8    shuffle_tables[n_trees * depth * 16]   T[v] = (v > bin) ? (1<<d) : 0

A split at level ``d`` (0 = root) sets leaf bit ``d`` when the byte-space bin of
its feature is strictly greater than the split's bin.  The leaf index is the sum
of those bits, so the root split is the LOWEST bit (CatBoost oblivious encoding).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["IMSY_MAGIC", "ImysModel", "parse_blob"]

IMSY_MAGIC = b"IMSY"

_SUPPORTED_VERSIONS = (2, 3)  # 2 = byte bins only, 3 = adds shuffle tables
_SHUFFLE_TABLE_VERSION = 3
_MATRIX_NDIM = 2
_HEADER_BYTES = 24
_TABLE_COLUMNS = 16

ByteBins = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
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
    tree_offsets: UInt32Array
    tree_base: UInt32Array
    splits: VoidArray
    leaf_values: Float32Array
    n_borders: UInt32Array
    borders: list[Float32Array]
    level_shift: ByteBins
    shuffle_tables: ByteBins | None

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

    def raw_scores(self, bin_matrix: ByteBins) -> Float64Array:
        """Sum leaf values over all trees for precomputed byte bins."""
        n_cells = bin_matrix.shape[0]
        raw: Float64Array = np.zeros(n_cells, dtype=np.float64)
        for t in range(self.n_trees):
            idx = np.zeros(n_cells, dtype=np.int64)
            off0 = t * self.depth
            for d in range(self.depth):
                split = self.splits[off0 + d]
                idx |= (bin_matrix[:, split["feat"]] > split["bin"]).astype(np.int64) << d
            base = int(self.tree_offsets[t])
            raw += self.leaf_values[base + idx].astype(np.float64)
        return raw

    def predict_proba(self, x: NDArray[np.floating]) -> Float64Array:
        """Positive-class probability for each row of ``x``."""
        raw = self.raw_scores(self.bins(x))
        probabilities: Float64Array = 1.0 / (1.0 + np.exp(-raw))
        return probabilities

    def predict_grid(self, x: NDArray[np.floating], grid: int = 64) -> Float32Array:
        """Positive-class probability reshaped to a ``grid x grid`` map."""
        grid_map: Float32Array = self.predict_proba(x).reshape(grid, grid).astype(np.float32)
        return grid_map


def _u32(blob: bytes, offset: int, count: int) -> UInt32Array:
    """Read a little-endian ``uint32`` array from ``blob`` (zero-copy view)."""
    return np.frombuffer(blob, dtype="<u4", count=count, offset=offset)


def parse_blob(blob: bytes) -> ImysModel:
    """Parse an ``IMSY`` blob into an :class:`ImysModel`."""
    if blob[:4] != IMSY_MAGIC:
        msg = f"not an IMSY blob (magic {blob[:4]!r})"
        raise ValueError(msg)
    version, n_trees, n_features, n_leafs_total, depth = (
        int(v) for v in struct.unpack_from("<IIIII", blob, 4)
    )
    if version not in _SUPPORTED_VERSIONS:
        msg = f"unsupported IMSY version {version}"
        raise ValueError(msg)
    off = _HEADER_BYTES
    tree_offsets = _u32(blob, off, n_trees + 1)
    off += 4 * (n_trees + 1)
    tree_base = _u32(blob, off, n_trees + 1)
    off += 4 * (n_trees + 1)
    n_splits = n_trees * depth
    splits = np.frombuffer(blob, dtype=_SPLIT_DTYPE, count=n_splits, offset=off)
    off += n_splits * 4
    leaf_values = np.frombuffer(blob, dtype="<f4", count=n_leafs_total, offset=off)
    off += n_leafs_total * 4
    n_borders = _u32(blob, off, n_features)
    off += n_features * 4
    borders: list[Float32Array] = []
    for f in range(n_features):
        nb = int(n_borders[f])
        borders.append(np.frombuffer(blob, dtype="<f4", count=nb, offset=off).copy())
        off += nb * 4
    level_shift: ByteBins = np.frombuffer(blob, dtype="u1", count=n_features, offset=off).copy()
    off += n_features
    shuffle_tables: ByteBins | None = None
    if version >= _SHUFFLE_TABLE_VERSION:
        shuffle_tables = (
            np.frombuffer(blob, dtype="u1", count=n_splits * _TABLE_COLUMNS, offset=off)
            .reshape(n_splits, _TABLE_COLUMNS)
            .copy()
        )
        off += n_splits * _TABLE_COLUMNS
    if off != len(blob):
        msg = f"IMSY blob has {len(blob) - off} trailing bytes"
        raise ValueError(msg)
    return ImysModel(
        version=version,
        n_trees=n_trees,
        n_features=n_features,
        n_leafs_total=n_leafs_total,
        depth=depth,
        tree_offsets=tree_offsets,
        tree_base=tree_base,
        splits=splits,
        leaf_values=leaf_values,
        n_borders=n_borders,
        borders=borders,
        level_shift=level_shift,
        shuffle_tables=shuffle_tables,
    )
