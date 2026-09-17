"""Build the self-contained ``IMSY`` tree blob from a fitted CatBoost model.

The JSON written by ``CatBoostClassifier.save_model(format="json")`` is the only
model input; borders, split bins and leaf values are copied verbatim, so the
blob is a faithful, compact re-encoding rather than an approximation.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import MAX_BORDER_COUNT
from .runtime import IMSY_MAGIC

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["BlobInfo", "build_blob", "catboost_json_to_blob"]

_SPLIT_STRUCT = struct.Struct("<HBB")  # u16 feature, u8 bin, u8 pad
_TABLE_COLUMNS = MAX_BORDER_COUNT + 1  # one shuffle entry per possible 4-bit bin


class BlobInfo(dict[str, Any]):
    """Summary of a built blob (kept dict-like for easy metadata embedding)."""


@dataclass
class _EncodedTrees:
    """Byte fragments produced while walking the CatBoost trees in order."""

    tree_offsets: list[int]
    tree_base: list[int]
    split_bytes: bytearray
    leaf_bytes: bytearray
    max_referenced_bin: int


def _validate_trees(trees: list[Mapping[str, Any]]) -> tuple[int, int]:
    """Return ``(n_trees, depth)`` and reject ragged trees."""
    depth = len(trees[0]["splits"])
    for tree in trees:
        if len(tree["splits"]) != depth:
            msg = "mismatched depth across trees"
            raise ValueError(msg)
        if len(tree["leaf_values"]) != (1 << depth):
            msg = "leaf count != 2**depth"
            raise ValueError(msg)
    return len(trees), depth


def _feature_borders(model_json: Mapping[str, Any]) -> list[list[float]]:
    """Per-feature border lists, ordered by feature index."""
    float_features = sorted(
        model_json["features_info"]["float_features"], key=lambda f: f["feature_index"]
    )
    return [list(f["borders"]) for f in float_features]


def _check_scale_and_bias(model_json: Mapping[str, Any]) -> None:
    """Reject any output transform that the identity-logit blob cannot carry."""
    scale, bias = 1.0, 0.0
    if "scale_and_bias" in model_json:
        transform = model_json["scale_and_bias"]
        scale = float(transform[0])
        bias = float(transform[1][0]) if transform[1] else 0.0
    if scale != 1.0 or bias != 0.0:
        msg = f"only scale=1/bias=0 supported, got scale={scale} bias={bias}"
        raise ValueError(msg)


def _tree_splits(tree: Mapping[str, Any]) -> list[tuple[int, float]]:
    """``(feature_index, border)`` pairs in split order (root first)."""
    return [(int(sp["float_feature_index"]), float(sp["border"])) for sp in tree["splits"]]


def _bin_index(
    borders: list[list[float]], border_map: list[dict[float, int]], fid: int, border: float
) -> int:
    """Position of ``border`` in feature ``fid``'s border list."""
    if math.isnan(border):
        msg = "NaN split border; refusing to export a broken tree"
        raise ValueError(msg)
    if border not in border_map[fid]:
        nearest = min(borders[fid], key=lambda b: abs(b - border))
        msg = (
            f"feature {fid} split border {border!r} not in borders "
            f"(nearest {nearest!r}); bin space cannot encode this tree"
        )
        raise ValueError(msg)
    return border_map[fid][border]


def _encode_trees(
    trees: list[Mapping[str, Any]],
    depth: int,
    borders: list[list[float]],
    border_map: list[dict[float, int]],
) -> _EncodedTrees:
    """Pack split records and leaf values, tracking the largest referenced bin."""
    tree_offsets = [0]
    tree_base = [0]
    split_bytes = bytearray()
    leaf_bytes = bytearray()
    max_referenced_bin = 0
    for tree in trees:
        for value in tree["leaf_values"]:
            leaf_bytes += struct.pack("<f", float(value))
        tree_offsets.append(tree_offsets[-1] + (1 << depth))
        for fid, border in _tree_splits(tree):
            bin_index = _bin_index(borders, border_map, fid, border)
            max_referenced_bin = max(max_referenced_bin, bin_index)
            split_bytes += _SPLIT_STRUCT.pack(fid, bin_index, 0)
        tree_base.append(len(split_bytes))
    return _EncodedTrees(
        tree_offsets=tree_offsets,
        tree_base=tree_base,
        split_bytes=split_bytes,
        leaf_bytes=leaf_bytes,
        max_referenced_bin=max_referenced_bin,
    )


def _shuffle_table_bytes(
    trees: list[Mapping[str, Any]], border_map: list[dict[float, int]]
) -> bytes:
    """One 16-byte ``vpshufb`` table per split: ``T[v] = (v > bin) << d``."""
    table = bytearray()
    for tree in trees:
        for d, (fid, border) in enumerate(_tree_splits(tree)):
            bin_index = border_map[fid][border]
            if bin_index > MAX_BORDER_COUNT:
                msg = f"4-bit unsafe: feature {fid} referenced bin {bin_index}"
                raise ValueError(msg)
            table += bytes(((1 << d) if v > bin_index else 0) for v in range(_TABLE_COLUMNS))
    return bytes(table)


def build_blob(
    model_json: Mapping[str, Any], level_shift: list[int], *, shuffle_tables: bool = True
) -> tuple[bytes, BlobInfo]:
    """Serialize a CatBoost symmetric model to ``IMSY`` bytes.

    ``level_shift`` gives one coarse-cell shift per kept feature (see
    :func:`fastdet.features.feature_level_bits`).  With ``shuffle_tables`` the
    blob is version 3 and every referenced bin index must be <= 15.
    """
    trees = model_json["oblivious_trees"]
    n_trees, depth = _validate_trees(trees)
    borders = _feature_borders(model_json)
    n_features = len(borders)
    if len(level_shift) != n_features:
        msg = f"level_shift has {len(level_shift)} entries, expected {n_features}"
        raise ValueError(msg)
    border_map = [{b: i for i, b in enumerate(feature_borders)} for feature_borders in borders]
    _check_scale_and_bias(model_json)
    encoded = _encode_trees(trees, depth, borders, border_map)

    border_blob = struct.pack(f"<{n_features}I", *[len(b) for b in borders])
    for feature_borders in borders:
        if feature_borders:
            border_blob += np.asarray(feature_borders, dtype=np.float32).tobytes()
    level_blob = struct.pack(f"<{n_features}B", *level_shift)

    table_blob = b""
    if shuffle_tables:
        table_blob = _shuffle_table_bytes(trees, border_map)
        expected_tables = n_trees * depth * _TABLE_COLUMNS
        if len(table_blob) != expected_tables:
            msg = "shuffle-table size mismatch"
            raise ValueError(msg)

    version = 3 if shuffle_tables else 2
    header = IMSY_MAGIC + struct.pack(
        "<IIIII", version, n_trees, n_features, n_trees * (1 << depth), depth
    )
    blob = (
        header
        + struct.pack(f"<{n_trees + 1}I", *encoded.tree_offsets)
        + struct.pack(f"<{n_trees + 1}I", *encoded.tree_base)
        + bytes(encoded.split_bytes)
        + bytes(encoded.leaf_bytes)
        + border_blob
        + level_blob
        + table_blob
    )
    info = BlobInfo(
        version=version,
        n_trees=n_trees,
        n_features=n_features,
        depth=depth,
        n_borders=sum(len(b) for b in borders),
        max_referenced_bin=encoded.max_referenced_bin,
        blob_bytes=len(blob),
    )
    return blob, info


def catboost_json_to_blob(
    model_json: Mapping[str, Any], level_shift: list[int], *, shuffle_tables: bool = True
) -> tuple[bytes, BlobInfo]:
    """Alias of :func:`build_blob` (explicit about the input being CatBoost JSON)."""
    return build_blob(model_json, level_shift, shuffle_tables=shuffle_tables)
