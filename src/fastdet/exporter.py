"""Build the self-contained ``IMSY`` tree blob from a fitted CatBoost model.

The JSON written by ``CatBoostClassifier.save_model(format="json")`` is the only
model input; borders and split bins are copied verbatim and the leaves are put on
the low-bit grid the model was fitted for (see :mod:`fastdet.runtime` for the
layout).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import MAX_BORDER_COUNT
from .runtime import BLOB_VERSION, IMSY_MAGIC

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["BlobInfo", "build_blob", "catboost_json_to_blob"]

_SPLIT_STRUCT = struct.Struct("<HBB")  # u16 feature, u8 bin, u8 pad


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


def _pad_trees(trees: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Bring every oblivious tree to the deepest tree's depth, exactly.

    CatBoost stops a symmetric tree early when no split improves the loss, so a
    model can mix depths; the blob stores one depth for all trees.  A depth-``d``
    tree becomes depth ``D`` by appending ``D - d`` placeholder splits and
    repeating its leaves so the new (high) index bits are ignored:
    ``leaves'[i] = leaves[i & (2**d - 1)]``, since the root split is bit 0.
    The placeholder split's outcome never matters; it just has to be encodable.
    """
    depth = max(len(tree["splits"]) for tree in trees)
    if all(len(tree["splits"]) == depth for tree in trees):
        return trees
    filler = next((tree["splits"][0] for tree in trees if tree["splits"]), None)
    if filler is None:
        msg = "every tree is a constant: nothing to encode"
        raise ValueError(msg)
    padded: list[Mapping[str, Any]] = []
    for tree in trees:
        if len(tree["splits"]) == depth:
            padded.append(tree)
            continue
        splits = list(tree["splits"])
        leaves = list(tree["leaf_values"])
        mask = (1 << len(splits)) - 1
        pad = splits[0] if splits else filler
        padded.append(
            {
                **tree,
                "splits": splits + [pad] * (depth - len(splits)),
                "leaf_values": [leaves[i & mask] for i in range(1 << depth)],
            }
        )
    return padded


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
    leaf_bytes = bytearray()  # unused since v4: leaves travel as codes
    max_referenced_bin = 0
    for tree in trees:
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


def build_blob(  # noqa: PLR0913 -- the model's few knobs, all explicit
    model_json: Mapping[str, Any],
    level_shift: list[int],
    *,
    leaf_bits: int = 8,
    leaf_chunk: int = 16,
    coarse_trees: int = 0,
    exit_stages: Sequence[tuple[int, float]] = (),
) -> tuple[bytes, BlobInfo]:
    """Serialize a CatBoost symmetric model to ``IMSY`` version 4 bytes.

    Leaves are stored as the low-bit grid of :func:`fastdet.training.leaf_grid`:
    a ``leaf_bits`` code per leaf, a float32 offset per tree and a power-of-two
    step per chunk of ``leaf_chunk`` trees (chunks restart at ``coarse_trees``, the
    stage boundary, exactly as a quantisation-aware fit counts them).  Both
    runtimes sum the codes as integers, so they agree to the bit.  ``exit_stages``
    are ``(trees, threshold)`` pairs the scorer may apply (see the runtime).
    ``level_shift`` gives one coarse-cell shift per kept feature.
    """
    from .training import leaf_grid  # noqa: PLC0415 -- avoids a circular import

    trees = _pad_trees(model_json["oblivious_trees"])
    n_trees, depth = _validate_trees(trees)
    if not 0 <= coarse_trees <= n_trees:
        msg = f"coarse_trees={coarse_trees} outside [0, {n_trees}]"
        raise ValueError(msg)
    stage_starts = (0, coarse_trees) if 0 < coarse_trees < n_trees else (0,)
    grid = leaf_grid(
        np.asarray([tree["leaf_values"] for tree in trees], dtype=np.float64),
        leaf_bits,
        leaf_chunk,
        stage_starts,
    )
    borders = _feature_borders(model_json)
    n_features = len(borders)
    if len(level_shift) != n_features:
        msg = f"level_shift has {len(level_shift)} entries, expected {n_features}"
        raise ValueError(msg)
    border_map = [{b: i for i, b in enumerate(feature_borders)} for feature_borders in borders]
    _check_scale_and_bias(model_json)
    encoded = _encode_trees(trees, depth, borders, border_map)
    if encoded.max_referenced_bin > MAX_BORDER_COUNT:
        msg = f"a split references bin {encoded.max_referenced_bin} > {MAX_BORDER_COUNT}"
        raise ValueError(msg)

    border_blob = struct.pack(f"<{n_features}I", *[len(b) for b in borders])
    for feature_borders in borders:
        if feature_borders:
            border_blob += np.asarray(feature_borders, dtype=np.float32).tobytes()
    level_blob = struct.pack(f"<{n_features}B", *level_shift)
    stage_blob = b"".join(struct.pack("<If", int(t), float(theta)) for t, theta in exit_stages)

    header = IMSY_MAGIC + struct.pack(
        "<IIIIIIIIiII",
        BLOB_VERSION,
        n_trees,
        n_features,
        n_trees * (1 << depth),
        depth,
        leaf_bits,
        leaf_chunk,
        coarse_trees,
        grid.e_min,
        grid.n_chunks,
        len(exit_stages),
    )
    blob = (
        header
        + struct.pack(f"<{n_trees + 1}I", *encoded.tree_offsets)
        + struct.pack(f"<{n_trees + 1}I", *encoded.tree_base)
        + bytes(encoded.split_bytes)
        + grid.codes.reshape(-1).astype(np.uint8).tobytes()
        + grid.offsets.astype("<f4").tobytes()
        + grid.shifts.astype(np.uint8).tobytes()
        + stage_blob
        + border_blob
        + level_blob
    )
    info = BlobInfo(
        version=BLOB_VERSION,
        n_trees=n_trees,
        n_features=n_features,
        depth=depth,
        leaf_bits=leaf_bits,
        leaf_chunk=leaf_chunk,
        coarse_trees=coarse_trees,
        n_borders=sum(len(b) for b in borders),
        max_referenced_bin=encoded.max_referenced_bin,
        exit_stages=[[int(t), float(theta)] for t, theta in exit_stages],
        blob_bytes=len(blob),
    )
    return blob, info


def catboost_json_to_blob(
    model_json: Mapping[str, Any], level_shift: list[int], **kwargs: Any
) -> tuple[bytes, BlobInfo]:
    """Alias of :func:`build_blob` (explicit about the input being CatBoost JSON)."""
    return build_blob(model_json, level_shift, **kwargs)
