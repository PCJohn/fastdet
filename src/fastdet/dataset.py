"""Dataset pairing, near-duplicate-aware splitting, and training-cell sampling.

Hand-curated photo sets contain bursts (several frames of one scene, re-saves,
crops).  A plain random split puts near-duplicates on both sides and inflates
validation numbers, so the split is done at the granularity of pHash clusters:
no cluster is ever split across train and validation.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import imfeat
import numpy as np
from numpy.typing import NDArray

from .images import collect_pairs, read_image

if TYPE_CHECKING:
    import numpy.typing as npt

    from .config import TrainConfig
    from .features import FeatureCache

__all__ = [
    "build_split",
    "gather_training_matrix",
    "grouped_split",
    "image_hashes",
    "near_duplicate_groups",
    "sample_training_cells",
]

PHASH_SIZE = 256  # Square thumb used only for hashing (unrelated to the feature thumb).
PHASH_ROW = 2  # imfeat.HASHES == ("ahash", "whash", "phash")
_BLOCK = 512  # Column tile for the pairwise Hamming scan.
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)

BoolArray = NDArray[np.bool_]
Int16Array = NDArray[np.int16]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]
Float32Array = NDArray[np.float32]


def _popcount64(x: UInt64Array) -> Int16Array:
    """Elementwise bit count of a uint64 array."""
    fast = getattr(np, "bitwise_count", None)
    if fast is not None:
        counted: Int16Array = fast(x).astype(np.int16)
        return counted
    bytes_view = np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8)
    table_count: Int16Array = _POPCOUNT_TABLE[bytes_view].sum(axis=-1).astype(np.int16)
    return table_count


def _phash_cache_key(path: str) -> str:
    """Cache key for an image: resolved path plus size and mtime when readable."""
    resolved = str(Path(path).resolve())
    try:
        stat = Path(path).stat()
    except OSError:
        return resolved
    return f"{resolved}|{stat.st_size}|{int(stat.st_mtime)}"


def image_hashes(
    pairs: list[tuple[str, str]], cache_path: str | None = None, desc: str = "dupe-check"
) -> tuple[UInt64Array, BoolArray]:
    """64-bit pHash per image -> ``(hashes uint64[n], ok bool[n])``.

    Decodes each image once at ``PHASH_SIZE`` and memoizes results to
    ``cache_path`` keyed by path+size+mtime.
    """
    disk: dict[str, str] = {}
    if cache_path is not None and Path(cache_path).exists():
        try:
            with Path(cache_path).open(encoding="utf-8") as fh:
                disk = json.load(fh)
        except (OSError, ValueError):
            disk = {}

    computer = imfeat.FeatureComputer(shape=(PHASH_SIZE, PHASH_SIZE, 3), grid=[(5, 5)], stride=2)
    hashes = np.zeros(len(pairs), dtype=np.uint64)
    ok = np.zeros(len(pairs), dtype=bool)

    start = time.time()
    cached = 0
    for i, (img_path, _) in enumerate(pairs):
        key = _phash_cache_key(img_path)
        if key in disk:
            hashes[i] = np.uint64(int(disk[key]))
            ok[i] = True
            cached += 1
            continue
        img = read_image(img_path)
        if img is None:
            continue
        thumb = cv2.resize(img, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        hashes[i] = np.uint64(computer.features(hsv).hashes[PHASH_ROW, 2])
        ok[i] = True
        disk[key] = str(int(hashes[i]))
        if (i + 1) % 250 == 0:
            print(f"  [{desc}] hashed {i + 1}/{len(pairs)} ({time.time() - start:.1f}s)")

    if cache_path is not None:
        try:
            with Path(cache_path).open("w", encoding="utf-8") as fh:
                json.dump(disk, fh)
        except OSError:
            pass
    return hashes, ok


def near_duplicate_groups(hashes: UInt64Array, ok: BoolArray, max_hamming: int) -> Int64Array:
    """Union-find over pHash pairs within ``max_hamming`` bits -> group id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    idx = np.flatnonzero(ok)
    kept = hashes[idx]
    for start in range(0, len(idx), _BLOCK):
        chunk = kept[start : start + _BLOCK]
        distances = _popcount64(np.bitwise_xor(chunk[:, None], kept[None, :]))
        rows, cols = np.nonzero(distances <= max_hamming)
        for row, col in zip(rows, cols, strict=True):
            gi, gj = int(idx[start + row]), int(idx[col])
            if gi < gj:
                union(gi, gj)

    roots: dict[int, int] = {}
    groups = np.empty(n, dtype=np.int64)
    for i in range(n):
        root = find(i)
        roots.setdefault(root, len(roots))
        groups[i] = roots[root]
    return groups


def grouped_split(
    pairs: list[tuple[str, str]], groups: Int64Array, val_frac: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Random split at group granularity -> ``(train_pairs, val_pairs)``."""
    unique = sorted({int(g) for g in groups})
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not cryptography
    rng.shuffle(unique)
    target = max(1, int(len(pairs) * val_frac))
    members: dict[int, list[int]] = {g: [] for g in unique}
    for i, g in enumerate(groups):
        members[int(g)].append(i)

    val_idx: list[int] = []
    count = 0
    for g in unique:
        if count >= target:
            break
        val_idx.extend(members[g])
        count += len(members[g])
    val_set = set(val_idx)
    train_idx = [i for i in range(len(pairs)) if i not in val_set]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def build_split(
    images_dir: str,
    masks_dir: str,
    cfg: TrainConfig,
    phash_cache: str | None = ".phash_cache.json",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Reproduce the frozen group-aware split -> ``(pairs, train_pairs, val_pairs)``."""
    pairs, missing = collect_pairs(images_dir, masks_dir)
    if missing:
        msg = f"{len(missing)} images have no matching mask, e.g. {missing[:3]}"
        raise RuntimeError(msg)
    hashes, ok = image_hashes(pairs, cache_path=phash_cache)
    groups = near_duplicate_groups(hashes, ok, cfg.max_hamming)
    train_pairs, val_pairs = grouped_split(pairs, groups, cfg.val_frac, cfg.split_seed)
    return pairs, train_pairs, val_pairs


def _sample_targets(
    n_pos_total: int, n_neg_total: int, neg_pos_ratio: float | None, max_train_cells: int | None
) -> tuple[int, int]:
    """Final positive/negative counts, resolved before any cell is chosen."""
    n_pos, n_neg = n_pos_total, n_neg_total
    if neg_pos_ratio is not None:
        n_neg = min(n_neg_total, max(1, round(neg_pos_ratio * n_pos_total)))
    if max_train_cells is not None and n_pos + n_neg > max_train_cells:
        pos_frac = n_pos / max(n_pos + n_neg, 1)
        n_pos = min(n_pos_total, max(1, round(max_train_cells * pos_frac)))
        n_neg = min(n_neg, max(1, max_train_cells - n_pos))
    return n_pos, n_neg


def _concat_or_empty(chunks: list[NDArray[Any]], dtype: npt.DTypeLike) -> NDArray[Any]:
    """Concatenate ``chunks``, or return an empty array of ``dtype`` when there are none."""
    if chunks:
        return np.concatenate(chunks)
    return np.array([], dtype=dtype)


def sample_training_cells(
    cache: FeatureCache, cfg: TrainConfig, seed: int, desc: str = "train"
) -> tuple[Int32Array, Int64Array, BoolArray]:
    """Choose the ``(image, cell)`` pairs to train on.

    Returns ``(img_ids int32, local_ids int64, labels bool)``.  Positives are
    cells whose coverage clears ``gt_cell_thresh``; negatives are the rest,
    optionally downsampled by ``neg_pos_ratio`` and ``max_train_cells``.
    """
    rng = np.random.RandomState(seed)
    pos_i: list[Int32Array] = []
    pos_l: list[Int64Array] = []
    neg_i: list[Int32Array] = []
    neg_l: list[Int64Array] = []
    for i in range(cache.n):
        coverage = cache.gt_coverage_list[i]
        positive = coverage >= cfg.gt_cell_thresh
        local_pos = np.flatnonzero(positive)
        local_neg = np.flatnonzero(~positive)
        if len(local_pos):
            pos_i.append(np.full(len(local_pos), i, np.int32))
            pos_l.append(local_pos)
        if len(local_neg):
            neg_i.append(np.full(len(local_neg), i, np.int32))
            neg_l.append(local_neg)

    pos_i_arr: Int32Array = _concat_or_empty(pos_i, np.int32)
    pos_l_arr: Int64Array = _concat_or_empty(pos_l, np.int64)
    neg_i_arr: Int32Array = _concat_or_empty(neg_i, np.int32)
    neg_l_arr: Int64Array = _concat_or_empty(neg_l, np.int64)

    n_pos_total, n_neg_total = len(pos_i_arr), len(neg_i_arr)
    if n_pos_total == 0:
        msg = "training split contains zero positive cells"
        raise RuntimeError(msg)

    n_pos, n_neg = _sample_targets(n_pos_total, n_neg_total, cfg.neg_pos_ratio, cfg.max_train_cells)
    sel_pos = (
        np.arange(n_pos_total)
        if n_pos >= n_pos_total
        else rng.choice(n_pos_total, size=n_pos, replace=False)
    )
    sel_neg = (
        np.arange(n_neg_total)
        if n_neg >= n_neg_total
        else rng.choice(n_neg_total, size=n_neg, replace=False)
    )

    img_ids = np.concatenate([pos_i_arr[sel_pos], neg_i_arr[sel_neg]]).astype(np.int32)
    local_ids = np.concatenate([pos_l_arr[sel_pos], neg_l_arr[sel_neg]]).astype(np.int64)
    labels = np.concatenate([np.ones(len(sel_pos), bool), np.zeros(len(sel_neg), bool)])

    perm = rng.permutation(len(img_ids))
    img_ids, local_ids, labels = img_ids[perm], local_ids[perm], labels[perm]
    print(
        f"  [{desc}] sampled cells={len(labels):,} positive={int(labels.sum()):,} "
        f"({100.0 * labels.mean():.2f}%)"
    )
    return img_ids, local_ids, labels


def gather_training_matrix(
    cache: FeatureCache,
    img_ids: Int32Array,
    local_ids: Int64Array,
    col_keep: NDArray[np.integer] | None = None,
) -> Float32Array:
    """Assemble the ``(n_cells, width)`` float32 design matrix for sampled cells."""
    width = len(col_keep) if col_keep is not None else cache.total_width()
    design = np.empty((len(img_ids), width), dtype=np.float32)

    order = np.argsort(img_ids, kind="stable")
    sorted_ids = img_ids[order]
    boundaries = np.searchsorted(sorted_ids, np.arange(cache.n + 1))
    for image_id in range(cache.n):
        start, end = boundaries[image_id], boundaries[image_id + 1]
        if start == end:
            continue
        selection = order[start:end]
        design[selection] = cache.gather(image_id, local_ids[selection], col_keep=col_keep)
    return design
