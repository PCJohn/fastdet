"""End-to-end tests: fit a tiny detector, export ONE file, reload, score."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from fastdet import Config, Detector, ModelConfig
from fastdet.artifact import ModelArtifact
from fastdet.features import feature_level_bits
from fastdet.training import build_ranking

if TYPE_CHECKING:
    from pathlib import Path


def test_export_load_roundtrip_is_bit_exact(fitted: tuple[Detector, Path], tmp_path: Path) -> None:
    """Reloading the exported single file reproduces scores exactly."""
    det, images_dir = fitted
    sample = images_dir / "img_00.png"

    before = det.predict_proba(sample)
    assert before.shape == (64, 64)
    assert np.isfinite(before).all()
    assert float(before.min()) >= 0.0

    out = det.export(tmp_path / "model.fdt")
    assert out.exists()
    assert out.stat().st_size > 0

    loaded = Detector.load(out)
    after = loaded.predict_proba(sample)
    np.testing.assert_array_equal(before, after)

    assert loaded.feature_names == det.feature_names
    assert loaded.metrics["pr_auc"] == pytest.approx(det.metrics["pr_auc"])
    assert loaded.config.to_dict() == det.config.to_dict()


def test_artifact_blob_matches_live_runtime(fitted: tuple[Detector, Path], tmp_path: Path) -> None:
    """The bytes stored in the container are stable across builds."""
    det, _images_dir = fitted
    artifact = det.build_artifact()
    written = artifact.save(tmp_path / "model.fdt")
    reloaded = ModelArtifact.load(written)
    assert reloaded.blob == artifact.blob
    assert reloaded.feature_names == artifact.feature_names
    assert det.runtime is not None
    assert det.runtime.n_trees == det.config.model.n_trees


def _write_ranking(path: Path, names: list[str]) -> Path:
    """A ranking over the live column names, as a real regeneration would produce."""
    path.write_text(json.dumps({"names": list(names)}), encoding="utf-8")
    return path


def test_prune_top_k(fitted: tuple[Detector, Path], tmp_path: Path) -> None:
    """prune(top_k) keeps that many ranked columns, ascending, and invalidates the fit."""
    det, _images_dir = fitted
    assert det.base_names is not None
    assert det.runtime is not None  # fitted

    det.feature_ranks = None
    det.config.train.feature_ranks = str(_write_ranking(tmp_path / "ranks.json", det.base_names))
    kept = det.prune(top_k=64)
    assert kept == 64
    assert det.col_keep is not None
    assert len(det.col_keep) == 64
    assert np.all(np.diff(det.col_keep) > 0)

    # Re-pruning changes the design matrix, so the fitted booster no longer
    # matches it: the detector must refuse to score rather than mix them.
    # (Read the fields back through getattr: mypy keeps the pre-prune narrowing.)
    assert getattr(det, "booster") is None  # noqa: B009
    assert getattr(det, "runtime") is None  # noqa: B009
    with pytest.raises(RuntimeError, match="not fitted"):
        det.predict_proba(np.zeros((32, 32, 3), dtype=np.uint8))


def test_prune_before_fit_keeps_detector_usable(small_config: Config, tmp_path: Path) -> None:
    """Pruning an unfitted detector is the normal path and clears nothing."""
    det = Detector(small_config)
    det.config.train.feature_ranks = str(
        _write_ranking(tmp_path / "ranks.json", list(det.extractor.base_names))
    )
    kept = det.prune(top_k=32)
    assert kept == 32
    assert det.booster is None
    assert det.config.train.top_k_features == 32


def test_model_config_only_constructor() -> None:
    """A bare ModelConfig is accepted and flattened into a full Config."""
    det = Detector(ModelConfig(depth=4, n_trees=10))
    assert det.config.model.depth == 4
    assert det.config.model.n_trees == 10


def test_predict_before_fit_raises() -> None:
    """Scoring an unfitted detector is an error, not a silent empty map."""
    det = Detector()
    with pytest.raises(RuntimeError, match="not fitted"):
        det.predict_proba(np.zeros((32, 32, 3), dtype=np.uint8))


def test_rank_prune_export_flow(
    fitted: tuple[Detector, Path], tiny_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    """Full-width fit -> ranking -> pruned refit -> export -> reload, end to end."""
    full, _images_dir = fitted
    assert full.booster is not None
    assert full.feature_names is not None
    assert len(full.feature_names) == full.extractor.total_width()  # fitted at full width

    ranking = build_ranking(full.booster, full.feature_names)
    assert ranking["width"] == len(full.feature_names)
    assert ranking["gains"] == sorted(ranking["gains"], reverse=True)
    ranks_path = tmp_path / "ranks.json"
    ranks_path.write_text(json.dumps(ranking), encoding="utf-8")

    cfg = Config.from_dict(full.config.to_dict())
    cfg.train.top_k_features = 48
    cfg.train.feature_ranks = str(ranks_path)
    images_dir, masks_dir = tiny_dataset
    pruned = Detector(cfg).fit(images_dir, masks_dir)
    assert pruned.col_keep is not None
    assert len(pruned.col_keep) == 48
    assert set(pruned.feature_names or []) <= set(ranking["names"][:48])

    loaded = Detector.load(pruned.export(tmp_path / "pruned.fdt"))
    sample = images_dir / "img_00.png"
    np.testing.assert_array_equal(loaded.predict_proba(sample), pruned.predict_proba(sample))


def test_pad_trees_is_exact() -> None:
    """A shallow oblivious tree padded to full depth scores identically."""
    from fastdet.exporter import _pad_trees  # noqa: PLC0415 -- private helper under test

    split: dict[str, Any] = {"float_feature_index": 0, "border": 0.5}
    shallow: dict[str, Any] = {"splits": [split], "leaf_values": [-1.0, 2.0]}  # depth 1
    deep: dict[str, Any] = {"splits": [split] * 3, "leaf_values": [float(i) for i in range(8)]}
    padded, kept = _pad_trees([shallow, deep])
    assert kept is deep  # already full depth: untouched
    assert len(padded["splits"]) == 3
    for idx in range(8):  # root split is bit 0, so only bit 0 may matter
        assert padded["leaf_values"][idx] == shallow["leaf_values"][idx & 1]


def _expand_native(det: Detector, native: np.ndarray) -> np.ndarray:
    """Repeat each native value over its level block: the dense per-cell layout."""
    grid = 64
    columns, pos = [], 0
    for name in det.feature_names or []:
        side = feature_level_bits(name)[0]
        block = native[pos : pos + side * side].reshape(side, side)
        pos += side * side
        factor = grid // side
        columns.append(np.repeat(np.repeat(block, factor, axis=0), factor, axis=1).ravel())
    assert pos == native.size, "native buffer has trailing values"
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("top_k", [0, 48])
def test_native_matrix_expands_to_design_matrix(
    fitted: tuple[Detector, Path], tmp_path: Path, top_k: int
) -> None:
    """The native buffer is the design matrix without its repetition, bit for bit."""
    det, images_dir = fitted
    if top_k:
        assert det.feature_names is not None
        ranks = tmp_path / "ranks.json"
        # Rank a strided sample first so the kept set spans every level and globals.
        names = det.feature_names
        ranked = names[:: len(names) // top_k] + [
            n for n in names if n not in names[:: len(names) // top_k]
        ]
        ranks.write_text(json.dumps({"names": ranked}), encoding="utf-8")
        det.feature_ranks = None
        det.config.train.feature_ranks = str(ranks)
        det.prune(top_k=top_k)
        assert det.col_keep is not None
        assert det.base_names is not None
        det.feature_names = [det.base_names[int(i)] for i in det.col_keep]
    for sample in sorted(images_dir.glob("*.png"))[:3]:
        native = det.native_matrix(sample)
        dense = det.design_matrix(sample)
        assert native.dtype == dense.dtype == np.float32
        assert native.size <= dense.size
        np.testing.assert_array_equal(_expand_native(det, native), dense)
    levels = {feature_level_bits(n)[0] for n in det.feature_names or []}
    assert {1, 2, 64} <= levels, f"kept set should span globals to finest, got {sorted(levels)}"
