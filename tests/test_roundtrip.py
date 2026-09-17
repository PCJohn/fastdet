"""End-to-end tests: fit a tiny detector, export ONE file, reload, score."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Config, Detector, ModelConfig
from fastdet.artifact import ModelArtifact

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fitted(
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Detector, Path]:
    """A detector fitted once on the tiny dataset, in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir)
    return det, images_dir


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


def test_prune_top_k(fitted: tuple[Detector, Path]) -> None:
    """prune(top_k) keeps that many ranked columns, ascending, and invalidates the fit."""
    det, _images_dir = fitted
    assert det.base_names is not None
    assert det.runtime is not None  # fitted

    kept = det.prune(top_k=64)
    assert kept == 64
    assert det.col_keep is not None
    assert len(det.col_keep) == 64
    assert np.all(np.diff(det.col_keep) > 0)

    # Re-pruning changes the design matrix, so the fitted booster no longer
    # matches it: the detector must refuse to score rather than mix them.
    assert det.booster is None
    assert det.runtime is None
    with pytest.raises(RuntimeError, match="not fitted"):
        det.predict_proba(np.zeros((32, 32, 3), dtype=np.uint8))


def test_prune_before_fit_keeps_detector_usable(small_config: Config) -> None:
    """Pruning an unfitted detector is the normal path and clears nothing."""
    det = Detector(small_config)
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
