"""The hyperparameter sweep: report, charts and the heuristic baseline on the tiny dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet.tune import (
    build_parser,
    framegate_text_heuristic,
    roc_curve,
    run_sweep,
    summarise,
    sweep_from_args,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_metrics_on_a_separable_toy() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = np.array([True, True, True, False, False, False])
    m = summarise(scores, labels)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    _fpr, _tpr, auc = roc_curve(scores[::-1], labels)  # reversed scores: perfectly wrong
    assert auc == pytest.approx(0.0)


def test_heuristic_prefers_texture() -> None:
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 128.0, np.float32)
    var = np.zeros((64, 64), np.float32)
    var[16:48, 16:48] = 900.0  # textured block
    score = framegate_text_heuristic(
        flat, var, np.zeros_like(var), np.zeros_like(var), np.full_like(var, 0.5)
    )
    assert score[32, 32] > score[2, 2]
    assert np.all(score >= 0)
    assert rng.random() < 1.0


def test_cli_sweep_parsing() -> None:
    args = build_parser().parse_args(
        [
            "--images",
            "a",
            "--masks",
            "b",
            "--n-trees",
            "8,16,8",
            "--leaf-bits",
            "4",
            "--levels",
            "64/32",
        ]
    )
    sweep = sweep_from_args(args)
    assert sweep == {"n_trees": [8, 16], "leaf_bits": [4], "levels": [(64, 32)]}
    args = build_parser().parse_args(
        [
            "--images",
            "a",
            "--masks",
            "b",
            "--neg-pos-ratio",
            "none,3,2.5",
            "--scale-pos-weight",
            "None,4",
            "--use-exit",
            "false",
        ]
    )
    assert sweep_from_args(args) == {
        "neg_pos_ratio": [None, 3.0, 2.5],
        "scale_pos_weight": [None, 4.0],
        "use_exit": [False],
    }


def test_sweep_writes_report_and_charts(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    images_dir, masks_dir = tiny_dataset
    out = tmp_path / "report"
    sweep = {
        "n_trees": [12, 24],
        "depth": [3],
        "val_frac": [0.25],
        "thumb": [256],
        "stride": [1],
        "top_k_features": [0],
    }
    report = run_sweep(images_dir, masks_dir, out, sweep, top_charts=2)
    text = report.read_text()
    assert "| run | n_trees | PR-AUC" in text
    assert "baseline: framegate heuristic" in text
    assert (out / "pr.png").exists()
    assert (out / "roc.png").exists()
    assert len(list((out / "models").glob("*.fdt"))) == 2
    # resuming does not refit
    report2 = run_sweep(images_dir, masks_dir, out, sweep, top_charts=2)
    assert report2.read_text().count("| baseline") == 1
