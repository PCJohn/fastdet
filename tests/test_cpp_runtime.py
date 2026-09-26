"""Gate the C++ runtime against the Python one on the same exported file.

The C++ scorer already self-checks (native binner vs reference binner, SIMD vs
scalar traversal, threads vs one thread) and compares against a recorded expectation
when one is passed, exiting non-zero on any failure.  These tests build it with CMake
(Highway, targeting the build machine), feed it the artifact and a fixture produced
by the Python path, and assert that exit code -- so drift between the two runtimes
fails CI instead of being found by hand.

Skipped when cmake is unavailable or the scorer cannot be built.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Detector
from fastdet.exporter import build_blob

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from subprocess import CompletedProcess

    from numpy.typing import NDArray

    RunScorer = Callable[..., CompletedProcess[str]]

GRID_CELLS = 64 * 64


def _write_f32(path: Path, values: NDArray[np.floating]) -> None:
    path.write_bytes(np.ascontiguousarray(values, dtype=np.float32).tobytes())


def _write_fixture(det: Detector, sample: Path, scratch: Path) -> tuple[Path, Path]:
    """The native fixture the C++ reads, and the Python reference's full scores.

    The expectation comes from the dense reference runtime (every tree on every cell),
    so the two paths share no feature layout code.
    """
    fixture_path, expected_path = scratch / "fixture.f32", scratch / "expected.f32"
    _write_f32(fixture_path, det.native_matrix(sample))
    assert det.runtime is not None
    full = det.runtime.predict_grid(det.design_matrix(sample), use_exit=False)
    _write_f32(expected_path, full.reshape(-1))
    return fixture_path, expected_path


@pytest.mark.slow
def test_cpp_matches_python_runtime(
    run_scorer: RunScorer, scratch: Path, tiny_dataset: tuple[Path, Path], tiny_model: Path
) -> None:
    """The exported file scores identically under both runtimes, on every image.

    Every image, not one: a binner bug can hide behind values that happen to land
    in bin 0 (an image-wide feature below its first border writes the same byte
    whether or not its plane is filled), so one image is not enough coverage.
    """
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    for sample in sorted(images_dir.glob("*.png")):
        fixture_path, expected_path = _write_fixture(det, sample, scratch)
        run = run_scorer(tiny_model, fixture_path, expected_path, 2)
        detail = f"{sample.name}:\n{run.stdout}\n{run.stderr}"
        assert run.returncode == 0, f"C++ gate failed on {detail}"
        assert "BIT-IDENTICAL" in run.stdout, detail
        assert "DIVERGED" not in run.stdout, detail
        assert "0 mismatches PASS" in run.stdout, detail
    det.close()


@pytest.mark.slow
@pytest.mark.parametrize(
    "stage",
    [
        pytest.param(("-1e30", True), id="unmissable"),
        pytest.param(("1e30", False), id="unreachable"),
    ],
)
def test_cpp_early_exit_keeps_or_drops_every_pack(
    run_scorer: RunScorer,
    scratch: Path,
    tiny_dataset: tuple[Path, Path],
    tiny_model: Path,
    stage: tuple[str, bool],
) -> None:
    """Optional early exit keeps or drops whole packs of tiles at a stage.

    A threshold nothing can miss changes no score; one nothing can reach stops every
    pack at that stage.
    """
    theta, all_finish = stage
    images_dir, _masks_dir = tiny_dataset
    det = Detector.load(tiny_model)
    fixture_path, expected_path = _write_fixture(det, min(images_dir.glob("*.png")), scratch)
    det.close()
    run = run_scorer(tiny_model, fixture_path, expected_path, 2, f"16:{theta}")
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    found = re.search(r"(\d+) of (\d+) packs ran every tree", run.stdout)
    assert found is not None, run.stdout
    finished, packs = int(found.group(1)), int(found.group(2))
    assert finished == (packs if all_finish else 0), run.stdout
    if all_finish:
        assert "4096 of 4096 cells bit-identical" in run.stdout, run.stdout


@pytest.mark.slow
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_cpp_image_wide_feature_fills_every_cell(
    run_scorer: RunScorer, scratch: Path, value: float
) -> None:
    """An image-wide (level_shift 12) feature must bin the same in all 4096 cells.

    A hand-built one-split model makes this deterministic: with the value above
    its border every cell takes leaf 1, including the high-nibble half of the
    packed plane that a per-block binner cannot reach from the first row.  The
    value below the border (bin 0) is the case that used to pass by accident.
    """
    model_json = {
        "features_info": {"float_features": [{"feature_index": 0, "borders": [0.5]}]},
        "oblivious_trees": [
            {"splits": [{"float_feature_index": 0, "border": 0.5}], "leaf_values": [-1.0, 2.0]}
        ],
    }
    blob, _info = build_blob(model_json, level_shift=[12])
    model_path = scratch / "model.imsy"
    model_path.write_bytes(blob)
    fixture_path = scratch / "fixture.f32"
    expected_path = scratch / "expected.f32"
    _write_f32(fixture_path, np.full(1, value, dtype=np.float32))  # one native value
    leaf = 2.0 if value > 0.5 else -1.0
    _write_f32(expected_path, np.full(GRID_CELLS, 1.0 / (1.0 + np.exp(-leaf)), dtype=np.float32))
    run = run_scorer(model_path, fixture_path, expected_path, 2)
    assert run.returncode == 0, f"C++ gate failed:\n{run.stdout}\n{run.stderr}"
    assert "0 mismatches PASS" in run.stdout, run.stdout
