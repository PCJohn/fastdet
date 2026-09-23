"""Gate the C++ runtime against the Python one on the same exported file.

The C++ scorer already self-checks (native binner vs reference binner, SIMD vs
scalar traversal) and compares against a recorded expectation when one is
passed, exiting non-zero on any failure.  This test builds it with CMake (Highway,
targeting the build machine), feeds it the artifact and a fixture produced by the
Python path, and asserts that exit code -- so drift between the two runtimes
fails CI instead of being found by hand.

Skipped when cmake is unavailable or the scorer cannot be built.
"""

from __future__ import annotations

import re
import struct
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Config, Detector
from fastdet.exporter import build_blob

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

GRID_CELLS = 64 * 64


def _write_f32(path: Path, values: NDArray[np.floating]) -> None:
    path.write_bytes(np.ascontiguousarray(values, dtype=np.float32).tobytes())


@pytest.mark.slow
def test_cpp_matches_python_runtime(
    scorer: Path,
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exported file scores identically under both runtimes, on every image.

    Every image, not one: a binner bug can hide behind values that happen to land
    in bin 0 (an image-wide feature below its first border writes the same byte
    whether or not its plane is filled), so one image is not enough coverage.
    """
    binary = scorer
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    model_path = det.export(tmp_path / "model.fdt")

    for sample in sorted(images_dir.glob("*.png")):
        # The C++ reads the native fixture; the expectation comes from the dense
        # Python reference runtime, so the two paths share no feature layout code.
        fixture_path = tmp_path / "fixture.f32"
        expected_path = tmp_path / "expected.f32"
        _write_f32(fixture_path, det.native_matrix(sample))
        _write_f32(expected_path, det.predict_proba(sample).reshape(-1))

        run = subprocess.run(  # noqa: S603 -- argv is the binary this test just built
            [str(binary), str(model_path), str(fixture_path), str(expected_path), "2"],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
            pytest.skip("host CPU lacks the SIMD target the scorer was built for")
        detail = f"{sample.name}:\n{run.stdout}\n{run.stderr}"
        assert run.returncode == 0, f"C++ gate failed on {detail}"
        assert "BIT-IDENTICAL" in run.stdout, detail
        assert "0 mismatches PASS" in run.stdout, detail


@pytest.mark.slow
@pytest.mark.parametrize(("theta", "all_finish"), [("-1e30", True), ("1e30", False)])
def test_cpp_early_exit_keeps_or_drops_every_pack(  # noqa: PLR0913, PLR0917 -- fixtures
    scorer: Path,
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    theta: str,
    *,
    all_finish: bool,
) -> None:
    """Optional early exit keeps or drops whole packs of tiles at a stage.

    A threshold nothing can miss changes no score; one nothing can reach stops every
    pack at that stage.
    """
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    model_path = det.export(tmp_path / "model.fdt")
    sample = min(images_dir.glob("*.png"))
    _write_f32(tmp_path / "fixture.f32", det.native_matrix(sample))
    _write_f32(tmp_path / "expected.f32", det.predict_proba(sample).reshape(-1))
    run = subprocess.run(  # noqa: S603 -- argv is the binary this test just built
        [str(scorer), str(model_path), "fixture.f32", "expected.f32", "2", f"16:{theta}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
        pytest.skip("host CPU lacks the SIMD target the scorer was built for")
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    found = re.search(r"(\d+) of (\d+) packs ran every tree", run.stdout)
    assert found is not None, run.stdout
    finished, packs = int(found.group(1)), int(found.group(2))
    assert finished == (packs if all_finish else 0), run.stdout
    if all_finish:
        assert "4096 of 4096 cells bit-identical" in run.stdout, run.stdout


def test_fixture_layout_is_row_major() -> None:
    """The fixture the C++ reads is (cells, features) float32, row-major."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    packed = np.ascontiguousarray(values).tobytes()
    assert struct.unpack("<6f", packed) == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


@pytest.mark.slow
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_cpp_image_wide_feature_fills_every_cell(
    scorer: Path, tmp_path: Path, value: float
) -> None:
    """An image-wide (level_shift 12) feature must bin the same in all 4096 cells.

    A hand-built one-split model makes this deterministic: with the value above
    its border every cell takes leaf 1, including the high-nibble half of the
    packed plane that a per-block binner cannot reach from the first row.  The
    value below the border (bin 0) is the case that used to pass by accident.
    """
    binary = scorer
    model_json = {
        "features_info": {"float_features": [{"feature_index": 0, "borders": [0.5]}]},
        "oblivious_trees": [
            {"splits": [{"float_feature_index": 0, "border": 0.5}], "leaf_values": [-1.0, 2.0]}
        ],
    }
    blob, _info = build_blob(model_json, level_shift=[12])
    model_path = tmp_path / "model.imsy"
    model_path.write_bytes(blob)
    fixture_path = tmp_path / "fixture.f32"
    expected_path = tmp_path / "expected.f32"
    _write_f32(fixture_path, np.full(1, value, dtype=np.float32))  # one native value
    leaf = 2.0 if value > 0.5 else -1.0
    _write_f32(expected_path, np.full(GRID_CELLS, 1.0 / (1.0 + np.exp(-leaf)), dtype=np.float32))

    run = subprocess.run(  # noqa: S603 -- argv is the binary this test just built
        [str(binary), str(model_path), str(fixture_path), str(expected_path), "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
        pytest.skip("host CPU lacks the SIMD target the scorer was built for")
    assert run.returncode == 0, f"C++ gate failed:\n{run.stdout}\n{run.stderr}"
    assert "0 mismatches PASS" in run.stdout, run.stdout
