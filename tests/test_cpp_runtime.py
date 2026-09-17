"""Gate the C++ runtime against the Python one on the same exported file.

The C++ scorer already self-checks (fused binner vs reference binner, AVX2 vs
scalar) and compares against a recorded expectation when one is passed, exiting
non-zero on any failure.  This test builds it, feeds it the artifact and a
fixture produced by the Python path, and asserts that exit code -- so drift
between the two runtimes fails CI instead of being found by hand.

Skipped when no C++ compiler is available or the toolchain cannot produce an
AVX2 binary that runs on this machine.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Config, Detector

if TYPE_CHECKING:
    from numpy.typing import NDArray

SOURCE = Path(__file__).resolve().parents[1] / "cpp" / "fastdet_score.cpp"
GRID_CELLS = 64 * 64


def _compiler() -> list[str] | None:
    """Return a compile command prefix for an optimized AVX2 build, or None."""
    for name in ("g++", "clang++"):
        exe = shutil.which(name)
        if exe:
            return [exe, "-O2", "-std=c++17", "-mavx2"]
    if sys.platform == "win32" and shutil.which("cl"):
        return ["cl", "/O2", "/EHsc", "/std:c++17", "/arch:AVX2"]
    return None


def _build(tmp_path: Path) -> Path:
    """Compile the scorer into ``tmp_path`` and return the binary path."""
    cmd = _compiler()
    if cmd is None:
        pytest.skip("no C++ compiler available")
    binary = tmp_path / ("fastdet_score.exe" if sys.platform == "win32" else "fastdet_score")
    if cmd[0] == "cl":
        build = [*cmd, str(SOURCE), f"/Fe:{binary}"]
    else:
        build = [*cmd, str(SOURCE), "-o", str(binary)]
    result = subprocess.run(  # noqa: S603 -- argv is built from a vetted compiler path
        build, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the C++ runtime:\n{result.stderr[-2000:]}")
    return binary


def _write_f32(path: Path, values: NDArray[np.floating]) -> None:
    path.write_bytes(np.ascontiguousarray(values, dtype=np.float32).tobytes())


@pytest.mark.slow
def test_cpp_matches_python_runtime(
    tmp_path: Path,
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exported file scores identically under both runtimes."""
    binary = _build(tmp_path)
    monkeypatch.chdir(tmp_path)
    images_dir, masks_dir = tiny_dataset
    det = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    model_path = det.export(tmp_path / "model.fdt")

    sample = images_dir / "img_00.png"
    design = det.design_matrix(sample)
    assert design.shape[0] == GRID_CELLS
    expected = det.predict_proba(sample).reshape(-1)

    fixture_path = tmp_path / "fixture.f32"
    expected_path = tmp_path / "expected.f32"
    _write_f32(fixture_path, design)
    _write_f32(expected_path, expected)

    run = subprocess.run(  # noqa: S603 -- argv is the binary this test just built
        [str(binary), str(model_path), str(fixture_path), str(expected_path), "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0 and "Illegal instruction" in (run.stderr or ""):
        pytest.skip("host CPU does not support AVX2")
    assert run.returncode == 0, f"C++ gate failed:\n{run.stdout}\n{run.stderr}"
    assert "BIT-IDENTICAL" in run.stdout
    assert "0 mismatches PASS" in run.stdout


def test_fixture_layout_is_row_major() -> None:
    """The fixture the C++ reads is (cells, features) float32, row-major."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    packed = np.ascontiguousarray(values).tobytes()
    assert struct.unpack("<6f", packed) == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
