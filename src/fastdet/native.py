"""The in-process C++ scorer: ``fastdet._native_ext``, a nanobind module built by ``pip install``.

The extension wraps the Highway scorer in ``cpp/fastdet_score.cpp`` (see ``cpp/bindings.cpp``);
scikit-build-core compiles it into the wheel, so a normal install scores at full speed.  The
NumPy runtime in :mod:`fastdet.runtime` is the bit-identical reference the tests compare against,
not a fallback: a missing extension is a broken install, and :func:`load_scorer` says so.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["NativeScorer", "load_scorer"]


class NativeScorer:
    """One loaded model in the C++ scorer."""

    def __init__(self, blob: bytes) -> None:
        """Load ``blob`` (an FDT1 container or bare IMSY blob) into the extension."""
        ext = _extension()
        self._scorer = ext.Scorer(blob)
        self.native_size = int(self._scorer.native_size)  # floats in Detector.native_matrix
        self.cells = int(self._scorer.cells)
        self.target = str(ext.target())  # the SIMD target the module was compiled for

    def score(self, native: NDArray[np.floating], *, use_exit: bool = True) -> NDArray[np.float32]:
        """Probabilities for the whole grid (row-major) from ``Detector.native_matrix`` output."""
        values = np.ascontiguousarray(native, dtype=np.float32).reshape(-1)
        return np.asarray(self._scorer.score(values, use_exit), dtype=np.float32)


def _extension() -> Any:
    try:
        return importlib.import_module("fastdet._native_ext")
    except ImportError as exc:
        msg = (
            "fastdet._native_ext (the C++ scorer) is not built. It is compiled by `pip install .`; "
            "this install is missing it, so reinstall the package with cmake and a C++17 compiler available."
        )
        raise ImportError(msg) from exc


def load_scorer(blob: bytes) -> NativeScorer:
    """The C++ scorer for ``blob``."""
    return NativeScorer(blob)
