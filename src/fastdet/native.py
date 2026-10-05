"""The in-process C++ scorer: ``fastdet._native_ext``, a nanobind module built by ``pip install``.

The extension wraps the Highway scorer in ``cpp/fastdet_score.cpp`` (see ``cpp/bindings.cpp``);
scikit-build-core compiles it into the wheel, so a normal install scores at full speed.  The
NumPy runtime in :mod:`fastdet.runtime` is the bit-identical reference the tests compare against,
not a fallback: a missing extension is a broken install, and :func:`load_scorer` says so.

A scorer takes its input two ways: :meth:`NativeScorer.score` the packed matrix of
:meth:`Detector.native_matrix` (every kept column's values contiguous), and
:meth:`NativeScorer.score_maps` the level maps as the front-end leaves them, read where they
lie -- the live path, which makes no packed copy.  Same bytes out either way.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .features import N_BANK_PLANES, _write_coords, default_threads

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .features import FloatArray, MapSources

__all__ = ["NativeScorer", "load_scorer"]


class NativeScorer:
    """One loaded model in the C++ scorer, with its worker threads.

    ``threads`` is how many threads score each image (``None`` = :func:`default_threads`);
    the output is bit-identical at any count.  The workers park between images and are
    joined when the scorer is dropped (:meth:`Detector.close` does that explicitly).
    """

    def __init__(self, blob: bytes, threads: int | None = None) -> None:
        """Load ``blob`` (an FDT1 container or bare IMSY blob) into the extension."""
        ext = _extension()
        self._scorer = ext.Scorer(blob, threads if threads is not None else default_threads())
        self.native_size = int(self._scorer.native_size)  # floats in Detector.native_matrix
        self.cells = int(self._scorer.cells)
        self.threads = int(self._scorer.threads)  # the count actually used (clamped to 1..16)
        self.target = str(ext.target())  # the SIMD target the module was compiled for
        self.sources: MapSources | None = None  # the layout score_maps reads, once set
        # per bank slot of the layout, the (7, side, side) buffer the scorer writes its planes
        # into (planes 0 and 1, the cell coordinates, written here once)
        self.bank_buffers: dict[int, FloatArray] = {}

    def score(self, native: NDArray[np.floating], *, use_exit: bool = True) -> NDArray[np.float32]:
        """Probabilities for the whole grid (row-major) from ``Detector.native_matrix`` output."""
        values = np.ascontiguousarray(native, dtype=np.float32).reshape(-1)
        return np.asarray(self._scorer.score(values, use_exit), dtype=np.float32)

    def set_sources(self, sources: MapSources) -> None:
        """Tell the scorer where :meth:`score_maps` finds each feature (once per layout).

        ``sources`` comes from :meth:`FeatureExtractor.map_sources` for the model's kept
        columns; the extension checks every feature's grid side against its slot's.  A
        slot the scorer computes (``sources.bank_slots``: a level's context banks) gets a
        buffer here, kept for the scorer's lifetime.
        """
        self._scorer.set_sources(sources.slot_side, sources.feature_slot, sources.feature_index)
        self.bank_buffers = {}
        for slot, raw_slot, column in sources.bank_slots:
            side = int(sources.slot_side[slot])
            buffer = np.zeros((N_BANK_PLANES, side, side), dtype=np.float32)
            _write_coords(buffer)
            self._scorer.set_bank_slot(slot, raw_slot, column, buffer)
            self.bank_buffers[slot] = buffer
        self.sources = sources

    def score_maps(
        self, arrays: Sequence[NDArray[np.float32] | None], *, use_exit: bool = True
    ) -> NDArray[np.float32]:
        """Probabilities read straight from the level maps, no packed copy.

        ``arrays`` are the frame's arrays in the slot order of :meth:`set_sources`
        (:meth:`MapSources.arrays` or :meth:`MapSources.raw_arrays` picks them): float32,
        each a ``(side, side, n)`` bank of any strides or the 1-D broadcast vector, and
        ``None`` for a slot the scorer computes -- its context banks, made inside the pass
        from the level's raw map on the calling thread while the other threads bin, then
        binned from its own buffer.  The same bytes as :meth:`score` on the packed matrix of
        the same values.
        """
        if self.sources is None:
            msg = "set_sources() first: the scorer does not know the arrays' layout"
            raise RuntimeError(msg)
        return np.asarray(self._scorer.score_maps(arrays, use_exit), dtype=np.float32)


def _extension() -> Any:
    try:
        return importlib.import_module("fastdet._native_ext")
    except ImportError as exc:
        package = Path(__file__).resolve().parent
        hint = (
            "the import resolved to the source checkout, which a plain `pip install .` leaves "
            "without the extension (it goes to site-packages, and src/ on sys.path shadows it): "
            "import the installed package, or install editable (`pip install -e .`)"
            if (package.parent / "pyproject.toml").is_file()
            else "reinstall the package with cmake and a C++17 compiler available"
        )
        msg = (
            f"fastdet._native_ext (the C++ scorer, built by `pip install .`) is not importable "
            f"from {package}: {hint}."
        )
        raise ImportError(msg) from exc


def load_scorer(blob: bytes, threads: int | None = None) -> NativeScorer:
    """The C++ scorer for ``blob`` on ``threads`` threads (``None`` = :func:`default_threads`)."""
    return NativeScorer(blob, threads)
