"""In-process C++ scoring through ctypes (the shared library built from ``cpp/``).

The library is looked up in this order: the ``FASTDET_NATIVE_LIB`` environment
variable, then ``fastdet/_native/`` inside the package, then the default CMake
build directories of the source tree (``cpp/build``, ``build``).  When it is not
found :func:`load_native` returns ``None`` and the detector scores with the NumPy
runtime instead, which is bit-identical but slow.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["NativeScorer", "load_native", "native_library_path"]

_LIB_NAMES = {
    "win32": ["fastdet_native.dll", "libfastdet_native.dll"],
    "darwin": ["libfastdet_native.dylib"],
}
_DEFAULT_NAMES = ["libfastdet_native.so"]


def _candidate_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent.parent  # <repo>/src/fastdet -> <repo>
    builds = (
        "cpp/build",
        "build",
        "cpp/build/Release",
        "build/Release",
        "build/pytest-cpp",
        "build/pytest-cpp/Release",
    )
    return [here / "_native", *(root / build for build in builds)]


def native_library_path() -> Path | None:
    """Where the shared library is, or ``None``."""
    env = os.environ.get("FASTDET_NATIVE_LIB")
    if env:
        return Path(env) if Path(env).exists() else None
    names = _LIB_NAMES.get(sys.platform, _DEFAULT_NAMES)
    for directory in _candidate_dirs():
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


class NativeScorer:
    """One loaded model in the C++ scorer."""

    def __init__(self, lib: ctypes.CDLL, blob: bytes) -> None:
        """Load ``blob`` (an FDT1 container or bare IMSY blob) into the C++ scorer."""
        self._lib = lib
        buffer = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        self._handle = lib.fastdet_open(buffer, len(blob))
        if not self._handle:
            msg = "the C++ scorer rejected the model blob"
            raise ValueError(msg)
        self.native_size = int(lib.fastdet_native_size(self._handle))
        self.cells = int(lib.fastdet_cells(self._handle))
        self.target = lib.fastdet_target().decode()

    def __del__(self) -> None:
        """Release the C++ model."""
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.fastdet_close(handle)
            self._handle = None

    def score(self, native: NDArray[np.floating], *, use_exit: bool = True) -> NDArray[np.float32]:
        """Probabilities for the whole grid (row-major) from ``Detector.native_matrix`` output."""
        values = np.ascontiguousarray(native, dtype=np.float32).reshape(-1)
        if values.size != self.native_size:
            msg = f"native matrix has {values.size} values, the model wants {self.native_size}"
            raise ValueError(msg)
        out = np.empty(self.cells, dtype=np.float32)
        status = self._lib.fastdet_score(
            self._handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1 if use_exit else 0,
        )
        if status != 0:
            msg = f"the C++ scorer failed with status {status}"
            raise RuntimeError(msg)
        return out


_LIBS: dict[Path, ctypes.CDLL] = {}


def _bind(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.fastdet_open.restype = ctypes.c_void_p
    lib.fastdet_open.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.fastdet_close.argtypes = [ctypes.c_void_p]
    lib.fastdet_native_size.restype = ctypes.c_size_t
    lib.fastdet_native_size.argtypes = [ctypes.c_void_p]
    lib.fastdet_cells.restype = ctypes.c_size_t
    lib.fastdet_cells.argtypes = [ctypes.c_void_p]
    lib.fastdet_target.restype = ctypes.c_char_p
    lib.fastdet_score.restype = ctypes.c_int
    lib.fastdet_score.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    return lib


def load_native(blob: bytes) -> NativeScorer | None:
    """A :class:`NativeScorer` for ``blob`` when the library can be found, else ``None``.

    The lookup is repeated on every call (a few file checks) so a library built or
    pointed to later in the process is picked up; a found library is loaded once.
    """
    path = native_library_path()
    if path is None:
        return None
    lib = _LIBS.get(path)
    if lib is None:
        lib = _LIBS[path] = _bind(path)
    return NativeScorer(lib, blob)
