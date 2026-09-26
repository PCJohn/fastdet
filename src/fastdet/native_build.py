"""Build the C++ scorer and install it next to the package (``fastdet-native-build``).

``Detector`` scores in-process through ``fastdet_native`` (see :mod:`fastdet.native`);
without it the NumPy runtime is used, which is bit-identical but hundreds of times
slower.  This command runs CMake on ``cpp/`` and copies the shared library into
``fastdet/_native/`` inside the installed package, where every installation (editable
or not) finds it.  Requires ``cmake`` and a C++17 compiler; Highway is fetched by CMake.

::

    fastdet-native-build                 # build Release, install into the package
    fastdet-native-build --build-dir b   # reuse a build directory
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import native

__all__ = ["main"]

_LIB_CANDIDATES = (
    "libfastdet_native.so",
    "libfastdet_native.dylib",
    "fastdet_native.dll",
    "Release/fastdet_native.dll",
    "Release/libfastdet_native.so",
)


def _cpp_dir() -> Path | None:
    """The ``cpp/`` source directory: next to ``src/`` in a checkout, else none."""
    here = Path(__file__).resolve().parent
    for root in (here.parent.parent, here.parent):
        if (root / "cpp" / "CMakeLists.txt").exists():
            return root / "cpp"
    return None


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``fastdet-native-build``."""
    parser = argparse.ArgumentParser(
        prog="fastdet-native-build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cpp",
        type=Path,
        default=None,
        help="the cpp/ directory (default: found from the checkout)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=None, help="CMake build directory (default: <cpp>/build)"
    )
    parser.add_argument("--config", default="Release", help="CMake build configuration")
    args = parser.parse_args(argv)
    cpp = args.cpp or _cpp_dir()
    if cpp is None or not (cpp / "CMakeLists.txt").exists():
        print(
            "cpp/CMakeLists.txt not found: run from a checkout of the repository or pass --cpp",
            file=sys.stderr,
        )
        return 2
    cmake = shutil.which("cmake")
    if cmake is None:
        print("cmake is not on PATH", file=sys.stderr)
        return 2
    build_dir = args.build_dir or cpp / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        [cmake, "-S", str(cpp), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={args.config}"],
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            args.config,
            "--target",
            "fastdet_native",
            "--target",
            "fastdet_score",
        ],
    ]
    for step in steps:
        print("[fastdet-native-build] " + " ".join(step))
        result = subprocess.run(step, check=False)  # noqa: S603 -- cmake with fixed arguments
        if result.returncode != 0:
            return result.returncode
    built = next(
        (build_dir / name for name in _LIB_CANDIDATES if (build_dir / name).exists()), None
    )
    if built is None:
        print(
            f"build finished but no fastdet_native library found under {build_dir}", file=sys.stderr
        )
        return 3
    target_dir = Path(native.__file__).resolve().parent / "_native"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / built.name
    shutil.copy2(built, target)
    print(f"[fastdet-native-build] installed {target}")
    found = native.native_library_path()
    print(
        f"[fastdet-native-build] fastdet will load: {found}"
        if found
        else "[fastdet-native-build] WARNING: the library is still not found by fastdet.native"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
