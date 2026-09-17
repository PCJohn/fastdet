"""Robust image/mask loading and image<->mask pairing.

``read_image`` decodes through OpenCV first, then Pillow, then ffmpeg, so
formats OpenCV cannot read (and native-codec warning spam) are handled
gracefully.  The front-end never assumes a particular extension is readable.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .features import GRID

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "IMAGE_EXTS",
    "collect_pairs",
    "find_mask_for_image",
    "mask_to_grid_coverage",
    "read_image",
    "read_mask",
]

Image = NDArray[np.uint8]

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".avif",
    ".gif",
}

_FFMPEG_PATH = shutil.which("ffmpeg")
_FFMPEG_TIMEOUT_S = 30


@contextlib.contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Silence C-level stderr (OpenCV/ffmpeg warnings) for the wrapped block."""
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def _try_pillow(path: str) -> Image | None:
    try:
        from PIL import Image as PillowImage  # noqa: PLC0415 -- optional dependency
    except ImportError:
        return None
    try:
        with PillowImage.open(path) as handle:
            rgb = np.array(handle.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return np.asarray(bgr, dtype=np.uint8)
    except Exception:  # noqa: BLE001 -- any decoder failure falls through to ffmpeg
        return None


def _try_ffmpeg(path: str) -> Image | None:
    if _FFMPEG_PATH is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "frame.png"
        try:
            subprocess.run(  # noqa: S603 -- fixed ffmpeg binary resolved from PATH
                [_FFMPEG_PATH, "-y", "-i", path, "-update", "1", "-frames:v", "1", str(out_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_FFMPEG_TIMEOUT_S,
                check=True,
            )
        except Exception:  # noqa: BLE001 -- any ffmpeg failure means "cannot decode"
            return None
        decoded = cv2.imread(str(out_path))
        return None if decoded is None else np.asarray(decoded, dtype=np.uint8)


def _decode(path: str, *, grayscale: bool) -> Image | None:
    with _suppress_native_stderr():
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is not None:
            return np.asarray(img, dtype=np.uint8)
        bgr = _try_pillow(path)
        if bgr is None:
            bgr = _try_ffmpeg(path)
    if bgr is None:
        return None
    if grayscale:
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.asarray(grey, dtype=np.uint8)
    return bgr


def read_image(path: str) -> Image | None:
    """Decode ``path`` as a BGR image, or ``None`` if nothing could read it."""
    return _decode(path, grayscale=False)


def read_mask(path: str) -> Image | None:
    """Decode ``path`` as a single-channel mask, or ``None`` on failure."""
    return _decode(path, grayscale=True)


def find_mask_for_image(fname: str, mask_files_lower: dict[str, str]) -> str | None:
    """Find the ``<name>_mask.<ext>`` counterpart of ``fname`` (case-insensitive)."""
    target = Path(fname)
    stem, suffix = target.stem, target.suffix
    expected = f"{stem}_mask{suffix}".lower()
    if expected in mask_files_lower:
        return mask_files_lower[expected]
    for alt_ext in IMAGE_EXTS:
        alt = f"{stem}_mask{alt_ext}".lower()
        if alt in mask_files_lower:
            return mask_files_lower[alt]
    return None


def _image_names(directory: str) -> list[str]:
    return sorted(p.name for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def collect_pairs(images_dir: str, masks_dir: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair every image with its mask; return ``(pairs, images_without_mask)``."""
    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    mask_files_lower = {name.lower(): name for name in _image_names(masks_dir)}

    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for fname in _image_names(images_dir):
        mask_name = find_mask_for_image(fname, mask_files_lower)
        if mask_name is None:
            missing.append(fname)
        else:
            pairs.append((str(image_root / fname), str(mask_root / mask_name)))
    return pairs, missing


def mask_to_grid_coverage(mask: Image, grid: int = GRID) -> NDArray[np.float32]:
    """Resize a uint8 mask to ``grid x grid`` coverage in [0, 1] (area mean)."""
    mask01 = mask.astype(np.float32) / 255.0
    coverage = cv2.resize(mask01, (grid, grid), interpolation=cv2.INTER_AREA)
    return np.asarray(coverage, dtype=np.float32)
