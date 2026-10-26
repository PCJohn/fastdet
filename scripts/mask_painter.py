#!/usr/bin/env python3
"""Interactive binary-mask painter for an image dataset.

The image and mask are kept separately. The mask is always an 8-bit,
single-channel image containing only 0 (background) and 255 (foreground).

Install:
    python -m pip install Pillow

Run:
    python mask_painter.py path/to/images
    python mask_painter.py                  # opens a folder chooser
    python mask_painter.py path/to/images --recursive
    python mask_painter.py path/to/images --review

The output directory is a sibling of the input directory named ``masks``.
For example, ``dataset/images/page.jpg`` produces
``dataset/masks/page_mask.png``. Masks are always saved as PNG, regardless
of the source image's format, so the mask remains exactly binary.

Controls:
    Left drag                 paint foreground (paint mode) / draw threshold
                               rectangle (rect mode) / paint a freeform swath
                               to threshold (lasso mode)
    Left click                flood-fill from the clicked pixel by chroma
                               similarity (wand mode)
    Shift+Left click           in wand mode, subtract the flood-filled region
                               instead of adding it
    Right drag                 erase foreground (brush eraser, works in any
                               mode)
    Mouse wheel                zoom around cursor
    Middle drag                pan
    1 / 2 / 3 / 4               switch to paint / rect / lasso / wand mode
    R                           cycle to the next mode
    I                           invert the mask inside the last thresholded
                               region (rect or lasso)
    Space / Ctrl+S              save current mask, stay on image
    Enter                       save current mask, advance
    PageDown / PageUp           next / previous (asks if unsaved)
    Ctrl+Z / Ctrl+Y             undo / redo
    [ / ] or - / +               decrease / increase brush radius
    Ctrl + wheel                 decrease / increase brush radius (hover canvas)
    Brush spinbox                type or step an exact brush radius (toolbar)
    F                           fit image to window
    X                           clear mask (undoable)
    H                           show help

Rectangle-threshold mode lets you drag out a rectangle over a region; the
pixels inside are split into foreground/background with an automatic
(Otsu) threshold, and the smaller of the two groups is treated as the
foreground (painted white). Press I right afterward to flip that guess if
it picked the wrong side. Every rectangle result is a normal undoable
mask edit.

Lasso-threshold mode is the same idea but for a freehand swath instead of
a rectangle: drag along curved, crumpled, or diagonal text and only the
pixels under your stroke are thresholded (Otsu, same foreground-polarity
guess), so you don't need to chain together many small rectangles. I
inverts just the swath, not its bounding box.

Magic-wand mode flood-fills from a single click using color similarity in
the image's LAB chroma channels (a/b) only, ignoring lightness — this
keeps it robust to shading/gradients on large, flat-colored text or
backgrounds. The "Wand tol" spinbox in the toolbar controls how far a
pixel's chroma can drift from the seed before the fill stops. Shift+click
subtracts the filled region instead of adding it.

By default, images with an existing mask are omitted from the work queue, so
restarting continues at the first unannotated image. ``--review`` includes all
images and loads existing masks for correction.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise SystemExit(
        "Tkinter is unavailable. Install the Tk component supplied with your Python build."
    ) from exc

try:
    from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageTk, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit("Missing dependency. Run: python -m pip install Pillow") from exc


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
LOSSLESS_MASK_EXTENSIONS = {".bmp", ".png", ".tif", ".tiff"}  # kept for legacy-path lookup only
HELP = """Mask Painter controls

Left drag       Paint foreground (paint mode) / draw a threshold
                rectangle (rect mode) / paint a freeform swath to
                threshold (lasso mode)
Left click      Flood-fill from the clicked pixel by chroma similarity
                (wand mode)
Shift+Click     In wand mode, subtract the filled region instead of
                adding it
Right drag      Erase foreground (brush eraser, works in any mode)
Wheel           Zoom around the pointer
Middle drag     Pan the image

1 / 2 / 3 / 4   Switch to paint / rect / lasso / wand mode
R               Cycle to the next mode
I               Invert the mask inside the last thresholded region
                (rect or lasso)

Space           Save checkpoint and stay on this image
Enter           Save and move to the next image
PageDown        Move to next image
PageUp          Move to previous image
Ctrl+Z / Ctrl+Y Undo / redo
[ / ]           Smaller / larger brush
Ctrl+Wheel      Smaller / larger brush (while hovering the canvas)
Brush box       Type an exact brush radius, or use its arrows
F               Fit image in the window
X               Clear the mask

Rectangle-threshold mode: drag a rectangle over a region and it is split
into foreground/background with an automatic threshold, guessing the
smaller group as foreground. Press I right after to invert that guess if
it chose the wrong side.

Lasso-threshold mode: same idea, but you paint a freehand swath (like
brush mode) instead of a rectangle, so you can follow curved, crumpled,
or diagonal text without chaining many small rectangles together. Only
the pixels under your stroke are thresholded; I inverts just that swath.

Magic-wand mode: click once on a flat-colored region (e.g. large text on
a solid background) and it flood-fills based on color similarity in the
LAB a/b (chroma) channels only, so lighting gradients don't stop the
fill as easily as RGB would. Adjust "Wand tol" in the toolbar to grow or
shrink the fill. Shift+click subtracts instead of adding.

An untouched mask is all zero. Save it with Space or Enter to mark a true
negative as annotated.
"""


@dataclass
class UndoItem:
    box: tuple[int, int, int, int]
    before: Image.Image
    after: Image.Image


def output_suffix(source_suffix: str) -> str:
    """Masks are always saved as PNG, regardless of the source format."""
    return ".png"


def mask_path_for(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    """Return the canonical mask path, mirroring subdirectories if recursive."""
    relative = image_path.relative_to(image_root)
    suffix = output_suffix(image_path.suffix)
    return mask_root / relative.parent / f"{image_path.stem}_mask{suffix}"


def possible_mask_paths(image_path: Path, image_root: Path, mask_root: Path) -> list[Path]:
    """Include a same-suffix legacy path when the canonical mask is PNG."""
    canonical = mask_path_for(image_path, image_root, mask_root)
    relative = image_path.relative_to(image_root)
    same_suffix = mask_root / relative.parent / f"{image_path.stem}_mask{image_path.suffix}"
    return list(dict.fromkeys((canonical, same_suffix)))


def existing_mask_path(image_path: Path, image_root: Path, mask_root: Path) -> Path | None:
    return next(
        (path for path in possible_mask_paths(image_path, image_root, mask_root) if path.is_file()),
        None,
    )


def find_images(root: Path, recursive: bool, mask_root: Path) -> list[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    images: list[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if mask_root == resolved.parent or mask_root in resolved.parents:
            continue
        images.append(resolved)
    return sorted(images, key=lambda path: str(path.relative_to(root)).casefold())


def atomic_save_mask(mask: Image.Image, destination: Path) -> None:
    """Write a strictly binary mask atomically so interruption cannot corrupt it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = mask.convert("L").point(lambda value: 255 if value else 0, mode="L")
    formats = {".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}
    image_format = formats.get(destination.suffix.lower(), "PNG")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        binary.save(temporary, format=image_format)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def otsu_threshold(histogram: list[int]) -> int:
    """Return the Otsu threshold (0-255) that best splits a 256-bin histogram.

    Pure-Python implementation (no numpy dependency): the histogram itself
    is only 256 entries, so this is cheap regardless of the region size the
    caller sampled it from.
    """
    total = sum(histogram)
    if total == 0:
        return 127
    sum_total = sum(level * count for level, count in enumerate(histogram))

    weight_background = 0
    sum_background = 0
    best_variance = -1.0
    best_threshold = 0
    for level, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if variance_between > best_variance:
            best_variance = variance_between
            best_threshold = level
    return best_threshold


def threshold_region(region: Image.Image) -> Image.Image:
    """Binarize an RGB/L crop with Otsu's method, guessing foreground polarity.

    Foreground (255) is assigned to whichever side of the threshold covers
    fewer pixels, on the assumption that a hand-drawn rectangle usually
    contains mostly background with a smaller object of interest. Callers
    can flip this guess with an invert action.
    """
    grayscale = region.convert("L")
    histogram = grayscale.histogram()
    threshold = otsu_threshold(histogram)
    low_count = sum(histogram[: threshold + 1])
    high_count = sum(histogram) - low_count
    if low_count <= high_count:
        return grayscale.point(lambda value: 255 if value <= threshold else 0, mode="L")
    return grayscale.point(lambda value: 255 if value > threshold else 0, mode="L")


def threshold_masked_region(region: Image.Image, region_mask: Image.Image) -> Image.Image:
    """Like ``threshold_region``, but the Otsu histogram only counts pixels
    where ``region_mask`` is nonzero.

    Used by lasso-threshold mode: the caller draws an arbitrary freehand
    swath (e.g. following curved or diagonal text) instead of a rectangle,
    so the crop's bounding box usually contains a lot of untouched pixels
    that shouldn't skew the threshold. The whole crop is still classified
    (cheap, and simplest to composite), but the histogram driving that
    classification is restricted to the swath itself.
    """
    grayscale = region.convert("L")
    histogram = [0] * 256
    for value, keep in zip(grayscale.getdata(), region_mask.getdata()):
        if keep:
            histogram[value] += 1
    if sum(histogram) == 0:
        histogram = grayscale.histogram()
    threshold = otsu_threshold(histogram)
    low_count = sum(histogram[: threshold + 1])
    high_count = sum(histogram) - low_count
    if low_count <= high_count:
        return grayscale.point(lambda value: 255 if value <= threshold else 0, mode="L")
    return grayscale.point(lambda value: 255 if value > threshold else 0, mode="L")


class MaskPainter:
    MIN_ZOOM = 0.02
    MAX_ZOOM = 64.0
    ZOOM_STEP = 1.25
    MAX_HISTORY = 40

    def __init__(
        self,
        root: tk.Tk,
        image_root: Path,
        *,
        recursive: bool,
        review: bool,
        brush_radius: int,
        opacity: float,
    ) -> None:
        self.root = root
        self.image_root = image_root.resolve()
        self.mask_root = self.image_root.parent / "masks"
        self.recursive = recursive
        self.review = review
        self.brush_radius = brush_radius
        self.opacity = opacity

        self.all_images = find_images(self.image_root, recursive, self.mask_root.resolve())
        if not self.all_images:
            raise ValueError(f"No supported images found in {self.image_root}")
        if review:
            self.images = self.all_images
        else:
            self.images = [
                image
                for image in self.all_images
                if existing_mask_path(image, self.image_root, self.mask_root) is None
            ]
        destinations: dict[Path, Path] = {}
        for image in self.all_images:
            destination = mask_path_for(image, self.image_root, self.mask_root)
            previous = destinations.get(destination)
            if previous is not None:
                raise ValueError(
                    "Two source images map to the same lossless mask filename:\n"
                    f"{previous}\n{image}\n\nRename one source image before annotating."
                )
            destinations[destination] = image

        self._base_images: list[Path] = list(self.images)
        self._shuffled_images: list[Path] | None = None
        self.shuffled = False

        self.index = 0
        self.image_path: Path | None = None
        self.image: Image.Image | None = None
        self.mask: Image.Image | None = None
        self.mask_path: Path | None = None
        self.dirty = False
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.photo: ImageTk.PhotoImage | None = None
        self.image_item: int | None = None
        self.cursor_item: int | None = None
        self.last_image_point: tuple[float, float] | None = None
        self.pan_start: tuple[int, int, float, float] | None = None
        self.stroke_before: Image.Image | None = None
        self.undo_stack: list[UndoItem] = []
        self.redo_stack: list[UndoItem] = []
        self.render_pending = False
        self.fit_pending = False

        self.mode = "paint"  # "paint", "rect", "lasso", or "wand"
        self.rect_start_canvas: tuple[int, int] | None = None
        self.rect_item: int | None = None
        self.last_threshold_box: tuple[int, int, int, int] | None = None
        self.last_threshold_region: Image.Image | None = None
        self.lasso_region: Image.Image | None = None
        self._wand_chroma: Image.Image | None = None

        self.status_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.brush_var = tk.StringVar()
        self._build_ui()
        self._bind_events()

        if not self.images:
            self._show_complete()
        else:
            self.load_index(0)

    def _build_ui(self) -> None:
        self.root.title("Binary Mask Painter")
        self.root.geometry("1280x820")
        self.root.minsize(700, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        toolbar = ttk.Frame(self.root, padding=(6, 5))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Save  [Space]", command=self.save).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save + Next  [Enter]", command=self.save_and_next).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        ttk.Button(toolbar, text="Previous", command=self.previous_image).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar, text="Clear", command=self.clear_mask).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar, text="Fit", command=self.fit_to_window).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.mode_var = tk.StringVar(value=self.mode)
        for label, mode_key in (
            ("Paint [1]", "paint"),
            ("Rect [2]", "rect"),
            ("Lasso [3]", "lasso"),
            ("Wand [4]", "wand"),
        ):
            ttk.Radiobutton(
                toolbar,
                text=label,
                value=mode_key,
                variable=self.mode_var,
                command=self._on_mode_selected,
                style="Toolbutton",
            ).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Brush:").pack(side=tk.LEFT, padx=(8, 2))
        self.brush_size_var = tk.IntVar(value=self.brush_radius)
        ttk.Spinbox(
            toolbar,
            from_=1,
            to=512,
            width=4,
            textvariable=self.brush_size_var,
            command=self._on_brush_spinbox_changed,
        ).pack(side=tk.LEFT)
        self.brush_size_var.trace_add("write", self._on_brush_var_written)
        ttk.Label(toolbar, text="Wand tol:").pack(side=tk.LEFT, padx=(8, 2))
        self.wand_tolerance = tk.IntVar(value=24)
        ttk.Spinbox(toolbar, from_=1, to=150, width=4, textvariable=self.wand_tolerance).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Invert Region  [I]", command=self.invert_last_threshold).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.shuffle_button = ttk.Button(toolbar, text="Shuffle", command=self.toggle_shuffle)
        self.shuffle_button.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Help", command=self.show_help).pack(side=tk.RIGHT)

        nav = ttk.Frame(self.root, padding=(6, 0, 6, 5))
        nav.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(nav, text="Jump to:").pack(side=tk.LEFT)
        self.slider = ttk.Scale(
            nav,
            from_=1,
            to=max(1, len(self.images)),
            orient=tk.HORIZONTAL,
            command=self._on_slider_move,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.slider_label = ttk.Label(nav, text=f"1 / {max(1, len(self.images))}")
        self.slider_label.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            self.root,
            background="#202124",
            cursor="crosshair",
            highlightthickness=0,
            takefocus=True,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(self.root, padding=(7, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.progress_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Label(status, textvariable=self.brush_var).pack(side=tk.RIGHT)

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<ButtonPress-2>", self.begin_pan)
        self.canvas.bind("<B2-Motion>", self.continue_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_at(event.x, event.y, 1))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_at(event.x, event.y, -1))
        self.canvas.bind("<Control-MouseWheel>", self.on_brush_scroll)
        self.canvas.bind("<Control-Button-4>", lambda event: self.change_brush(1))
        self.canvas.bind("<Control-Button-5>", lambda event: self.change_brush(-1))
        self.canvas.bind("<Motion>", self.update_cursor)
        self.canvas.bind("<Leave>", lambda _event: self.hide_cursor())
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", lambda _event: self.canvas.focus_set(), add="+")

        self.root.bind("<space>", lambda _event: self._key(self.save))
        self.root.bind("<Control-s>", lambda _event: self._key(self.save))
        self.root.bind("<Return>", lambda _event: self._key(self.save_and_next))
        self.root.bind("<KP_Enter>", lambda _event: self._key(self.save_and_next))
        self.root.bind("<Next>", lambda _event: self._key(self.next_image))
        self.root.bind("<Prior>", lambda _event: self._key(self.previous_image))
        self.root.bind("<Control-z>", lambda _event: self._key(self.undo))
        self.root.bind("<Control-y>", lambda _event: self._key(self.redo))
        self.root.bind("<bracketleft>", lambda _event: self._key(lambda: self.change_brush(-1)))
        self.root.bind("<bracketright>", lambda _event: self._key(lambda: self.change_brush(1)))
        self.root.bind("<minus>", lambda _event: self._key(lambda: self.change_brush(-1)))
        self.root.bind("<equal>", lambda _event: self._key(lambda: self.change_brush(1)))
        self.root.bind("<Key-f>", lambda _event: self._key(self.fit_to_window))
        self.root.bind("<Key-x>", lambda _event: self._key(self.clear_mask))
        self.root.bind("<Key-h>", lambda _event: self._key(self.show_help))
        self.root.bind("<Key-r>", lambda _event: self._key(self.cycle_mode))
        self.root.bind("<Key-i>", lambda _event: self._key(self.invert_last_threshold))
        self.root.bind("<Key-1>", lambda _event: self._key(lambda: self.set_mode("paint")))
        self.root.bind("<Key-2>", lambda _event: self._key(lambda: self.set_mode("rect")))
        self.root.bind("<Key-3>", lambda _event: self._key(lambda: self.set_mode("lasso")))
        self.root.bind("<Key-4>", lambda _event: self._key(lambda: self.set_mode("wand")))

    @staticmethod
    def _key(action: object) -> str:
        if callable(action):
            action()
        return "break"

    def load_index(self, index: int) -> None:
        if not 0 <= index < len(self.images):
            self._show_complete()
            return
        path = self.images[index]
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            messagebox.showerror("Cannot open image", f"{path}\n\n{exc}")
            if index + 1 < len(self.images):
                self.load_index(index + 1)
            return

        canonical = mask_path_for(path, self.image_root, self.mask_root)
        existing = existing_mask_path(path, self.image_root, self.mask_root)
        if existing is None:
            mask = Image.new("L", image.size, 0)
        else:
            try:
                with Image.open(existing) as opened_mask:
                    loaded = opened_mask.convert("L")
                if loaded.size != image.size:
                    raise ValueError(
                        f"mask is {loaded.size[0]}x{loaded.size[1]}, image is "
                        f"{image.width}x{image.height}"
                    )
                mask = loaded.point(lambda value: 255 if value >= 128 else 0, mode="L")
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                messagebox.showerror("Invalid mask", f"{existing}\n\n{exc}")
                mask = Image.new("L", image.size, 0)

        self.index = index
        self.image_path = path
        self.image = image
        self.mask = mask
        self.mask_path = canonical
        self.dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.last_threshold_box = None
        self.last_threshold_region = None
        self.rect_start_canvas = None
        self.rect_item = None
        self.lasso_region = None
        self._wand_chroma = None
        self.canvas.delete("all")
        self.image_item = None
        self.cursor_item = None
        self.root.title(f"Binary Mask Painter — {path.name}")
        self.update_status("loaded existing mask" if existing else "new zero mask")
        self._sync_slider()
        self.fit_pending = True
        self.root.after_idle(self.fit_to_window)

    def _sync_slider(self) -> None:
        """Reflect the current index/queue length on the jump slider."""
        total = max(1, len(self.images))
        self.slider.configure(to=total)
        self.slider.set(self.index + 1)
        self.slider_label.configure(text=f"{self.index + 1} / {len(self.images)}")

    def _on_slider_move(self, value: str) -> None:
        # Fires continuously while dragging; only update the label so we
        # don't thrash image loads (and dirty-save prompts) mid-drag.
        total = len(self.images)
        target = max(1, min(round(float(value)), max(1, total)))
        self.slider_label.configure(text=f"{target} / {total}")

    def _on_slider_release(self, _event: tk.Event) -> None:
        if not self.images:
            return
        target = round(self.slider.get()) - 1
        target = max(0, min(target, len(self.images) - 1))
        if target == self.index:
            self._sync_slider()
            return
        if not self._resolve_dirty():
            self._sync_slider()
            return
        self.load_index(target)

    def toggle_shuffle(self) -> None:
        if not self.images:
            return
        if not self._resolve_dirty():
            return
        current_path = self.image_path

        if not self.shuffled:
            if self._shuffled_images is None:
                shuffled = list(self._base_images)
                random.shuffle(shuffled)
                self._shuffled_images = shuffled
            self.images = self._shuffled_images
            self.shuffled = True
            self.shuffle_button.configure(text="Shuffle: On")
        else:
            self.images = self._base_images
            self.shuffled = False
            self.shuffle_button.configure(text="Shuffle")

        if current_path is not None and current_path in self.images:
            new_index = self.images.index(current_path)
        else:
            new_index = 0
        self.load_index(new_index)

    def _show_complete(self) -> None:
        self.index = len(self.images)
        self.image_path = None
        self.image = None
        self.mask = None
        self.mask_path = None
        self.photo = None
        self.dirty = False
        self.canvas.delete("all")
        remaining = [
            path
            for path in self.all_images
            if existing_mask_path(path, self.image_root, self.mask_root) is None
        ]
        if self.review:
            completion_text = "End of review queue.\nPress PageUp to go back."
        elif remaining:
            completion_text = (
                f"End of current queue; {len(remaining)} image(s) remain unannotated.\n"
                "Press PageUp to go back or restart to resume at the first one."
            )
        else:
            completion_text = "All images are annotated.\nUse --review to inspect existing masks."
        self.canvas.create_text(
            max(1, self.canvas.winfo_width() // 2),
            max(1, self.canvas.winfo_height() // 2),
            text=completion_text,
            fill="white",
            font=("TkDefaultFont", 18),
            justify=tk.CENTER,
        )
        annotated = len(self.all_images) - len(remaining)
        self.progress_var.set(f"Annotated: {annotated}/{len(self.all_images)}")
        self.status_var.set(str(self.mask_root))
        self.brush_var.set("")

    def update_status(self, message: str = "") -> None:
        if self.image is None or self.image_path is None:
            return
        relative = self.image_path.relative_to(self.image_root)
        state = "modified" if self.dirty else message or "saved"
        self.progress_var.set(f"{self.index + 1}/{len(self.images)}  {relative}")
        self.status_var.set(f"{self.image.width}×{self.image.height}  •  {state}")
        mode_labels = {
            "paint": "paint",
            "rect": "rect threshold",
            "lasso": "lasso threshold",
            "wand": "magic wand",
        }
        mode_label = mode_labels.get(self.mode, self.mode)
        self.brush_var.set(
            f"mode: {mode_label}  •  brush {self.brush_radius}px  •  zoom {self.zoom:.2f}×"
        )

    def request_render(self) -> None:
        if self.render_pending:
            return
        self.render_pending = True
        self.root.after_idle(self.render)

    def render(self) -> None:
        self.render_pending = False
        if self.image is None or self.mask is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        x0 = max(0, int((-self.offset_x) / self.zoom))
        y0 = max(0, int((-self.offset_y) / self.zoom))
        x1 = min(self.image.width, int((canvas_width - self.offset_x) / self.zoom) + 2)
        y1 = min(self.image.height, int((canvas_height - self.offset_y) / self.zoom) + 2)
        if x1 <= x0 or y1 <= y0:
            if self.image_item is not None:
                self.canvas.itemconfigure(self.image_item, state=tk.HIDDEN)
            return

        crop_box = (x0, y0, x1, y1)
        target_size = (
            max(1, round((x1 - x0) * self.zoom)),
            max(1, round((y1 - y0) * self.zoom)),
        )
        base = self.image.crop(crop_box).resize(target_size, Image.Resampling.BILINEAR)
        mask = self.mask.crop(crop_box).resize(target_size, Image.Resampling.NEAREST)
        alpha_level = round(255 * self.opacity)
        alpha = mask.point(lambda value: alpha_level if value else 0, mode="L")
        overlay = Image.new("RGB", target_size, (255, 35, 35))
        composite = Image.composite(overlay, base, alpha)
        if self.lasso_region is not None:
            lasso_crop = self.lasso_region.crop(crop_box).resize(
                target_size, Image.Resampling.NEAREST
            )
            lasso_alpha = lasso_crop.point(lambda value: 130 if value else 0, mode="L")
            lasso_overlay = Image.new("RGB", target_size, (60, 160, 255))
            composite = Image.composite(lasso_overlay, composite, lasso_alpha)
        self.photo = ImageTk.PhotoImage(composite)
        screen_x = self.offset_x + x0 * self.zoom
        screen_y = self.offset_y + y0 * self.zoom
        if self.image_item is None:
            self.image_item = self.canvas.create_image(
                screen_x, screen_y, image=self.photo, anchor=tk.NW, tags=("image",)
            )
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo, state=tk.NORMAL)
            self.canvas.coords(self.image_item, screen_x, screen_y)
        self.canvas.tag_lower(self.image_item)
        self.update_status()

    def fit_to_window(self) -> None:
        if self.image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.zoom = min(canvas_width / self.image.width, canvas_height / self.image.height)
        self.zoom = max(self.MIN_ZOOM, min(self.zoom, self.MAX_ZOOM))
        self.offset_x = (canvas_width - self.image.width * self.zoom) / 2.0
        self.offset_y = (canvas_height - self.image.height * self.zoom) / 2.0
        self.fit_pending = False
        self.request_render()

    def on_canvas_resize(self, _event: tk.Event) -> None:
        if self.fit_pending:
            self.root.after_idle(self.fit_to_window)
        else:
            self.request_render()

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if self.image is None:
            return None
        image_x = (x - self.offset_x) / self.zoom
        image_y = (y - self.offset_y) / self.zoom
        if 0 <= image_x < self.image.width and 0 <= image_y < self.image.height:
            return image_x, image_y
        return None

    def set_mode(self, mode: str) -> None:
        if mode not in ("paint", "rect", "lasso", "wand"):
            return
        self.mode = mode
        self.mode_var.set(mode)
        cursors = {"paint": "crosshair", "rect": "tcross", "lasso": "crosshair", "wand": "hand2"}
        self.canvas.configure(cursor=cursors[mode])
        self.hide_cursor()
        self.rect_start_canvas = None
        if self.rect_item is not None:
            self.canvas.delete(self.rect_item)
            self.rect_item = None
        if self.lasso_region is not None:
            self.lasso_region = None
            self.request_render()
        self.last_image_point = None
        self.update_status()

    def _on_mode_selected(self) -> None:
        self.set_mode(self.mode_var.get())

    def cycle_mode(self) -> None:
        order = ["paint", "rect", "lasso", "wand"]
        current = order.index(self.mode) if self.mode in order else 0
        self.set_mode(order[(current + 1) % len(order)])

    def on_left_press(self, event: tk.Event) -> None:
        if self.mode == "rect":
            self.begin_rect(event)
        elif self.mode == "lasso":
            self.begin_lasso(event)
        elif self.mode == "wand":
            self.apply_wand(event)
        else:
            self.begin_stroke(event, 255)

    def on_left_drag(self, event: tk.Event) -> None:
        if self.mode == "rect":
            self.continue_rect(event)
        elif self.mode == "lasso":
            self.continue_lasso(event)
        elif self.mode == "wand":
            pass  # single-click tool; dragging does nothing
        else:
            self.continue_stroke(event, 255)

    def on_left_release(self, event: tk.Event) -> None:
        if self.mode == "rect":
            self.end_rect(event)
        elif self.mode == "lasso":
            self.end_lasso(event)
        elif self.mode == "wand":
            pass
        else:
            self.end_stroke(event)

    def on_right_press(self, event: tk.Event) -> None:
        self.begin_stroke(event, 0)

    def on_right_drag(self, event: tk.Event) -> None:
        self.continue_stroke(event, 0)

    def on_right_release(self, event: tk.Event) -> None:
        self.end_stroke(event)

    def begin_rect(self, event: tk.Event) -> None:
        if self.image is None:
            return
        self.canvas.focus_set()
        self.rect_start_canvas = (event.x, event.y)
        if self.rect_item is not None:
            self.canvas.delete(self.rect_item)
        self.rect_item = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#3df23d", width=2, dash=(5, 3)
        )

    def continue_rect(self, event: tk.Event) -> None:
        if self.rect_start_canvas is None or self.rect_item is None:
            return
        start_x, start_y = self.rect_start_canvas
        self.canvas.coords(self.rect_item, start_x, start_y, event.x, event.y)

    def end_rect(self, event: tk.Event) -> None:
        if self.rect_start_canvas is None:
            return
        start = self.rect_start_canvas
        self.rect_start_canvas = None
        if self.rect_item is not None:
            self.canvas.delete(self.rect_item)
            self.rect_item = None
        self._apply_rect_threshold(start, (event.x, event.y))

    def _canvas_to_image_clamped(self, x: float, y: float) -> tuple[float, float]:
        image_x = (x - self.offset_x) / self.zoom
        image_y = (y - self.offset_y) / self.zoom
        image_x = max(0.0, min(float(self.image.width), image_x))
        image_y = max(0.0, min(float(self.image.height), image_y))
        return image_x, image_y

    def _apply_rect_threshold(
        self, start_canvas: tuple[int, int], end_canvas: tuple[int, int]
    ) -> None:
        if self.image is None or self.mask is None:
            return
        point_a = self._canvas_to_image_clamped(*start_canvas)
        point_b = self._canvas_to_image_clamped(*end_canvas)
        x0, x1 = sorted((point_a[0], point_b[0]))
        y0, y1 = sorted((point_a[1], point_b[1]))
        box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            self.update_status("rectangle too small, ignored")
            return

        binary = threshold_region(self.image.crop(box))
        before = self.mask.copy()
        self.mask.paste(binary, box)
        self._record_change(before)
        self.last_threshold_box = box
        self.last_threshold_region = None  # None = whole rectangle is the region
        self.dirty = True
        self.request_render()
        self.update_status(f"thresholded {box[2] - box[0]}×{box[3] - box[1]} region")

    def invert_last_threshold(self) -> None:
        if self.mask is None or self.last_threshold_box is None:
            self.update_status("no thresholded region to invert yet")
            return
        box = self.last_threshold_box
        before = self.mask.copy()
        current_crop = self.mask.crop(box)
        inverted = ImageChops.invert(current_crop)
        if self.last_threshold_region is not None:
            # Lasso result: only invert the freehand swath, not its bbox.
            inverted = Image.composite(inverted, current_crop, self.last_threshold_region)
        self.mask.paste(inverted, box)
        self._record_change(before)
        self.dirty = True
        self.request_render()
        self.update_status("inverted last thresholded region")

    def begin_lasso(self, event: tk.Event) -> None:
        if self.image is None:
            return
        self.canvas.focus_set()
        self.lasso_region = Image.new("L", self.image.size, 0)
        point = self.canvas_to_image(event.x, event.y)
        self.last_image_point = point
        if point is not None:
            self._paint_into(self.lasso_region, point, point, 255)
        self.request_render()

    def continue_lasso(self, event: tk.Event) -> None:
        if self.lasso_region is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        start = self.last_image_point if self.last_image_point is not None else point
        self._paint_into(self.lasso_region, start, point, 255)
        self.last_image_point = point
        self.request_render()

    def end_lasso(self, _event: tk.Event) -> None:
        self.last_image_point = None
        region = self.lasso_region
        self.lasso_region = None
        if region is not None:
            self._apply_lasso_threshold(region)
        self.request_render()

    def _apply_lasso_threshold(self, region: Image.Image) -> None:
        if self.image is None or self.mask is None:
            return
        bbox = region.getbbox()
        if bbox is None:
            self.update_status("lasso stroke too small, ignored")
            return
        region_crop = region.crop(bbox)
        image_crop = self.image.crop(bbox)
        binary = threshold_masked_region(image_crop, region_crop)
        mask_crop = self.mask.crop(bbox)
        composited = Image.composite(binary, mask_crop, region_crop)
        before = self.mask.copy()
        self.mask.paste(composited, bbox)
        self._record_change(before)
        self.last_threshold_box = bbox
        self.last_threshold_region = region_crop
        self.dirty = True
        self.request_render()
        self.update_status(f"lasso-thresholded {bbox[2] - bbox[0]}×{bbox[3] - bbox[1]} region")

    def apply_wand(self, event: tk.Event) -> None:
        if self.image is None or self.mask is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.canvas.focus_set()
        seed = (int(point[0]), int(point[1]))
        subtract = bool(event.state & 0x0001)  # Shift held -> subtract
        region = self._wand_region_mask(seed)
        if region is None or region.getbbox() is None:
            self.update_status("magic wand: nothing matched at that tolerance")
            return
        before = self.mask.copy()
        fill_value = 0 if subtract else 255
        solid = Image.new("L", self.image.size, fill_value)
        result = Image.composite(solid, self.mask, region)
        self.mask.paste(result, (0, 0))
        self._record_change(before)
        self.dirty = True
        self.request_render()
        action = "subtracted" if subtract else "added"
        self.update_status(f"magic wand {action} region (tol {self.wand_tolerance.get()})")

    def _wand_region_mask(self, seed: tuple[int, int]) -> Image.Image | None:
        if self.image is None:
            return None
        x, y = seed
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            return None
        if self._wand_chroma is None:
            lab = self.image.convert("LAB")
            _, a, b = lab.split()
            zero = Image.new("L", self.image.size, 0)
            self._wand_chroma = Image.merge("RGB", (a, b, zero))
        filled = self._wand_chroma.copy()
        try:
            tolerance = float(self.wand_tolerance.get())
        except (tk.TclError, ValueError):
            tolerance = 24.0
        ImageDraw.floodfill(filled, (x, y), (0, 0, 255), thresh=tolerance)
        marker = filled.split()[2]
        return marker.point(lambda value: 255 if value == 255 else 0, mode="L")

    def begin_stroke(self, event: tk.Event, value: int) -> None:
        if self.mask is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            self.last_image_point = None
            return
        self.canvas.focus_set()
        self.stroke_before = self.mask.copy()
        self.last_image_point = point
        self._paint_segment(point, point, value)

    def continue_stroke(self, event: tk.Event, value: int) -> None:
        if self.mask is None or self.last_image_point is None:
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            self.last_image_point = None
            return
        self._paint_segment(self.last_image_point, point, value)
        self.last_image_point = point

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_image_point = None
        if self.stroke_before is not None:
            self._record_change(self.stroke_before)
        self.stroke_before = None

    def _paint_segment(
        self, start: tuple[float, float], end: tuple[float, float], value: int
    ) -> None:
        if self.mask is None:
            return
        self._paint_into(self.mask, start, end, value)
        self.dirty = True
        self.request_render()

    def _paint_into(
        self,
        target: Image.Image,
        start: tuple[float, float],
        end: tuple[float, float],
        value: int,
    ) -> None:
        draw = ImageDraw.Draw(target)
        radius = self.brush_radius
        draw.line((start, end), fill=value, width=radius * 2 + 1)
        x, y = end
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)

    def _record_change(self, before: Image.Image) -> None:
        if self.mask is None:
            return
        box = ImageChops.difference(before, self.mask).getbbox()
        if box is None:
            return
        self.undo_stack.append(UndoItem(box, before.crop(box), self.mask.crop(box)))
        if len(self.undo_stack) > self.MAX_HISTORY:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.update_status("modified")

    def undo(self) -> None:
        if self.mask is None or not self.undo_stack:
            return
        item = self.undo_stack.pop()
        self.mask.paste(item.before, item.box)
        self.redo_stack.append(item)
        self.dirty = True
        self.request_render()

    def redo(self) -> None:
        if self.mask is None or not self.redo_stack:
            return
        item = self.redo_stack.pop()
        self.mask.paste(item.after, item.box)
        self.undo_stack.append(item)
        self.dirty = True
        self.request_render()

    def clear_mask(self) -> None:
        if self.mask is None or self.mask.getbbox() is None:
            return
        if not messagebox.askyesno("Clear mask", "Erase the entire current mask?"):
            return
        before = self.mask.copy()
        self.mask.paste(0, (0, 0, self.mask.width, self.mask.height))
        self._record_change(before)
        self.dirty = True
        self.request_render()

    def change_brush(self, direction: int) -> None:
        if direction > 0:
            self.brush_radius = min(
                512, max(self.brush_radius + 1, round(self.brush_radius * 1.25))
            )
        else:
            self.brush_radius = max(1, min(self.brush_radius - 1, round(self.brush_radius / 1.25)))
        self._sync_brush_var()
        self.update_status()
        self._refresh_cursor_preview()

    def set_brush_radius(self, radius: int) -> None:
        """Set the brush radius directly, e.g. from the toolbar spinbox."""
        clamped = max(1, min(512, radius))
        if clamped == self.brush_radius:
            return
        self.brush_radius = clamped
        self.update_status()
        self._refresh_cursor_preview()

    def _refresh_cursor_preview(self) -> None:
        if self.cursor_item is None:
            return
        pointer_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        pointer_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        fake_event = type("_Pos", (), {"x": pointer_x, "y": pointer_y})()
        self.update_cursor(fake_event)

    def _sync_brush_var(self) -> None:
        if self.brush_size_var.get() != self.brush_radius:
            self.brush_size_var.set(self.brush_radius)

    def _on_brush_spinbox_changed(self) -> None:
        try:
            self.set_brush_radius(self.brush_size_var.get())
        except tk.TclError:
            pass

    def _on_brush_var_written(self, *_args) -> None:
        # Fires on typing too, not just the spinbox arrows, so edits made by
        # hand take effect without needing to press Enter or click away.
        try:
            value = self.brush_size_var.get()
        except tk.TclError:
            return
        self.set_brush_radius(value)

    def on_brush_scroll(self, event: tk.Event) -> str:
        self.change_brush(1 if event.delta > 0 else -1)
        return "break"

    def update_cursor(self, event: tk.Event) -> None:
        if self.image is None:
            return
        radius = self.brush_radius * self.zoom
        coordinates = (event.x - radius, event.y - radius, event.x + radius, event.y + radius)
        if self.cursor_item is None:
            self.cursor_item = self.canvas.create_oval(
                *coordinates, outline="white", width=1, tags=("cursor",)
            )
        else:
            self.canvas.coords(self.cursor_item, *coordinates)
            self.canvas.itemconfigure(self.cursor_item, state=tk.NORMAL)
        self.canvas.tag_raise(self.cursor_item)

    def hide_cursor(self) -> None:
        if self.cursor_item is not None:
            self.canvas.itemconfigure(self.cursor_item, state=tk.HIDDEN)

    def on_mousewheel(self, event: tk.Event) -> str:
        self.zoom_at(event.x, event.y, 1 if event.delta > 0 else -1)
        return "break"

    def zoom_at(self, canvas_x: float, canvas_y: float, direction: int) -> None:
        if self.image is None:
            return
        old_zoom = self.zoom
        factor = self.ZOOM_STEP if direction > 0 else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(old_zoom * factor, self.MAX_ZOOM))
        image_x = (canvas_x - self.offset_x) / old_zoom
        image_y = (canvas_y - self.offset_y) / old_zoom
        self.zoom = new_zoom
        self.offset_x = canvas_x - image_x * new_zoom
        self.offset_y = canvas_y - image_y * new_zoom
        self.request_render()

    def begin_pan(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y, self.offset_x, self.offset_y)
        self.canvas.configure(cursor="fleur")

    def continue_pan(self, event: tk.Event) -> None:
        if self.pan_start is None:
            return
        start_x, start_y, offset_x, offset_y = self.pan_start
        self.offset_x = offset_x + event.x - start_x
        self.offset_y = offset_y + event.y - start_y
        self.request_render()

    def end_pan(self, _event: tk.Event) -> None:
        self.pan_start = None
        self.canvas.configure(cursor="crosshair")

    def save(self) -> bool:
        if self.mask is None or self.mask_path is None:
            return False
        try:
            atomic_save_mask(self.mask, self.mask_path)
        except OSError as exc:
            messagebox.showerror("Could not save mask", f"{self.mask_path}\n\n{exc}")
            return False
        self.dirty = False
        self.update_status("saved zero mask" if self.mask.getbbox() is None else "saved")
        return True

    def save_and_next(self) -> None:
        if self.save():
            self._advance(1, ask_about_dirty=False)

    def next_image(self) -> None:
        self._advance(1, ask_about_dirty=True)

    def previous_image(self) -> None:
        self._advance(-1, ask_about_dirty=True)

    def _advance(self, delta: int, ask_about_dirty: bool) -> None:
        if ask_about_dirty and not self._resolve_dirty():
            return
        target = self.index + delta
        if 0 <= target < len(self.images):
            self.load_index(target)
        elif target >= len(self.images):
            self._show_complete()

    def _resolve_dirty(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved mask", "Save changes to the current mask before continuing?"
        )
        if answer is None:
            return False
        return self.save() if answer else True

    def show_help(self) -> None:
        messagebox.showinfo("Mask Painter help", HELP)

    def on_close(self) -> None:
        if self._resolve_dirty():
            self.root.destroy()


def choose_directory() -> Path | None:
    chooser = tk.Tk()
    chooser.withdraw()
    selected = filedialog.askdirectory(title="Choose the image dataset folder", parent=chooser)
    chooser.destroy()
    return Path(selected).resolve() if selected else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively paint single-channel binary masks for an image folder."
    )
    parser.add_argument("folder", nargs="?", type=Path, help="folder containing target images")
    parser.add_argument("--recursive", action="store_true", help="include image subdirectories")
    parser.add_argument(
        "--review",
        action="store_true",
        help="include already annotated images and load their masks",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=8,
        metavar="PIXELS",
        help="initial brush radius in image pixels (default: 8)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="red mask-overlay opacity, from 0 to 1 (default: 0.45)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.brush < 1:
        raise SystemExit("--brush must be at least 1")
    if not 0.0 <= args.opacity <= 1.0 or not math_is_finite(args.opacity):
        raise SystemExit("--opacity must be between 0 and 1")
    folder = args.folder.resolve() if args.folder else choose_directory()
    if folder is None:
        return
    if not folder.is_dir():
        raise SystemExit(f"Image folder does not exist: {folder}")

    root = tk.Tk()
    try:
        MaskPainter(
            root,
            folder,
            recursive=args.recursive,
            review=args.review,
            brush_radius=args.brush,
            opacity=args.opacity,
        )
    except ValueError as exc:
        root.withdraw()
        messagebox.showerror("Mask Painter", str(exc), parent=root)
        root.destroy()
        raise SystemExit(1) from exc
    root.mainloop()


def math_is_finite(value: float) -> bool:
    # Kept tiny to avoid importing all of math for a single validation.
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
