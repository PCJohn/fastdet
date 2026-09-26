"""Live demo: probability overlay and latency panel for an image, a video or a webcam.

::

    fastdet-demo --model model.fdt --source photo.png
    fastdet-demo --model model.fdt --source clip.mp4
    fastdet-demo --model model.fdt --source 0              # webcam index

Left panel: the frame with the model's cell probabilities blended over it (JET
colormap, opacity following the probability).  Right panel: latency.  For an
image, the two numbers; for a video or webcam, a live time series of (1) feature
extraction (resize, colour conversion, imfeat, and the context banks and packing
into the scorer's layout) and (2) the model (binning, coarse tier, fine trees, sigmoid).
Frames are scored as they arrive; nothing is buffered ahead.

The window is matplotlib's, so it works with ``opencv-python-headless`` (OpenCV is
used only for decoding, resizing and colouring).  Keys: ``q``/``Esc`` quit,
``space`` pause, ``s`` save the figure to the working directory.  ``--headless
--output out.png`` renders without a window (tests, servers).
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .detector import Detector
from .features import GRID

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

__all__ = ["LatencyView", "main", "overlay", "score_frame"]

HISTORY = 240  # frames of latency history kept for the plot
_FEATURE_COLOUR = "#ffb347"  # amber
_MODEL_COLOUR = "#78dc78"  # green
_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def score_frame(
    det: Detector, frame: NDArray[np.uint8]
) -> tuple[NDArray[np.float32], float, float]:
    """``(probability map, feature ms, model ms)`` for one BGR frame.

    Feature extraction is the front-end (resize, colour conversion, imfeat, context
    banks, packing); the model is the scorer on that input.  Uses the C++ scorer when
    its library is built, the NumPy runtime otherwise (tens of ms, reported as such).
    """
    if det.runtime is None:
        msg = "detector is not fitted; call fit() or load()"
        raise RuntimeError(msg)
    use_exit = det.config.model.use_exit
    t0 = time.perf_counter()
    level_maps, broadcast_vecs = det.extractor.extract(frame)
    if det.native is not None:
        native = det.extractor.native(level_maps, broadcast_vecs, det.col_keep)
        t1 = time.perf_counter()
        probs = det.native.score(native, use_exit=use_exit)
    else:
        design = det.extractor.gather(
            level_maps, broadcast_vecs, np.arange(GRID * GRID), col_keep=det.col_keep
        )
        t1 = time.perf_counter()
        probs = det.runtime.predict_proba(design, use_exit=use_exit).astype(np.float32)
    t2 = time.perf_counter()
    return probs.reshape(GRID, GRID).astype(np.float32), 1e3 * (t1 - t0), 1e3 * (t2 - t1)


def overlay(
    frame: NDArray[np.uint8], probs: NDArray[np.float32], alpha: float = 0.6
) -> NDArray[np.uint8]:
    """The frame (BGR in) with the probability map blended over it, as RGB for matplotlib."""
    h, w = frame.shape[:2]
    heat = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight = (alpha * heat)[..., None]
    blended = frame.astype(np.float32) * (1.0 - weight) + colour.astype(np.float32) * weight
    rgb = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return np.asarray(rgb, dtype=np.uint8)


class LatencyView:
    """The two-panel matplotlib figure, updated in place frame after frame."""

    def __init__(self, *, target: str, live: bool, headless: bool) -> None:
        """Build the figure; ``headless`` selects the Agg backend (no window)."""
        import matplotlib as mpl  # noqa: PLC0415 -- optional dependency, backend chosen here

        if headless:
            mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        self.plt = plt
        self.live = live
        self.quit = False
        self.paused = False
        self.save_requested = False
        self.fig = plt.figure(figsize=(14, 6.5), facecolor="#181818", layout="constrained")
        grid = self.fig.add_gridspec(
            2, 2, width_ratios=[3, 1.6], height_ratios=[1, 2.6], hspace=0.05
        )
        self.ax_img = self.fig.add_subplot(grid[:, 0])
        self.ax_text = self.fig.add_subplot(grid[0, 1])
        self.ax_lat = self.fig.add_subplot(grid[1, 1])
        self.ax_text.set_axis_off()
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("fastdet")
        self.ax_img.set_axis_off()
        self.image_artist: Any = None
        ax = self.ax_lat
        ax.set_facecolor("#181818")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        self.ax_text.set_title(
            f"fastdet latency   [{target}]", color="#eeeeee", fontsize=11, loc="left"
        )
        ax.set_ylabel("ms", color="#bbbbbb", fontsize=8)
        (self.feature_line,) = ax.plot(
            [], [], color=_FEATURE_COLOUR, lw=1.8, label="feature extraction"
        )
        (self.model_line,) = ax.plot(
            [], [], color=_MODEL_COLOUR, lw=1.8, label="model (bin + trees)"
        )
        ax.grid(alpha=0.25)
        ax.set_xlim(0, HISTORY)
        ax.set_xlabel(f"last {HISTORY} frames" if live else "", color="#bbbbbb", fontsize=8)
        ax.legend(
            loc="upper left",
            fontsize=8,
            facecolor="#242424",
            edgecolor="#444444",
            labelcolor="#eeeeee",
        )
        self.text = self.ax_text.text(
            0.0,
            0.5,
            "",
            transform=self.ax_text.transAxes,
            color="#eeeeee",
            fontsize=10,
            family="monospace",
            va="center",
        )
        if not live:
            ax.set_xticks([])
            ax.set_yticks([])
        if not headless:
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig.canvas.mpl_connect("close_event", lambda _event: setattr(self, "quit", True))
            plt.ion()
            plt.show(block=False)

    def _on_key(self, event: Any) -> None:
        if event.key in {"q", "escape"}:
            self.quit = True
        elif event.key == " ":
            self.paused = not self.paused
        elif event.key == "s":
            self.save_requested = True

    def update(
        self,
        rgb: NDArray[np.uint8],
        feature_hist: collections.deque[float],
        model_hist: collections.deque[float],
        *,
        fps: float | None,
        frame_index: int,
    ) -> None:
        """Draw one frame: overlay left, numbers and series right."""
        if self.image_artist is None:
            self.image_artist = self.ax_img.imshow(rgb, interpolation="nearest")
        else:
            self.image_artist.set_data(rgb)
            if self.image_artist.get_extent()[1] != rgb.shape[1]:
                self.image_artist.set_extent((-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
        feat, model = feature_hist[-1], model_hist[-1]
        lines = [
            f"feature extraction {feat:7.2f} ms",
            f"model (bin+trees)  {model:7.2f} ms",
            f"total              {feat + model:7.2f} ms",
        ]
        if self.live:
            lines[0] += f"   median {np.median(feature_hist):6.2f}"
            lines[1] += f"   median {np.median(model_hist):6.2f}"
            if fps:
                lines[2] += f"   {fps:5.1f} fps   frame {frame_index}"
            x = np.arange(len(feature_hist))
            self.feature_line.set_data(x, np.asarray(feature_hist))
            self.model_line.set_data(x, np.asarray(model_hist))
            top = max(float(np.percentile(list(feature_hist) + list(model_hist), 98)) * 1.2, 1.0)
            self.ax_lat.set_ylim(0, top)
        else:
            lines.append("")
            lines.append("single image: one measurement")
        self.text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def pump(self, seconds: float = 0.001) -> None:
        """Let the window process events (keys, close); blocks while paused."""
        self.plt.pause(seconds)
        while self.paused and not self.quit:
            self.plt.pause(0.05)

    def save(self, path: Path) -> None:
        """Write the current figure as an image."""
        self.fig.savefig(path, dpi=110, facecolor=self.fig.get_facecolor())

    def block(self) -> None:
        """Keep a single image on screen until the window is closed or ``q`` is pressed."""
        while not self.quit and self.plt.fignum_exists(self.fig.number):
            self.plt.pause(0.05)
            if self.save_requested:
                self.save_requested = False
                self.save(Path("fastdet_image.png"))

    def close(self) -> None:
        """Close the window."""
        self.plt.close(self.fig)


def _frames(source: str) -> tuple[Iterator[NDArray[np.uint8]], bool]:
    """``(frames, live)``: one frame for an image, a stream for a video or a webcam index."""
    path = Path(source)
    if path.is_file() and path.suffix.lower() in _IMAGE_TYPES:
        image = cv2.imread(str(path))
        if image is None:
            msg = f"cannot read image {source}"
            raise SystemExit(msg)
        return iter([np.asarray(image, dtype=np.uint8)]), False
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        msg = f"cannot open video source {source}"
        raise SystemExit(msg)

    def stream() -> Iterator[NDArray[np.uint8]]:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield np.asarray(frame, dtype=np.uint8)
        capture.release()

    return stream(), True


def _fit_width(frame: NDArray[np.uint8], max_width: int) -> NDArray[np.uint8]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return np.asarray(
        cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastdet-demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path, help="an exported .fdt model")
    parser.add_argument(
        "--source", required=True, help="image file, video file, or webcam index (0, 1, ...)"
    )
    parser.add_argument(
        "--display-width", type=int, default=960, help="scale the frame panel down to this width"
    )
    parser.add_argument("--headless", action="store_true", help="no window; use with --output")
    parser.add_argument("--output", type=Path, default=None, help="write the (last) figure here")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop after this many frames (tests)"
    )
    parser.add_argument(
        "--native-lib",
        type=Path,
        default=None,
        help="the fastdet_native shared library, if not found automatically",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- the display loop
    """Entry point of ``fastdet-demo``."""
    args = build_parser().parse_args(argv)
    if args.native_lib is not None:
        import os  # noqa: PLC0415

        os.environ["FASTDET_NATIVE_LIB"] = str(args.native_lib)
    det = Detector.load(args.model)
    if det.native is None:
        print(
            "[fastdet-demo] WARNING: the C++ scorer (fastdet_native) was not found; scoring with the NumPy runtime,"
            " hundreds of times slower. Run `fastdet-native-build` once, or pass --native-lib / set FASTDET_NATIVE_LIB.",
            file=sys.stderr,
        )
    target = (
        det.native.target if det.native is not None else "NumPy runtime: run fastdet-native-build"
    )
    frames, live = _frames(args.source)
    view = LatencyView(target=target, live=live, headless=args.headless)
    feature_hist: collections.deque[float] = collections.deque(maxlen=HISTORY)
    model_hist: collections.deque[float] = collections.deque(maxlen=HISTORY)
    last_tick = time.perf_counter()
    fps: float | None = None
    n = 0
    for n, frame in enumerate(frames, 1):
        probs, feature_ms, model_ms = score_frame(det, frame)
        feature_hist.append(feature_ms)
        model_hist.append(model_ms)
        now = time.perf_counter()
        if live:
            instant = 1.0 / max(now - last_tick, 1e-6)
            fps = instant if fps is None else 0.9 * fps + 0.1 * instant
        last_tick = now
        view.update(
            overlay(_fit_width(frame, args.display_width), probs),
            feature_hist,
            model_hist,
            fps=fps,
            frame_index=n,
        )
        if not args.headless:
            view.pump()
            if view.save_requested:
                view.save_requested = False
                view.save(Path(f"fastdet_{n:05d}.png"))
            if view.quit:
                break
        if args.max_frames is not None and n >= args.max_frames:
            break
    if args.output is not None and n:
        view.save(args.output)
        print(f"[fastdet-demo] wrote {args.output}")
    if not args.headless and not live and not view.quit:
        view.block()
    view.close()
    if feature_hist:
        print(
            f"[fastdet-demo] {len(feature_hist)} frame(s): feature extraction median {np.median(feature_hist):.2f} ms,"
            f" model median {np.median(model_hist):.2f} ms ({target})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
