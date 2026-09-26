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
Frames are processed as they arrive; nothing is buffered ahead.

Keys: ``q``/``Esc`` quit, ``space`` pause, ``s`` save the composite to the working
directory.  ``--headless --output out.png`` renders without a window (tests, servers).
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .detector import Detector

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

__all__ = ["main", "render_composite", "score_frame"]

PANEL_WIDTH = 440
MIN_HEIGHT = 420  # the latency panel needs room for its plot; small frames are letterboxed
HISTORY = 240  # frames of latency history kept for the plot
_MIN_POINTS = 2  # a line needs two points
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = (235, 235, 235)
_MUTED = (140, 140, 140)
_FEATURE_COLOUR = (80, 200, 255)  # BGR: amber
_MODEL_COLOUR = (120, 220, 120)  # BGR: green


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
    t0 = time.perf_counter()
    level_maps, broadcast_vecs = det.extractor.extract(frame)
    if det.native is not None:
        native = det.extractor.native(level_maps, broadcast_vecs, det.col_keep)
        t1 = time.perf_counter()
        probs = det.native.score(native, use_exit=det.config.model.use_exit)
    else:
        from .features import GRID  # noqa: PLC0415

        design = det.extractor.gather(
            level_maps, broadcast_vecs, np.arange(GRID * GRID), col_keep=det.col_keep
        )
        t1 = time.perf_counter()
        probs = det.runtime.predict_proba(design, use_exit=det.config.model.use_exit).astype(
            np.float32
        )
    t2 = time.perf_counter()
    side = round(float(np.sqrt(probs.size)))
    return probs.reshape(side, side).astype(np.float32), 1e3 * (t1 - t0), 1e3 * (t2 - t1)


def overlay(
    frame: NDArray[np.uint8], probs: NDArray[np.float32], alpha: float = 0.6
) -> NDArray[np.uint8]:
    """The frame with the probability map blended over it (opacity follows the probability)."""
    h, w = frame.shape[:2]
    heat = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight = (alpha * heat)[..., None]
    blended = frame.astype(np.float32) * (1.0 - weight) + colour.astype(np.float32) * weight
    return np.asarray(np.clip(blended, 0, 255), dtype=np.uint8)


def _plot(
    panel: NDArray[np.uint8],
    top: int,
    bottom: int,
    series: list[tuple[str, tuple[int, int, int], collections.deque[float]]],
) -> None:
    """Line plots of the latency series into ``panel[top:bottom]``, shared y axis."""
    left, right = 44, panel.shape[1] - 12
    values = [v for _, _, hist in series for v in hist]
    y_max = max(1.0, float(np.percentile(values, 98)) * 1.15) if values else 1.0
    cv2.rectangle(panel, (left, top), (right, bottom), (60, 60, 60), 1)
    for k in range(1, 4):  # grid lines with labels
        y = bottom - int((bottom - top) * k / 4)
        cv2.line(panel, (left, y), (right, y), (50, 50, 50), 1)
        cv2.putText(panel, f"{y_max * k / 4:.1f}", (2, y + 4), _FONT, 0.38, _MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, "ms", (2, top + 10), _FONT, 0.38, _MUTED, 1, cv2.LINE_AA)
    for _label, colour, hist in series:
        if len(hist) < _MIN_POINTS:
            continue
        xs = np.linspace(left, right, HISTORY)[-len(hist) :]
        ys = bottom - (np.clip(np.asarray(hist, dtype=np.float64), 0, y_max) / y_max) * (
            bottom - top
        )
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], isClosed=False, color=colour, thickness=2, lineType=cv2.LINE_AA)


def render_composite(  # noqa: PLR0913 -- one panel, all of its inputs
    frame: NDArray[np.uint8],
    probs: NDArray[np.float32],
    feature_hist: collections.deque[float],
    model_hist: collections.deque[float],
    *,
    target: str,
    live: bool,
    fps: float | None = None,
) -> NDArray[np.uint8]:
    """Overlay panel on the left, latency panel on the right, same height."""
    view = overlay(frame, probs)
    if view.shape[0] < MIN_HEIGHT:  # letterbox small frames so the panel keeps its layout
        pad = MIN_HEIGHT - view.shape[0]
        view = np.asarray(
            cv2.copyMakeBorder(
                view, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
            ),
            dtype=np.uint8,
        )
    h = view.shape[0]
    panel = np.full((h, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    y = 28
    cv2.putText(panel, "fastdet latency", (12, y), _FONT, 0.6, _INK, 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, f"scorer: {target}", (12, y), _FONT, 0.42, _MUTED, 1, cv2.LINE_AA)
    y += 30
    rows = [
        ("feature extraction", _FEATURE_COLOUR, feature_hist),
        ("model (bin + trees)", _MODEL_COLOUR, model_hist),
    ]
    for label, colour, hist in rows:
        last = hist[-1] if hist else 0.0
        med = float(np.median(hist)) if hist else 0.0
        cv2.putText(panel, label, (12, y), _FONT, 0.48, colour, 1, cv2.LINE_AA)
        text = f"{last:6.2f} ms" + (f"   median {med:6.2f}" if live and len(hist) > 1 else "")
        cv2.putText(panel, text, (12, y + 20), _FONT, 0.48, _INK, 1, cv2.LINE_AA)
        y += 48
    total = (feature_hist[-1] if feature_hist else 0.0) + (model_hist[-1] if model_hist else 0.0)
    cv2.putText(
        panel,
        f"total {total:6.2f} ms" + (f"   {fps:5.1f} fps" if fps else ""),
        (12, y),
        _FONT,
        0.5,
        _INK,
        1,
        cv2.LINE_AA,
    )
    y += 20
    if live:
        _plot(panel, y + 10, h - 30, rows)
        cv2.putText(
            panel, f"last {HISTORY} frames", (12, h - 10), _FONT, 0.38, _MUTED, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            panel, "single image: one measurement", (12, y + 20), _FONT, 0.4, _MUTED, 1, cv2.LINE_AA
        )
    return np.asarray(cv2.hconcat([view, panel]), dtype=np.uint8)


def _frames(source: str) -> tuple[Iterator[NDArray[np.uint8]], bool]:
    """``(frames, live)``: one frame for an image, a stream for a video or a webcam index."""
    path = Path(source)
    if path.is_file() and path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }:
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
    parser.add_argument("--output", type=Path, default=None, help="write the (last) composite here")
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
    target = (
        det.native.target
        if det.native is not None
        else "NumPy runtime (build cpp/ for the C++ scorer)"
    )
    frames, live = _frames(args.source)
    feature_hist: collections.deque[float] = collections.deque(maxlen=HISTORY)
    model_hist: collections.deque[float] = collections.deque(maxlen=HISTORY)
    composite = None
    paused = False
    last_tick = time.perf_counter()
    fps: float | None = None
    for n, frame in enumerate(frames, 1):
        probs, feature_ms, model_ms = score_frame(det, frame)
        feature_hist.append(feature_ms)
        model_hist.append(model_ms)
        now = time.perf_counter()
        if live:
            fps = (
                0.9 * fps + 0.1 / max(now - last_tick, 1e-6)
                if fps
                else 1.0 / max(now - last_tick, 1e-6)
            )
        last_tick = now
        composite = render_composite(
            _fit_width(frame, args.display_width),
            probs,
            feature_hist,
            model_hist,
            target=target,
            live=live,
            fps=fps,
        )
        if not args.headless:
            cv2.imshow("fastdet", composite)
            key = cv2.waitKey(0 if not live or paused else 1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                cv2.imwrite(f"fastdet_{n:05d}.png", composite)
        if args.max_frames is not None and n >= args.max_frames:
            break
    if composite is not None and args.output is not None:
        cv2.imwrite(str(args.output), composite)
        print(f"[fastdet-demo] wrote {args.output}")
    if not args.headless:
        cv2.destroyAllWindows()
    if feature_hist:
        print(
            f"[fastdet-demo] {len(feature_hist)} frame(s): feature extraction median {np.median(feature_hist):.2f} ms,"
            f" model median {np.median(model_hist):.2f} ms ({target})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
