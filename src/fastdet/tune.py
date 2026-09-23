"""Hyperparameter sweep with a Markdown report and ROC / PR charts (``fastdet-tune``).

Point it at an image folder and a mask folder, name the knobs to sweep as
comma-separated lists, and it fits one detector per combination, scores the
held-out split exactly as the shipped runtime does (quantised leaves, coarse tier,
early exit), and writes:

* ``report.md`` -- one row per combination (only the swept knobs as columns) with
  PR-AUC, ROC-AUC, best F1 and its threshold, precision / recall / IoU at that
  threshold, fit time, model size and (when the C++ library is built) latency; the
  framegate text heuristic as a baseline row; the best combination's full config.
* ``pr.png`` / ``roc.png`` -- curves of the baseline and the best runs.
* ``results.json`` -- every run's config and metrics; re-running resumes from it.

Every field of :class:`~fastdet.config.ModelConfig` and
:class:`~fastdet.config.TrainConfig` is a knob: ``--n-trees 1200,2400``,
``--leaf-bits 4,8``, ``--thumb 256,512`` ... A knob left out keeps its default,
which is the tuned production value, so the sweep only grows with what you name.
Run ``fastdet-tune --help`` for the list and each knob's current default.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import Config, ModelConfig, TrainConfig
from .dataset import build_split
from .detector import Detector
from .features import _MEAN_IDX, GRID, RAW_FEATURE_NAMES, RAW_PER_CHANNEL, SPACE_INFO, FeatureCache
from .metrics import pooled_pr_auc, score_validation_per_image

__all__ = ["framegate_text_heuristic", "main", "run_sweep"]

Float64Array = NDArray[np.float64]

# ---------------------------------------------------------------------------------------------
# The framegate text heuristic (copied from framegate.signals.text so nothing is imported): a
# per-cell text likelihood from the finest grid's V/S moments and structure-tensor coherence.
_FG_ACHROMATIC_W = 0.5  # down-weight saturated cells
_FG_COARSE_K = 3  # neighbourhood of the coarse between-cell energy
_FG_LINE_K = 5  # horizontal smoothing (text-line coherence)
_FG_SKEW_W = 0.8  # bimodality gate weight
_FG_SKEW_REF = 1.2  # |standardised skew| at which the gate saturates
_FG_COH_W = 0.8  # isotropy gate weight
_M3_IDX = RAW_FEATURE_NAMES.index("m3")
_VAR_IDX = RAW_FEATURE_NAMES.index("var")
_COH_IDX = RAW_FEATURE_NAMES.index("coherence")


def _box(x: NDArray[np.float32], kx: int, ky: int) -> NDArray[np.float32]:
    return np.asarray(
        cv2.boxFilter(x, -1, (kx, ky), normalize=True, borderType=cv2.BORDER_REPLICATE),
        dtype=np.float32,
    )


def framegate_text_heuristic(
    v_mean: NDArray[np.float32],
    v_var: NDArray[np.float32],
    v_m3: NDArray[np.float32],
    s_mean: NDArray[np.float32],
    coherence: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Framegate's ``signals.text`` on per-cell maps (all ``(grid, grid)``, 0..255 scale).

    High-frequency within-cell contrast not explained by coarse between-cell
    variation, down-weighted by saturation, gated by per-cell skew (text is bimodal)
    and by gradient isotropy (text mixes stroke orientations), smoothed along lines.
    """
    var = np.maximum(v_var, 0.0).astype(np.float32)
    fine = np.sqrt(var).astype(np.float32)
    mean = v_mean.astype(np.float32)
    coarse = np.sqrt(
        np.maximum(
            _box(mean * mean, _FG_COARSE_K, _FG_COARSE_K)
            - _box(mean, _FG_COARSE_K, _FG_COARSE_K) ** 2,
            0.0,
        )
    )
    skew = np.abs(v_m3) / (var * fine + 1e-6)
    bimodal = 1.0 - _FG_SKEW_W * (1.0 - np.minimum(skew / _FG_SKEW_REF, 1.0))
    iso = np.clip(1.0 - _FG_COH_W * coherence, 0.0, 1.0)
    score = (
        np.maximum(fine - coarse, 0.0) * (1.0 - _FG_ACHROMATIC_W * s_mean / 255.0) * bimodal * iso
    )
    return _box(score.astype(np.float32), _FG_LINE_K, 1)


def heuristic_scores(det: Detector, cache: FeatureCache) -> list[Float64Array]:
    """The heuristic's score for every cell of every image in ``cache`` (row-major grid)."""
    space = SPACE_INFO[det.config.train.imfeat_space]
    lum, chroma = space.lum, space.chr
    out: list[Float64Array] = []
    for i in range(cache.n):
        finest = np.asarray(
            cache.level_maps_list[i][GRID][0], dtype=np.float32
        )  # (grid, grid, 3 * 54), raw bank
        score = framegate_text_heuristic(
            finest[..., lum * RAW_PER_CHANNEL + _MEAN_IDX],
            finest[..., lum * RAW_PER_CHANNEL + _VAR_IDX],
            finest[..., lum * RAW_PER_CHANNEL + _M3_IDX],
            finest[..., chroma * RAW_PER_CHANNEL + _MEAN_IDX],
            finest[..., lum * RAW_PER_CHANNEL + _COH_IDX],
        )
        out.append(score.reshape(-1).astype(np.float64))
    return out


# ---------------------------------------------------------------------------------------------
# Metrics and curves (NumPy only).


def roc_curve(
    scores: Float64Array, labels: NDArray[np.bool_]
) -> tuple[Float64Array, Float64Array, float]:
    """``(fpr, tpr, auc)`` over decreasing thresholds."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1.0)])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1.0)])
    auc = (
        float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
        if n_pos > 0 and n_neg > 0
        else float("nan")
    )
    return fpr, tpr, auc


def pr_curve(
    scores: Float64Array, labels: NDArray[np.bool_]
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """``(recall, precision, thresholds)`` over decreasing thresholds (one point per cell)."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1, dtype=np.float64)
    precision = tp / k
    recall = tp / max(float(y.sum()), 1.0)
    return recall, precision, scores[order]


def summarise(scores: Float64Array, labels: NDArray[np.bool_]) -> dict[str, float]:
    """PR-AUC, ROC-AUC and the operating point with the best F1."""
    recall, precision, thresholds = pr_curve(scores, labels)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1)) if len(f1) else 0
    tp = precision[best] * (best + 1) if len(f1) else 0.0
    n_pos = float(labels.sum())
    fp = (best + 1) - tp
    fn = n_pos - tp
    _fpr, _tpr, roc_auc = roc_curve(scores, labels)
    return {
        "pr_auc": pooled_pr_auc(scores, labels),
        "roc_auc": roc_auc,
        "f1": float(f1[best]) if len(f1) else 0.0,
        "threshold": float(thresholds[best]) if len(f1) else 0.0,
        "precision": float(precision[best]) if len(f1) else 0.0,
        "recall": float(recall[best]) if len(f1) else 0.0,
        "iou": float(tp / max(tp + fp + fn, 1e-12)),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
    }


# ---------------------------------------------------------------------------------------------
# The sweep.

_METRIC_COLUMNS = ("pr_auc", "roc_auc", "f1", "threshold", "precision", "recall", "iou")


def _knob_fields() -> dict[str, tuple[str, dataclasses.Field[Any]]]:
    """``{knob: (section, field)}`` for every ModelConfig / TrainConfig field."""
    knobs: dict[str, tuple[str, dataclasses.Field[Any]]] = {}
    for section, cls in (("model", ModelConfig), ("train", TrainConfig)):
        for f in fields(cls):
            knobs[f.name] = (section, f)
    return knobs


def _parse_value(text: str, default: Any) -> Any:
    """One knob value from its CLI text, typed like the default."""
    if isinstance(default, bool):
        return text.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    if isinstance(default, tuple):
        return tuple(float(v) if "." in v else int(v) for v in text.split("/") if v)
    if default is None:
        return (
            None
            if text.strip().lower() in {"none", ""}
            else (int(text) if text.strip().lstrip("-").isdigit() else text)
        )
    return text


def _run_key(overrides: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]


def _combinations(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = sorted(sweep)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(sweep[n] for n in names))
    ]


def _make_config(
    overrides: dict[str, Any], knobs: dict[str, tuple[str, dataclasses.Field[Any]]]
) -> Config:
    model: dict[str, Any] = {}
    train: dict[str, Any] = {}
    for name, value in overrides.items():
        section, _field = knobs[name]
        (model if section == "model" else train)[name] = value
    return Config(model=ModelConfig(**model), train=TrainConfig(**train))


def _validation_scores(
    det: Detector, cache: FeatureCache
) -> tuple[Float64Array, NDArray[np.bool_]]:
    if det.booster is None or det.runtime is None:
        msg = "detector is not fitted"
        raise RuntimeError(msg)
    runtime, use_exit = det.runtime, det.config.model.use_exit
    scores, targets, _grids = score_validation_per_image(
        det.booster,
        cache,
        col_keep=det.col_keep,
        scorer=lambda d: runtime.predict_proba(d, use_exit=use_exit),
    )
    return np.concatenate(scores), np.concatenate(targets).astype(bool)


def _latency_ms(det: Detector, image: NDArray[np.uint8], reps: int = 10) -> float | None:
    """Milliseconds per ``predict_proba`` with the C++ scorer, or ``None`` without it."""
    if det.native is None:
        return None
    for _ in range(3):
        det.predict_proba(image)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        det.predict_proba(image)
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def run_sweep(  # noqa: PLR0913 -- the whole sweep in one readable function
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    sweep: dict[str, list[Any]],
    *,
    baseline: bool = True,
    top_charts: int = 6,
    max_runs: int | None = None,
) -> Path:
    """Fit every combination in ``sweep``, evaluate it and write the report; returns ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results: list[dict[str, Any]] = (
        json.loads(results_path.read_text()) if results_path.exists() else []
    )
    done = {r["key"] for r in results}
    knobs = _knob_fields()
    combos = _combinations(sweep)
    if max_runs is not None:
        combos = combos[:max_runs]
    print(
        f"[fastdet-tune] {len(combos)} combination(s) over {sorted(sweep)}; {len(done & {_run_key(c) for c in combos})} already done"
    )

    caches: dict[str, FeatureCache] = {}
    curves: dict[str, tuple[Float64Array, NDArray[np.bool_]]] = {}
    baseline_done = False
    for i, overrides in enumerate(combos, 1):
        key = _run_key(overrides)
        if key in done:
            continue
        cfg = _make_config(overrides, knobs)
        train_key = json.dumps(dataclasses.asdict(cfg.train), sort_keys=True, default=str)
        label = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "defaults"
        print(f"[fastdet-tune] run {i}/{len(combos)}: {label}", flush=True)
        t0 = time.perf_counter()
        det = Detector(cfg).fit(images_dir, masks_dir, evaluate=False)
        fit_s = time.perf_counter() - t0
        if train_key not in caches:
            _all, _train_pairs, val_pairs = build_split(
                str(images_dir), str(masks_dir), cfg.train, phash_cache=None
            )
            caches[train_key] = FeatureCache(val_pairs, cfg.train, "val")
        cache = caches[train_key]
        scores, labels = _validation_scores(det, cache)
        metrics = summarise(scores, labels)
        model_path = out_dir / "models" / f"{key}.fdt"
        model_path.parent.mkdir(exist_ok=True)
        det.export(model_path)
        sample = np.asarray(cv2.imread(cache.img_paths[0]), dtype=np.uint8) if cache.n else None
        latency = _latency_ms(det, sample) if sample is not None else None
        results.append(
            {
                "key": key,
                "overrides": overrides,
                "config": cfg.to_dict(),
                "metrics": metrics,
                "fit_seconds": fit_s,
                "model_bytes": model_path.stat().st_size,
                "latency_ms": latency,
                "val_images": cache.n,
                "val_cells": len(labels),
            }
        )
        results_path.write_text(json.dumps(results, indent=1, default=str))
        np.save(out_dir / "models" / f"{key}.scores.npy", scores.astype(np.float32))
        np.save(out_dir / "models" / f"{key}.labels.npy", labels)
        curves[key] = (scores, labels)
        if baseline and not baseline_done:
            b_scores = np.concatenate(heuristic_scores(det, cache))
            b_metrics = summarise(b_scores, labels)
            (out_dir / "baseline.json").write_text(json.dumps({"metrics": b_metrics}, indent=1))
            np.save(out_dir / "baseline.scores.npy", b_scores.astype(np.float32))
            np.save(out_dir / "baseline.labels.npy", labels)
            baseline_done = True
        print(
            f"[fastdet-tune]   PR-AUC {metrics['pr_auc']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}  F1 {metrics['f1']:.4f}  fit {fit_s:.0f}s",
            flush=True,
        )
    return write_report(out_dir, results, sweep, top_charts=top_charts)


def _load_curve(out_dir: Path, key: str) -> tuple[Float64Array, NDArray[np.bool_]] | None:
    scores_path = out_dir / "models" / f"{key}.scores.npy"
    labels_path = out_dir / "models" / f"{key}.labels.npy"
    if not (scores_path.exists() and labels_path.exists()):
        return None
    return np.load(scores_path).astype(np.float64), np.load(labels_path).astype(bool)


def write_report(
    out_dir: Path,
    results: list[dict[str, Any]],
    sweep: dict[str, list[Any]],
    *,
    top_charts: int = 6,
) -> Path:
    """``report.md`` plus ``pr.png`` / ``roc.png`` from the stored results."""
    ranked = sorted(results, key=lambda r: -r["metrics"]["pr_auc"])
    swept = sorted(k for k, v in sweep.items() if len(v) > 1) or sorted(sweep)
    baseline_path = out_dir / "baseline.json"
    baseline = json.loads(baseline_path.read_text())["metrics"] if baseline_path.exists() else None
    lines = ["# fastdet hyperparameter sweep", ""]
    if ranked:
        lines.append(
            f"{len(ranked)} run(s); {ranked[0]['val_images']} validation images, {ranked[0]['val_cells']:,} cells, positive rate {ranked[0]['metrics']['positive_rate']:.3%}."
        )
        lines.append("")
    header = [
        "run",
        *swept,
        "PR-AUC",
        "ROC-AUC",
        "F1*",
        "thr*",
        "P*",
        "R*",
        "IoU*",
        "fit s",
        "size KB",
        "latency ms",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for rank, r in enumerate(ranked, 1):
        m = r["metrics"]
        cells = [
            f"{'**' if rank == 1 else ''}{rank}{'**' if rank == 1 else ''}",
            *(str(r["overrides"].get(k, "default")) for k in swept),
        ]
        cells += [
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['threshold']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['iou']:.3f}",
            f"{r['fit_seconds']:.0f}",
            f"{r['model_bytes'] / 1024:.0f}",
            "-" if r["latency_ms"] is None else f"{r['latency_ms']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if baseline is not None:
        cells = [
            "baseline: framegate heuristic",
            *(["-"] * len(swept)),
            f"{baseline['pr_auc']:.4f}",
            f"{baseline['roc_auc']:.4f}",
            f"{baseline['f1']:.4f}",
            f"{baseline['threshold']:.3f}",
            f"{baseline['precision']:.3f}",
            f"{baseline['recall']:.3f}",
            f"{baseline['iou']:.3f}",
            "-",
            "-",
            "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "`*` at the threshold with the best F1. PR-AUC is the primary metric (pooled over all validation cells).",
        "",
    ]
    if ranked:
        lines += [
            "## Best configuration",
            "",
            "```json",
            json.dumps(ranked[0]["config"], indent=1),
            "```",
            "",
        ]
    charts = _write_charts(out_dir, ranked[:top_charts], swept)
    if charts:
        lines += ["## Charts", ""] + [f"![{name}]({name})" for name in charts] + [""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    print(f"[fastdet-tune] wrote {report}")
    return report


def _write_charts(out_dir: Path, ranked: list[dict[str, Any]], swept: list[str]) -> list[str]:
    try:
        import matplotlib as mpl  # noqa: PLC0415 -- optional dependency

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[fastdet-tune] matplotlib not installed: no charts (pip install matplotlib)")
        return []
    entries: list[tuple[str, Float64Array, NDArray[np.bool_]]] = []
    for r in ranked:
        curve = _load_curve(out_dir, r["key"])
        if curve is not None:
            label = (
                ", ".join(f"{k}={r['overrides'].get(k, 'default')}" for k in swept) or "defaults"
            )
            entries.append((f"{label} (PR-AUC {r['metrics']['pr_auc']:.3f})", *curve))
    b_scores, b_labels = out_dir / "baseline.scores.npy", out_dir / "baseline.labels.npy"
    if b_scores.exists() and b_labels.exists():
        entries.append(
            (
                "baseline: framegate heuristic",
                np.load(b_scores).astype(np.float64),
                np.load(b_labels).astype(bool),
            )
        )
    if not entries:
        return []
    written = []
    for name, kind in (("pr.png", "pr"), ("roc.png", "roc")):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for label, scores, labels in entries:
            style = "--" if label.startswith("baseline") else "-"
            if kind == "pr":
                recall, precision, _ = pr_curve(scores, labels)
                ax.plot(recall, precision, style, label=label)
            else:
                fpr, tpr, auc = roc_curve(scores, labels)
                ax.plot(fpr, tpr, style, label=f"{label} AUC {auc:.3f}")
        if kind == "pr":
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title("Precision-recall (pooled validation cells)")
        else:
            ax.plot([0, 1], [0, 1], ":", color="grey")
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_title("ROC (pooled validation cells)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        written.append(name)
    return written


# ---------------------------------------------------------------------------------------------
# CLI.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastdet-tune",
        description="Sweep fastdet hyperparameters on a dataset and write a Markdown report with ROC / PR charts.",
        epilog="Each knob takes a comma-separated list; tuples use '/' (e.g. --levels 64/32/16). The sweep is the "
        "Cartesian product of every knob you name; unnamed knobs keep their default, the production value.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", required=True, type=Path, help="folder of images")
    parser.add_argument("--masks", required=True, type=Path, help="folder of masks (same stems)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tune_report"),
        help="report folder (resumes from its results.json)",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the framegate heuristic baseline"
    )
    parser.add_argument(
        "--top-charts", type=int, default=6, help="how many of the best runs to draw in the charts"
    )
    parser.add_argument(
        "--max-runs", type=int, default=None, help="stop after this many combinations (smoke tests)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild report.md and the charts from results.json",
    )
    for name, (section, f) in _knob_fields().items():
        default = f.default if f.default is not dataclasses.MISSING else None
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=f"knob_{name}",
            default=None,
            metavar="V[,V...]",
            help=f"[{section}] default {default!r}",
        )
    return parser


def sweep_from_args(args: argparse.Namespace) -> dict[str, list[Any]]:
    """``{knob: [values]}`` from the parsed CLI (only the knobs that were given)."""
    knobs = _knob_fields()
    sweep: dict[str, list[Any]] = {}
    for name, (_section, f) in knobs.items():
        text = getattr(args, f"knob_{name}", None)
        if text is None:
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        values = [_parse_value(v, default) for v in text.split(",")]
        sweep[name] = list(dict.fromkeys(values))  # de-duplicate, keep order
    return sweep


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``fastdet-tune``."""
    args = build_parser().parse_args(argv)
    sweep = sweep_from_args(args)
    if args.report_only:
        results_path = args.out / "results.json"
        if not results_path.exists():
            print(f"no results.json in {args.out}", file=sys.stderr)
            return 2
        write_report(
            args.out, json.loads(results_path.read_text()), sweep, top_charts=args.top_charts
        )
        return 0
    run_sweep(
        args.images,
        args.masks,
        args.out,
        sweep,
        baseline=not args.no_baseline,
        top_charts=args.top_charts,
        max_runs=args.max_runs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
