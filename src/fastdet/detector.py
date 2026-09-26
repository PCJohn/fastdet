"""The user-facing detector: fit, prune, predict, export, load.

A :class:`Detector` owns a :class:`~fastdet.config.Config`, the front-end
(:class:`~fastdet.features.FeatureExtractor`), and a fitted booster.  Scoring at
inference goes through the exported single-file artifact's Python runtime, so
``predict_proba`` and the C++ runtime consume identical bytes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .artifact import ModelArtifact
from .config import Config, ModelConfig
from .dataset import build_split, gather_training_matrix, sample_training_cells
from .exporter import build_blob
from .features import (
    GRID,
    FeatureCache,
    FeatureExtractor,
    FloatArray,
    LevelBanks,
    feature_level_bits,
)
from .images import read_image
from .metrics import pooled_pr_auc, score_validation_per_image
from .native import NativeScorer, load_scorer
from .runtime import ImysModel, parse_blob
from .training import (
    calibrate_exit_stages,
    fit_booster,
    load_ranking,
    select_columns,
    stage_tree_counts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from catboost import CatBoostClassifier

__all__ = ["Detector"]

Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]


_LARGE_MATRIX_GIB = 8.0  # above this the fit prints how to shrink the training matrix


class Detector:
    """Per-cell text detector over the frozen multi-level feature front-end.

    Examples:
        >>> det = Detector()                              # frozen defaults
        >>> det = Detector(depth=7, n_trees=2400)         # tweak the booster
        >>> det = Detector(config=Config.from_yaml("cfg.yaml"))
        >>> det = Detector.from_config_file("cfg.yaml")
        >>> det.fit("images/", "masks/")                  # doctest: +SKIP
        >>> det.export("model.fdt")                       # doctest: +SKIP
        >>> det = Detector.load("model.fdt")              # doctest: +SKIP
        >>> prob = det.predict_proba("page.png")          # doctest: +SKIP
    """

    def __init__(
        self,
        config: Config | ModelConfig | None = None,
        threads: int | None = None,
        **model_overrides: Any,
    ) -> None:
        """Build a detector, optionally overriding individual model fields.

        ``config`` may be a full :class:`Config` or just a :class:`ModelConfig`;
        ``**model_overrides`` (e.g. ``depth=7``) are applied on top of it.  ``threads``
        is imfeat's thread count for the front-end (``None`` = the machine's cores, at
        most 4); it changes speed only, the features are bit-identical at any count.
        """
        resolved = config or Config()
        # Accept a plain ModelConfig too, so Detector(ModelConfig(depth=7)) works.
        if isinstance(resolved, ModelConfig):
            resolved = Config(model=resolved)
        if model_overrides:
            model = dataclasses.replace(resolved.model, **model_overrides)
            resolved = dataclasses.replace(resolved, model=model)
        self.config = resolved
        self.threads = threads  # imfeat threads for the front-end (None = default_threads())

        self.booster: CatBoostClassifier | None = None  # when training in-process
        self.runtime: ImysModel | None = None  # scoring engine (fitted or loaded)
        self.exit_stages: list[tuple[int, float]] = []  # calibrated at fit time, stored in the blob
        self.native: NativeScorer | None = None  # the in-process C++ scorer, once fitted or loaded
        self.train_cache: FeatureCache | None = None  # the last fit's training features (reusable)
        self.col_keep: Int64Array | None = None  # kept columns into the full design
        self.base_names: list[str] | None = None  # full (unpruned) column names
        self.feature_names: list[str] | None = None  # kept column names
        self.feature_ranks: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = {}
        self.val_ids: list[str] = []
        self._extractor: FeatureExtractor | None = None

    # -- construction paths -------------------------------------------------
    @classmethod
    def from_config_file(cls, path: str | Path) -> Detector:
        """Build a detector from a JSON or YAML config file."""
        return cls(config=Config.from_file(path))

    @classmethod
    def load(cls, path: str | Path, threads: int | None = None) -> Detector:
        """Load a detector from a single-file artifact written by :meth:`export`.

        ``threads`` is imfeat's thread count for the front-end (see :meth:`__init__`).
        """
        artifact = ModelArtifact.load(path)
        det = cls(config=artifact.config, threads=threads)
        det.runtime = parse_blob(artifact.blob)
        det.feature_names = list(artifact.feature_names)
        det.base_names = det.extractor.base_names
        det.col_keep = det._columns_for(det.feature_names)
        det.metrics = dict(artifact.metadata)
        det.exit_stages = list(det.runtime.exit_stages)
        det.native = load_scorer(artifact.blob)
        return det

    @property
    def extractor(self) -> FeatureExtractor:
        """The front-end extractor for this detector's config (built once)."""
        if self._extractor is None:
            self._extractor = FeatureExtractor(self.config.train, self.threads)
        return self._extractor

    # -- feature selection --------------------------------------------------
    def prune(self, top_k: int | None = None) -> int:
        """Select the top-``top_k`` ranked columns from the front-end's design.

        Returns the number of columns kept.  ``top_k <= 0`` keeps everything.

        Column selection defines the design matrix the booster is trained on, so
        it must happen *before* fitting.  Calling this on a fitted detector would
        leave the kept columns and the blob's feature indices describing
        different matrices, so it discards the fitted booster and runtime
        instead: refit before scoring again.
        """
        top_k = self.config.train.top_k_features if top_k is None else top_k
        if self.base_names is None:
            self.base_names = self.extractor.base_names
        if self.feature_ranks is None:
            self.feature_ranks = load_ranking(self.config.train.feature_ranks)
        self.col_keep = select_columns(self.base_names, self.feature_ranks, top_k)
        self.config.train.top_k_features = int(top_k)
        if self.booster is not None or self.runtime is not None:
            self.booster = None
            self.runtime = None
            self.feature_names = None
            self.metrics.pop("pr_auc", None)
        return len(self.col_keep)

    def _columns_for(self, names: list[str]) -> Int64Array:
        """Indices of ``names`` in the extractor's canonical column order."""
        index = {n: i for i, n in enumerate(self.extractor.base_names)}
        missing = [n for n in names if n not in index]
        if missing:
            msg = (
                f"artifact feature names do not match this config's front-end "
                f"({len(missing)} missing, e.g. {missing[:3]})"
            )
            raise ValueError(msg)
        cols = np.array([index[n] for n in names], dtype=np.int64)
        return cols[np.argsort(cols)]

    # -- training -----------------------------------------------------------
    def fit(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        *,
        evaluate: bool = True,
        train_cache: FeatureCache | None = None,
    ) -> Detector:
        """Train on an images/masks directory pair.

        Builds the group-aware split, caches features, prunes columns, fits the
        symmetric booster, and (by default) records pooled validation PR-AUC.
        ``train_cache`` reuses the training features of an earlier fit with the same
        front-end settings and split (what ``fastdet-tune`` does between runs).
        """
        images_dir, masks_dir = str(images_dir), str(masks_dir)
        cfg = self.config
        _pairs, train_pairs, val_pairs = build_split(images_dir, masks_dir, cfg.train)

        print(f"[fastdet] train={len(train_pairs)} val={len(val_pairs)} images")
        if train_cache is not None and (
            train_cache.cfg != cfg.train or train_cache.img_paths != [p[0] for p in train_pairs]
        ):
            msg = "train_cache was built with different front-end settings or a different split"
            raise ValueError(msg)
        if train_cache is None:
            train_cache = FeatureCache(train_pairs, cfg.train, "train")
        self.train_cache = train_cache
        self.base_names = train_cache.base_names
        self.booster = None  # prune() requires an unfitted detector
        self.runtime = None
        self.prune()

        img_ids, local_ids, labels = sample_training_cells(
            train_cache, cfg.train, seed=cfg.model.random_seed
        )
        design = gather_training_matrix(train_cache, img_ids, local_ids, self.col_keep)
        gib = design.nbytes / 2**30
        print(
            f"[fastdet] training matrix: {design.shape[0]:,} cells x {design.shape[1]} features, "
            f"{int(labels.sum()):,} positives, {gib:.1f} GiB as float32 (CatBoost bins every feature into "
            f"border_count={cfg.model.border_count} intervals for training and this copy is then freed; "
            f"leaf values are stored at {cfg.model.leaf_bits} bits)"
        )
        if gib > _LARGE_MATRIX_GIB:
            print(
                f"[fastdet] {gib:.0f} GiB is a large training matrix: neg_pos_ratio or max_train_cells "
                "(TrainConfig) keep it in RAM, task_type='GPU' (ModelConfig) fits it faster"
            )
        kept = self.col_keep if self.col_keep is not None else np.arange(len(self.base_names))
        sides = [feature_level_bits(self.base_names[int(i)])[0] for i in kept]
        self.booster = fit_booster(design, labels, cfg.model, feature_sides=sides)
        del design
        self.exit_stages = []
        runtime = self._refresh_runtime()
        if cfg.model.use_exit:
            self.exit_stages = self._calibrate_exit(runtime, train_cache)
            runtime = self._refresh_runtime()  # the blob now carries the stages
        if evaluate:
            val_cache = FeatureCache(val_pairs, cfg.train, "val")

            # The booster's float leaves are not what ships: score with the exported
            # runtime, early exit included, so the reported PR-AUC is the model's.
            def shipped(design: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
                return runtime.predict_proba(design, use_exit=cfg.model.use_exit)

            scores, targets, _grids = score_validation_per_image(
                self.booster, val_cache, col_keep=self.col_keep, scorer=shipped
            )
            pr_auc = pooled_pr_auc(np.concatenate(scores), np.concatenate(targets).astype(bool))
            self.metrics["pr_auc"] = pr_auc
            self.val_ids = [Path(pair[0]).name for pair in val_pairs]
            print(f"[fastdet] validation pooled PR-AUC = {pr_auc:.4f}")
        return self

    def _calibrate_exit(self, runtime: ImysModel, cache: FeatureCache) -> list[tuple[int, float]]:
        """Stage thresholds from the training images (see :func:`calibrate_exit_stages`)."""
        cfg = self.config.model
        stages = stage_tree_counts(cfg)
        if not stages:
            return []
        full = np.arange(GRID * GRID)
        partials, finals = [], []
        for i in range(cache.n):
            bins = runtime.bins(cache.gather(i, full, col_keep=self.col_keep))
            partial = runtime.partial_scores(bins, [*stages, runtime.n_trees])
            partials.append(partial[:-1])
            finals.append(partial[-1])
        calibrated = calibrate_exit_stages(
            np.concatenate(partials, axis=1),
            np.concatenate(finals),
            stages,
            keep_prob=cfg.exit_keep_prob,
            margin=cfg.exit_margin,
        )
        print(
            "[fastdet] early-exit stages: "
            + ", ".join(f"{trees}:{theta:.2f}" for trees, theta in calibrated)
            + f" (keep p>={cfg.exit_keep_prob}, margin {cfg.exit_margin})"
        )
        return calibrated

    def _refresh_runtime(self) -> ImysModel:
        """(Re)parse the in-memory export of the fitted booster and return it."""
        blob, info = self._build_blob()
        runtime = parse_blob(blob)
        self.runtime = runtime
        self.native = load_scorer(blob)
        self.metrics.update(info)
        if self.col_keep is not None and self.base_names is not None:
            self.feature_names = [self.base_names[int(i)] for i in self.col_keep]
        return runtime

    def _build_blob(self) -> tuple[bytes, dict[str, Any]]:
        if self.booster is None or self.col_keep is None or self.base_names is None:
            msg = "no fitted booster; call fit() first"
            raise RuntimeError(msg)
        # The blob addresses features positionally in gathered order, and
        # gather() emits columns in ascending index order, so col_keep must be
        # strictly ascending or every split would read the wrong column.
        if np.any(np.diff(self.col_keep) <= 0):
            msg = "col_keep must be strictly ascending; the blob indexes columns positionally"
            raise RuntimeError(msg)
        level_shift = [feature_level_bits(self.base_names[int(i)])[1] for i in self.col_keep]
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "model.json"
            self.booster.save_model(str(json_path), format="json")
            with json_path.open(encoding="utf-8") as fh:
                model_json = json.load(fh)
        model_cfg = self.config.model
        return build_blob(
            model_json,
            level_shift,
            leaf_bits=model_cfg.leaf_bits,
            leaf_chunk=model_cfg.leaf_chunk,
            coarse_trees=(
                model_cfg.coarse_trees if model_cfg.coarse_trees < model_cfg.n_trees else 0
            ),
            exit_stages=self.exit_stages,
        )

    # -- inference ----------------------------------------------------------
    def design_matrix(self, image: str | Path | UInt8Array) -> NDArray[np.floating[Any]]:
        """Kept-column features for every cell: a ``(GRID * GRID, n_features)`` matrix.

        This is exactly the matrix :meth:`predict_proba` scores (the Python
        runtime is the reference, so it keeps the dense per-cell layout).
        """
        level_maps, broadcast_vecs = self.extractor.extract(self._decode(image))
        return self.extractor.gather(
            level_maps, broadcast_vecs, np.arange(GRID * GRID), col_keep=self.col_keep
        )

    def native_matrix(self, image: str | Path | UInt8Array) -> NDArray[np.floating[Any]]:
        """Kept-column features at native resolution: the C++ runtime's input.

        Each column contributes its level's ``side x side`` values, not one per
        cell, so no dense table is built; expanding every value over its block
        reproduces :meth:`design_matrix` exactly.
        """
        level_maps, broadcast_vecs = self.extractor.extract(self._decode(image))
        return self.extractor.native(level_maps, broadcast_vecs, col_keep=self.col_keep)

    @staticmethod
    def _decode(image: str | Path | UInt8Array) -> UInt8Array:
        """A path (read as BGR) or an already-decoded ``uint8`` array."""
        if isinstance(image, (str, os.PathLike)):
            decoded = read_image(str(image))
            if decoded is None:
                msg = f"could not read image {image!r}"
                raise ValueError(msg)
            return decoded
        return cast("UInt8Array", np.asarray(image))  # no dtype coercion, as before

    def close(self) -> None:
        """Release the front-end (imfeat's worker threads are joined) and the scorer.

        Python's refcounting does this when the detector is garbage-collected; call
        it explicitly in long-running hosts, or before interpreter shutdown on
        Windows, where joining threads during DLL unload can stall the process.
        """
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
        self.native = None

    @property
    def front_end_spec(self) -> dict[str, Any]:
        """The imfeat call this model's features come from (see :meth:`predict_from_imfeat`)."""
        return self.extractor.front_end_spec

    def predict_from_imfeat(
        self,
        result: Any,
        image_hw: tuple[int, int],
        extra_results: Sequence[Any] = (),
    ) -> NDArray[np.floating[Any]]:
        """Per-cell probabilities from an imfeat result the caller already computed.

        For hosts that run imfeat themselves (framegate): make the thumbnail and the
        ``FeatureComputer`` exactly as :attr:`front_end_spec` says, pass its result and
        the original frame's ``(height, width)``, and fastdet skips its own pass.  Same
        map as :meth:`predict_proba` on the frame, through the same scorer (C++ when
        the library is built, NumPy otherwise).
        """
        if self.runtime is None:
            msg = "detector is not fitted; call fit() or load()"
            raise RuntimeError(msg)
        level_maps, broadcast_vecs = self.extractor.compose(result, image_hw, extra_results)
        return self._predict_from_maps(level_maps, broadcast_vecs)

    def _predict_from_maps(
        self, level_maps: dict[int, LevelBanks], broadcast_vecs: dict[int, FloatArray]
    ) -> NDArray[np.floating[Any]]:
        if self.runtime is None:
            msg = "detector is not fitted; call fit() or load()"
            raise RuntimeError(msg)
        if self.native is None:
            msg = "detector is not fitted; call fit() or load()"
            raise RuntimeError(msg)
        native = self.extractor.native(level_maps, broadcast_vecs, self.col_keep)
        return self.native.score(native, use_exit=self.config.model.use_exit).reshape(GRID, GRID)

    def predict_proba(self, image: str | Path | UInt8Array) -> NDArray[np.floating[Any]]:
        """Per-cell positive probability for one image as a ``GRID x GRID`` map.

        ``image`` is a path or a BGR ``uint8`` array.  The map is the model's
        raw cell scores; resize it to the source resolution to overlay.
        """
        if self.runtime is None:
            msg = "detector is not fitted; call fit() or load()"
            raise RuntimeError(msg)
        level_maps, broadcast_vecs = self.extractor.extract(self._decode(image))
        return self._predict_from_maps(level_maps, broadcast_vecs)

    # -- persistence --------------------------------------------------------
    def export(self, path: str | Path) -> Path:
        """Write the fitted model as ONE self-contained file and return its path."""
        if self.booster is None:
            msg = "no fitted booster; call fit() first"
            raise RuntimeError(msg)
        artifact = self.build_artifact()
        written = artifact.save(path)
        print(f"[fastdet] wrote {written} ({written.stat().st_size:,} bytes)")
        return written

    def build_artifact(self) -> ModelArtifact:
        """Assemble the single-file artifact for the fitted model."""
        if self.booster is None or self.base_names is None or self.col_keep is None:
            msg = "no fitted booster; call fit() first"
            raise RuntimeError(msg)
        blob, info = self._build_blob()
        metadata = dict(self.metrics)
        metadata.update(info)
        metadata["val_ids"] = list(self.val_ids)
        metadata["exit_stages"] = [[int(t), float(theta)] for t, theta in self.exit_stages]
        return ModelArtifact(
            config=self.config,
            feature_names=[self.base_names[int(i)] for i in self.col_keep],
            blob=blob,
            metadata=metadata,
        )
