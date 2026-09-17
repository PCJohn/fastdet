"""Configuration objects for :mod:`fastdet`.

All tunables live in plain dataclasses so a model is fully described by a
serializable ``Config``.  The defaults are the frozen, measured configuration
(7-deep symmetric CatBoost, 2400 trees, 15 borders, lr 0.1, 64/32/16/8 feature
pyramid, 512 pruned columns); see ``README.md`` for the measured numbers.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["Config", "ExportConfig", "ModelConfig", "TrainConfig"]

# Largest bin count the 4-bit (`vpshufb`) traversal can encode.
MAX_BORDER_COUNT = 15


@dataclass
class ModelConfig:
    """Gradient-boosted symmetric (oblivious) tree model hyper-parameters."""

    depth: int = 7  # Tree depth; one split per level, 2**depth leaves per tree.
    n_trees: int = 2400  # Number of boosting iterations (symmetric trees).
    learning_rate: float = 0.1  # CatBoost learning_rate / shrinkage per tree.
    border_count: int = 15  # Per-feature split candidates; <= 15 for the 4-bit blob.
    random_seed: int = 42  # Seed for the booster; independent of the data split.
    loss_function: str = "Logloss"  # CatBoost objective; only Logloss is supported.
    grow_policy: str = "SymmetricTree"  # Must stay symmetric for the blob format.

    def __post_init__(self) -> None:
        """Validate the hyper-parameters against the export format's limits."""
        # One split per level and a single left/right decision per feature per
        # level: the whole export path assumes CatBoost's oblivious layout.
        if self.grow_policy != "SymmetricTree":
            msg = "fastdet only supports grow_policy='SymmetricTree'"
            raise ValueError(msg)
        if self.loss_function != "Logloss":
            msg = "fastdet only supports loss_function='Logloss'"
            raise ValueError(msg)
        if self.depth < 1:
            msg = "depth must be >= 1"
            raise ValueError(msg)
        if self.n_trees < 1:
            msg = "n_trees must be >= 1"
            raise ValueError(msg)
        if not 1 <= self.border_count <= MAX_BORDER_COUNT:
            msg = f"border_count must be in [1, {MAX_BORDER_COUNT}] for the 4-bit blob"
            raise ValueError(msg)


@dataclass
class TrainConfig:
    """Front-end, sampling and pruning knobs (how features become a dataset)."""

    levels: tuple[int, ...] = (64, 32, 16, 8)  # Pyramid levels, coarsest cell LxL, finest=64.
    feature_mode: str = "raw_plus_global_bard_context_ext"  # Which feature banks to build.
    thumb: int = 512  # Square resize target the feature pyramid is computed on.
    stride: int = 2  # imfeat sampling stride at the primary scale.
    extra_scales: str = "256:1"  # Extra imfeat scales, 'thumb:stride;...' ('' disables).
    resize_interp: str = "area"  # Thumbnail resize kernel: 'area' or 'nearest'.
    imfeat_space: str = "lab"  # Color space fed to imfeat: hsv|lab|luv|yuv.
    bank_gray: str = "lstar"  # Channel driving the stroke/context banks: v|y|lstar.
    bard_lags: tuple[int, ...] = (1, 2, 4)  # Bar-detector neighbor distances, in pixels.
    bard_tau: float = 8.0  # Per-pixel contrast floor (grey levels) for the bar detector.
    gt_cell_thresh: float = 0.10  # Cell coverage >= this is a positive training label.
    top_k_features: int = 512  # Columns kept after importance pruning (<= full width; 0 = all).
    feature_ranks: str | None = None  # Gain-ranking JSON; None = the bundled frozen ranking.
    split_seed: int = 42  # Seed for the group-aware train/val split.
    val_frac: float = 0.15  # Fraction of near-duplicate groups held out for validation.
    max_hamming: int = 8  # pHash Hamming distance below which images are one group.
    neg_pos_ratio: float | None = None  # Negative:positive sampling ratio; None = keep all.
    max_train_cells: int | None = None  # Cap on sampled training cells; None = no cap.

    def __post_init__(self) -> None:
        """Normalize multi-valued fields and validate the front-end knobs."""
        self.levels = tuple(sorted({int(v) for v in self.levels}, reverse=True))
        self.bard_lags = tuple(int(v) for v in self.bard_lags)
        if self.feature_mode not in _FEATURE_MODE_TAGS:
            msg = f"unknown feature_mode {self.feature_mode!r}"
            raise ValueError(msg)
        if self.imfeat_space not in _SPACE_LETTERS:
            msg = f"unknown imfeat_space {self.imfeat_space!r}"
            raise ValueError(msg)
        if self.resize_interp not in ("area", "nearest"):
            msg = "resize_interp must be 'area' or 'nearest'"
            raise ValueError(msg)
        if not 0.0 < self.val_frac < 1.0:
            msg = "val_frac must be in (0, 1)"
            raise ValueError(msg)


@dataclass
class ExportConfig:
    """Controls what the exported single-file artifact contains."""

    shuffle_tables: bool = True  # Emit 4-bit vpshufb tables (blob v3) for the fast path.
    verify: bool = True  # Decode the built artifact and compare to live scores.

    def __post_init__(self) -> None:
        """Reject export settings the shipped runtime cannot consume."""
        if not self.shuffle_tables:
            # The shipped C++ runtime scores via the shuffle tables; a v2 blob
            # has no fast traversal, so refuse to emit a model nothing can run.
            msg = "shuffle_tables must remain True for the shipped runtime"
            raise ValueError(msg)


@dataclass
class Config:
    """The full model description: architecture, training front-end, export."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved config as plain JSON-compatible data."""
        d = dataclasses.asdict(self)
        d["train"]["levels"] = list(self.train.levels)
        d["train"]["bard_lags"] = list(self.train.bard_lags)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        """Rebuild a :class:`Config` from :meth:`to_dict` output (or a subset)."""
        data = dict(data or {})
        model = ModelConfig(**(data.get("model") or {}))
        train_d = dict(data.get("train") or {})
        if "levels" in train_d:
            train_d["levels"] = tuple(train_d["levels"])
        if "bard_lags" in train_d:
            train_d["bard_lags"] = tuple(train_d["bard_lags"])
        train = TrainConfig(**train_d)
        export = ExportConfig(**(data.get("export") or {}))
        return cls(model=model, train=train, export=export)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize to a JSON string; also write it to ``path`` when given."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        """Load a config from a JSON file written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML (requires PyYAML); also write it when given."""
        import yaml  # noqa: PLC0415 -- optional dependency, imported on demand

        text: str = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config from a YAML file (requires PyYAML)."""
        import yaml  # noqa: PLC0415 -- optional dependency, imported on demand

        data: Mapping[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load a config, choosing JSON or YAML from the file extension."""
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(path)
        return cls.from_json(path)


# Imported lazily to keep this module importable without the feature stack.
_SPACE_LETTERS = {"hsv", "lab", "luv", "yuv"}
_FEATURE_MODE_TAGS = {
    "raw": ["raw"],
    "raw_plus_global": ["raw", "global"],
    "raw_plus_global_bard": ["raw", "global", "bard"],
    "raw_plus_global_bard_context_ext": ["raw", "global", "bard", "context", "ctx2"],
}


def iter_feature_mode_tags(mode: str) -> Iterable[str]:
    """Return the ordered feature-bank tags built for ``mode``."""
    return _FEATURE_MODE_TAGS[mode]
