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
# Leaf code widths the scorer reads: one or two nibble planes per leaf.
LEAF_BITS_CHOICES = (4, 8)
# A chunk's codes are summed in byte lanes before widening, so chunk * (2**4 - 1) must fit a byte.
MAX_LEAF_CHUNK = 17


@dataclass
class ModelConfig:
    """Gradient-boosted symmetric (oblivious) tree model hyper-parameters."""

    depth: int = 7  # Tree depth; one split per level, 2**depth leaves per tree.
    n_trees: int = 2400  # Number of boosting iterations (symmetric trees).
    learning_rate: float = 0.1  # CatBoost learning_rate / shrinkage per tree.
    border_count: int = 15  # Per-feature split candidates; <= 15 for the 4-bit blob.
    random_seed: int = 42  # Seed for the booster; independent of the data split.
    # Class weighting instead of (or on top of) negative subsampling: CatBoost's
    # scale_pos_weight multiplies every positive's gradient; None = 1.
    scale_pos_weight: float | None = None
    # Leaf values live on a low-bit grid (4, 6, 8 or 16 bits); None keeps float32
    # leaves.  8-bit costs no measurable accuracy and is the format a low-bit
    # scorer reads, so models are trained for it by default.
    # Leaf values live on a low-bit grid the scorer reads directly: per tree an
    # offset, per chunk of leaf_chunk trees a power-of-two step, per leaf a
    # leaf_bits code.  8-bit costs nothing measurable; 4-bit halves the fine-tree
    # work again and needs the quantisation-aware fit below.
    leaf_bits: int = 8
    leaf_chunk: int = 16  # Trees sharing one step; also the scorer's pass length (<= 17).
    # Quantisation-aware training: fit in chunks of leaf_chunk trees and quantise
    # each chunk's leaves before fitting the next, so later trees correct the
    # rounding of earlier ones and the fitted model IS the shipped model.
    quantisation_aware: bool = True
    # Resolution-tiered boosting: the first coarse_trees trees may only split on
    # features that are constant inside a 4x4 cell tile (side <= coarse_max_side);
    # the scorer evaluates them once per tile (a fifth of a fine tree, or less),
    # and their tile-level score gates the fine trees (early exit below).
    coarse_fraction: float = 2.0 / 3.0  # Share of n_trees in the coarse tier (0 disables tiering).
    coarse_max_side: int = 16
    # Early exit: after each stage a tile whose cells all score below the stage's
    # threshold keeps its partial score.  Thresholds are calibrated at fit time on
    # the training images: the lowest partial score of any cell that ends above
    # exit_keep_prob, minus exit_margin.  exit_stage_fractions place the stages
    # inside the fine tier (0 = right after the coarse tier).
    use_exit: bool = True
    exit_keep_prob: float = 0.05
    exit_margin: float = 2.0
    exit_stage_fractions: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75)
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
        self._validate_leaf_grid()
        self._validate_tiers()

    def _validate_leaf_grid(self) -> None:
        """Check the leaf quantisation settings."""
        if self.leaf_bits not in LEAF_BITS_CHOICES:
            msg = f"leaf_bits must be one of {sorted(LEAF_BITS_CHOICES)}"
            raise ValueError(msg)
        if not 1 <= self.leaf_chunk <= MAX_LEAF_CHUNK:
            msg = f"leaf_chunk must be in [1, {MAX_LEAF_CHUNK}]"
            raise ValueError(msg)
        if not 0.0 < self.exit_keep_prob < 1.0:
            msg = "exit_keep_prob must be in (0, 1)"
            raise ValueError(msg)
        if self.exit_margin < 0.0:
            msg = "exit_margin must be >= 0"
            raise ValueError(msg)
        if self.scale_pos_weight is not None and self.scale_pos_weight <= 0.0:
            msg = "scale_pos_weight must be > 0"
            raise ValueError(msg)
        self.exit_stage_fractions = tuple(float(f) for f in self.exit_stage_fractions)
        if any(not 0.0 <= f < 1.0 for f in self.exit_stage_fractions):
            msg = "exit_stage_fractions must be in [0, 1)"
            raise ValueError(msg)

    @property
    def coarse_trees(self) -> int:
        """Trees in the coarse tier: ``coarse_fraction`` of ``n_trees``, at least one fine tree left."""
        return min(round(self.n_trees * self.coarse_fraction), self.n_trees - 1)

    def _validate_tiers(self) -> None:
        """Check the resolution-tiering settings."""
        if not 0.0 <= self.coarse_fraction < 1.0:
            msg = "coarse_fraction must be in [0, 1)"
            raise ValueError(msg)
        if self.coarse_max_side < 1:
            msg = "coarse_max_side must be >= 1"
            raise ValueError(msg)


@dataclass
class TrainConfig:
    """Front-end, sampling and pruning knobs (how features become a dataset)."""

    # The front-end matches framegate's single imfeat pass (1024 px square, HSV,
    # stride 4, 64x64 finest grid, six dyadic levels) so a framegate process can
    # hand its Pyramid straight to a fastdet model instead of running a second pass.
    levels: tuple[int, ...] = (64, 32, 16, 8, 4, 2)  # Grid sizes per level; finest must be 64.
    feature_mode: str = "raw_plus_global_context_ext"  # Which feature banks to build.
    thumb: int = 1024  # Square resize target the feature pyramid is computed on.
    stride: int = 4  # imfeat sampling stride at the primary scale.
    extra_scales: str = ""  # Extra imfeat scales, 'thumb:stride;...' ('' disables).
    resize_interp: str = "area"  # Thumbnail resize kernel: 'area' or 'nearest'.
    imfeat_space: str = "hsv"  # Color space fed to imfeat: hsv|lab|luv|yuv.
    gt_cell_thresh: float = 0.10  # Cell coverage >= this is a positive training label.
    # 0 = keep every column.  The bundled ranking is keyed to the raw column names
    # of an older imfeat build; regenerate it from a full-width fit before pruning
    # again (see training.default_feature_ranks_path).
    top_k_features: int = 0
    feature_ranks: str | None = None  # Gain-ranking JSON; None = the bundled frozen ranking.
    split_seed: int = 42  # Seed for the group-aware train/val split.
    val_frac: float = 0.15  # Fraction of near-duplicate groups held out for validation.
    max_hamming: int = 8  # pHash Hamming distance below which images are one group.
    neg_pos_ratio: float | None = None  # Negative:positive sampling ratio; None = keep all.
    max_train_cells: int | None = None  # Cap on sampled training cells; None = no cap.

    def __post_init__(self) -> None:
        """Normalize multi-valued fields and validate the front-end knobs."""
        self.levels = tuple(sorted({int(v) for v in self.levels}, reverse=True))
        if not self.levels or self.levels[0] != _FINEST_GRID:
            msg = f"levels must start at the {_FINEST_GRID}x{_FINEST_GRID} output grid"
            raise ValueError(msg)
        if any(v < 1 or v & (v - 1) for v in self.levels):
            msg = "levels must be powers of two (dyadic pyramid)"
            raise ValueError(msg)
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
        d["model"]["exit_stage_fractions"] = list(self.model.exit_stage_fractions)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        """Rebuild a :class:`Config` from :meth:`to_dict` output (or a subset)."""
        data = dict(data or {})
        model = ModelConfig(**(data.get("model") or {}))
        train_d = dict(data.get("train") or {})
        if "levels" in train_d:
            train_d["levels"] = tuple(train_d["levels"])
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
_FINEST_GRID = 64  # The output grid; mirrors features.GRID.
_SPACE_LETTERS = {"hsv", "lab", "luv", "yuv"}
# The bar detector ("bard") is no longer a fastdet bank: imfeat computes it inside
# the raw block, so it arrives with "raw" and needs no tag of its own.
_FEATURE_MODE_TAGS = {
    "raw": ["raw"],
    "raw_plus_global": ["raw", "global"],
    "raw_plus_global_context": ["raw", "global", "context"],
    "raw_plus_global_context_ext": ["raw", "global", "context", "ctx2"],
}


def iter_feature_mode_tags(mode: str) -> Iterable[str]:
    """Return the ordered feature-bank tags built for ``mode``."""
    return _FEATURE_MODE_TAGS[mode]
