"""fastdet -- a compact per-cell document-region detector.

Train a symmetric CatBoost model on multi-level imfeat features, prune it to a
fixed column set, and export ONE self-contained file whose Python runtime and
C++ runtime score identical bytes.

Quickstart::

    from fastdet import Detector

    det = Detector().fit("images", "masks")
    det.export("model.fdt")

    det = Detector.load("model.fdt")
    prob = det.predict_proba("page.png")   # 64x64 per-cell probability map
"""

from __future__ import annotations

from .artifact import ModelArtifact
from .config import Config, ModelConfig, TrainConfig
from .detector import Detector
from .features import GRID, FeatureCache, FeatureExtractor
from .metrics import paired_image_bootstrap, pooled_pr_auc
from .runtime import ImysModel, parse_blob

__all__ = [
    "GRID",
    "Config",
    "Detector",
    "FeatureCache",
    "FeatureExtractor",
    "ImysModel",
    "ModelArtifact",
    "ModelConfig",
    "TrainConfig",
    "paired_image_bootstrap",
    "parse_blob",
    "pooled_pr_auc",
]

__version__ = "0.1.0"
