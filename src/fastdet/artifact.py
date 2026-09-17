"""Single-file model artifact: ``FDT1`` container around the ``IMSY`` tree blob.

A saved model is ONE file so a consumer never has to find a sidecar.  Layout
(little-endian):

===============  ==========================================================
``FDT1``         4-byte magic
uint32           container version
uint32           length of the JSON header in bytes
JSON header      UTF-8, padded with spaces to an 8-byte boundary (config,
                 feature names, resolved metadata, blob length)
``IMSY`` blob    the self-contained tree blob documented in :mod:`fastdet.runtime`
===============  ==========================================================

The embedded ``IMSY`` blob is the same object the C++ runtime consumes, so the
Python and C++ paths read identical bytes.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["CONTAINER_VERSION", "MAGIC", "ModelArtifact", "read_artifact", "write_artifact"]

MAGIC = b"FDT1"
CONTAINER_VERSION = 1
_HEADER_LEN = struct.Struct("<II")
_HEADER_ALIGN = 8


@dataclass
class ModelArtifact:
    """Everything a consumer needs to score cells, plus provenance."""

    config: Config
    feature_names: list[str]
    blob: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in the default provenance fields when the caller omitted them."""
        self.metadata.setdefault("container_version", CONTAINER_VERSION)
        self.metadata.setdefault("n_features", len(self.feature_names))
        self.metadata.setdefault("blob_bytes", len(self.blob))
        self.metadata.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def save(self, path: str | Path) -> Path:
        """Serialize the artifact to a single file at ``path``."""
        return write_artifact(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        """Load an artifact written by :meth:`save`."""
        return read_artifact(path)

    def header_dict(self) -> dict[str, Any]:
        """The JSON header embedded in the file (without the blob)."""
        return {
            "container_version": self.metadata.get("container_version", CONTAINER_VERSION),
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "blob_bytes": len(self.blob),
        }

    @classmethod
    def from_header(cls, header: Mapping[str, Any], blob: bytes) -> ModelArtifact:
        """Rebuild an artifact from a parsed header and the raw blob bytes."""
        return cls(
            config=Config.from_dict(header.get("config") or {}),
            feature_names=list(header.get("feature_names") or []),
            blob=blob,
            metadata=dict(header.get("metadata") or {}),
        )


def write_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    """Write ``artifact`` to ``path`` and return the path written."""
    header = json.dumps(artifact.header_dict(), separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % _HEADER_ALIGN)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(_HEADER_LEN.pack(CONTAINER_VERSION, len(header)))
        fh.write(header)
        fh.write(artifact.blob)
    return path


def read_artifact(path: str | Path) -> ModelArtifact:
    """Read and validate a single-file artifact."""
    with Path(path).open("rb") as fh:
        raw = fh.read()
    if raw[:4] != MAGIC:
        msg = f"{path}: not a fastdet artifact (bad magic {raw[:4]!r})"
        raise ValueError(msg)
    version, header_len = (int(v) for v in _HEADER_LEN.unpack_from(raw, 4))
    if version != CONTAINER_VERSION:
        msg = f"{path}: unsupported container version {version}"
        raise ValueError(msg)
    header_start = _HEADER_LEN.size + 4  # magic (4) + version + header_len
    header: dict[str, Any] = json.loads(
        raw[header_start : header_start + header_len].decode("utf-8")
    )
    blob = raw[header_start + header_len :]
    expected = header.get("blob_bytes")
    if expected is not None and expected != len(blob):
        msg = f"{path}: truncated blob ({len(blob)} != {expected} bytes)"
        raise ValueError(msg)
    retained = {k: v for k, v in header.items() if k not in ("config", "feature_names")}
    meta = retained.pop("metadata", {}) or {}
    meta.update(retained)
    return ModelArtifact(
        config=Config.from_dict(header.get("config") or {}),
        feature_names=list(header.get("feature_names") or []),
        blob=blob,
        metadata=meta,
    )
