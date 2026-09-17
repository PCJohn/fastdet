# fastdet

Fast per-cell text detection over the `imfeat` multi-level feature pyramid.

`fastdet` trains a symmetric (oblivious) CatBoost model on features extracted at
a 64/32/16/8 cell pyramid, prunes the columns to a fixed ranked subset, and
exports **one self-contained file**. A Python runtime and a small C++ runtime
score *identical bytes*, so a deployed model needs no Python.

## Quickstart

```python
from fastdet import Detector

det = Detector().fit("images", "masks")      # frozen defaults
det.export("model.fdt")                      # ONE self-contained file

det = Detector.load("model.fdt")
prob = det.predict_proba("page.png")         # 64x64 per-cell probability map
```

Tweak the booster without touching anything else:

```python
det = Detector(depth=7, n_trees=2400)        # **model overrides
det = Detector.from_config_file("cfg.yaml")  # full Config from JSON/YAML
```

## Install

```sh
pip install -e fastdet[dev,yaml]             # from the repo root
```

Runtime deps: `numpy`, `opencv-python`, `imfeat`, `catboost`. `PyYAML` is only
needed for YAML configs.

## API

| Call | Purpose |
| --- | --- |
| `Detector(config=None, **model_overrides)` | Build with frozen defaults or overrides |
| `Detector.from_config_file(path)` | Build from a JSON/YAML `Config` |
| `Detector.load(path)` | Load an exported single file |
| `fit(images_dir, masks_dir, *, evaluate=True)` | Group-aware split, cache, prune, fit |
| `prune(top_k=None)` | Keep the top-`top_k` ranked columns |
| `predict_proba(image, grid=GRID)` | `grid x grid` per-cell probabilities |
| `export(path)` / `build_artifact()` | Write / construct the single-file artifact |

## File format

A saved model is one file. There are two layered containers.

### `FDT1` container

Little-endian:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | 4 bytes | magic `FDT1` |
| 4 | uint32 | container version (1) |
| 8 | uint32 | JSON header length |
| 12 | JSON | header, space-padded to an 8-byte boundary |
| 12 + len | bytes | `IMSY` blob, verbatim |

The header holds `config`, `feature_names`, `metadata` and `blob_bytes`.

### `IMSY` blob (version 3)

Little-endian; a bare blob (no `FDT1` wrapper) is also accepted by both runtimes.

| Offset | Type | Field |
| --- | --- | --- |
| +0 | 4 bytes | magic `IMSY` |
| +4 | uint32 | version (2 or 3) |
| +8 | uint32 | `n_trees` |
| +12 | uint32 | `n_features` |
| +16 | uint32 | `n_leafs_total` |
| +20 | uint32 | `depth` |
| +24 | uint32[] | `tree_offsets[n_trees + 1]` — leaf-value start per tree |
| | uint32[] | `tree_base[n_trees + 1]` — split-byte offset per tree |
| | (u16,u8,u8)[] | `split[n_trees * depth]` — feature, bin, pad |
| | float32[] | `leaf_values[n_leafs_total]` |
| | uint32[] | `n_borders[n_features]` |
| | float32[] | `borders[sum(n_borders)]` |
| | uint8[] | `level_shift[n_features]` |
| v3 | uint8[] | `shuffle_tables[n_trees * depth * 16]`, `T[v] = (v > bin) ? (1<<d) : 0` |

A split at level `d` (0 = root) sets leaf bit `d` when the feature's byte-space
bin is **strictly greater** than the split bin; the leaf index is the sum of
those bits (root split = lowest bit). Binning is
`searchsorted-left(borders[f], x)`. Version 3 adds 4-bit `vpshufb` tables so the
shipped C++ path never compares — it ORs a table lookup per split. Borders are
limited to `MAX_BORDER_COUNT = 15` so a bin fits a nibble.

## Frozen defaults (the shipped model)

These are the measured, frozen defaults in `ModelConfig` / `TrainConfig`.

| Group | Setting |
| --- | --- |
| Model | depth 7, 2400 trees, lr 0.1, `border_count` 15, `Logloss`, `SymmetricTree`, seed 42 |
| Pyramid | levels 64/32/16/8, `thumb` 512, `stride` 2, `extra_scales` `256:1` |
| Banks | `raw_plus_global_bard_context_ext`, `lab` (imfeat space), `lstar` (grey), bard lags (1, 2, 4), tau 8.0 |
| Data | `gt_cell_thresh` 0.1, `val_frac` 0.15, `max_hamming` 8, split seed 42, `top_k_features` 512 |

Full feature width is **996**; the pruned model keeps **512** columns; the
frozen validation split is **288 images**.

Measured (frozen reference): pooled validation **PR-AUC ≈ 0.8106**, full C++
model **≈ 5.8 ms** per 64x64 image on the reference machine.

## C++ runtime

`cpp/fastdet_score.cpp` is a dependency-free reader/scorer for the same file. It
keeps exactly one shipped traversal (an AVX2 nibble path) plus one scalar
reference path and a fused packed binner, and it self-checks the binner and the
two traversals against each other.

Build (MSVC, from `fastdet/cpp`):

```bat
call "%VS%\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 fastdet_score.cpp /Fe:fastdet_score.exe
```

Run:

```bat
fastdet_score.exe model.fdt fixtures.f32 [expected.f32] [iters]
```

`fixtures.f32` is `4 * 4096 * n_features` bytes (little-endian float32) holding
64x64 cells for one image. With `expected.f32` supplied it also reports the max
absolute probability error.

## Development

```sh
python -m pytest          # end-to-end: fit, export one file, reload, score
python -m ruff check .    # full "ALL" ruleset
python -m black --check .
python -m mypy --strict .
```

The test suite fits a tiny synthetic detector, exports it, reloads it, and
asserts the two runtimes reproduce scores **bit-for-bit**.

## Layout

```
fastdet/
  cpp/fastdet_score.cpp      # dependency-free C++ runtime
  src/fastdet/               # config, images, features, dataset, training,
                             # exporter, artifact, runtime, metrics, detector
  src/fastdet/data/          # bundled frozen feature ranking
  tests/                     # end-to-end round-trip tests
```
