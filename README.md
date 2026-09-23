# fastdet

A fast per-cell text detector. It classifies every cell of a 64×64 grid as text
or not-text using classical image features and a small gradient-boosted tree
model, then exports the result as **one self-contained file**. A Python runtime
and a dependency-free C++ runtime read the same bytes and produce identical
scores.

The front-end matches [framegate](https://github.com/PCJohn/framegate)'s single
imfeat pass (1024 px square, HSV, stride 4, 64×64 finest grid, six levels), so a
framegate process can feed its own features to a fastdet model. **No model has
been trained or measured on this front-end yet**: the quality and latency figures
in [`research_report.md`](research_report.md) describe earlier front-ends
(512 px, CIELAB, an extra 256 px scale) and do not transfer. Regenerate the
feature ranking from a full-width fit before pruning (see *Pruning*).

## Install

```sh
git clone https://github.com/PCJohn/fastdet.git
cd fastdet
pip install -e ".[yaml,dev]"
```

Runtime dependencies are `numpy`, `opencv-python`, `imfeat`, and `catboost`.
`yaml` adds PyYAML for YAML configs; `dev` adds the test and lint tools. Drop
both extras for a minimal install.

## Quickstart

```python
from fastdet import Detector

det = Detector().fit("data/images", "data/masks")
det.export("model.fdt")

det = Detector.load("model.fdt")
prob = det.predict_proba("data/test/page.png")   # (64, 64) float32 in [0, 1]
```

## Training

Every image in `images/` needs a matching mask in `masks/` named
`<image-name>_mask.<ext>`, white on text and black elsewhere.

```python
from fastdet import Detector

det = Detector()                          # frozen, measured defaults
# det = Detector(depth=7, n_trees=2400)   # override booster fields only
# det = Detector.from_config_file("config.yaml")

det.fit("data/images", "data/masks")
print("PR-AUC", det.metrics["pr_auc"])
print("columns", len(det.feature_names))
```

`fit` runs the whole pipeline:

1. pair images with masks, hash each image, and split train/val by
   near-duplicate cluster so no cluster straddles the boundary;
2. cache the feature maps for the training images;
3. prune the design matrix to the top `top_k_features` columns by gain;
4. sample cells and fit the CatBoost booster;
5. score the validation split and record pooled PR-AUC.

Every knob lives in [`Config`](src/fastdet/config.py#L117) and has a default
that was measured on the reference data. No GPU and no config file are needed.

## Pruning and export

Columns are ranked once by split gain on the frozen front-end; the ranking ships
in the package (`src/fastdet/data/feature_ranks.json`). Pruning keeps the top
`top_k` columns (`0` = keep all, the default until the ranking is regenerated
for the current front-end). Choose a subset before fitting:

```python
from fastdet import Detector

det = Detector()
det.prune(top_k=256)                      # or set TrainConfig.top_k_features
det.fit("data/images", "data/masks")      # trains on the 256-column matrix

path = det.export("model.fdt")            # one self-contained file
print(f"{path.stat().st_size:,} bytes")
```

Pass `TrainConfig.feature_ranks` to rank a different front-end. `top_k=0` keeps
every column. `export` writes the model, the kept feature names, the config, and
the validation metadata into one `FDT1` file.

## Load and run inference

An exported file is all a consumer needs: no training data, no CatBoost, no
Python.

```python
from pathlib import Path

import numpy as np

from fastdet import Detector

det = Detector.load("model.fdt")

for path in Path("data/test").glob("*.png"):
    prob = det.predict_proba(path)         # (64, 64) float32 in [0, 1]
    cells = prob >= 0.5                    # threshold to a text mask
    ys, xs = np.nonzero(cells)             # text-cell coordinates
```

`predict_proba` takes a path or a BGR `uint8` array. Scale the grid back to the
source resolution to crop or overlay. The C++ runtime scores the same file; see
[C++ runtime](#c-runtime).

## How it works

### Features

Each image is squashed to a 1024×1024 thumbnail (`INTER_AREA`; the aspect ratio
is kept as a global feature instead), converted to HSV, and passed to `imfeat`
once at stride 4 with six pyramid levels (64/32/16/8/4/2 cells) — framegate's
exact configuration. For every cell of the finest 64×64 grid, the front-end
concatenates the features of all levels into one row (1178 columns):

- `raw` — imfeat's per-channel block per cell, per scale. This includes the
  multi-lag bar detector (`bard_*`), which fires when a pixel is darker or
  brighter than its neighbours on both sides: it is the one new feature that
  survived ablation (research report §3.5), and it now lives in imfeat rather
  than here;
- `global` — imfeat's whole-image block plus the original frame's aspect ratio
  and log area, broadcast to every cell;
- `context`, `ctx2` — small- and large-scale surround, ring, and range of
  imfeat's per-cell luminance mean.

**fastdet computes no image features of its own.** Every per-pixel quantity
comes from imfeat in a single pass; the banks above are cheap reductions of its
output. The assembly lives in
[`FeatureExtractor.extract`](src/fastdet/features.py) and
[`FeatureExtractor.gather`](src/fastdet/features.py).

### Labels and split

A cell is positive when at least 10% of its pixels are white in the mask
(`gt_cell_thresh = 0.1`); the rest are negative. The reference set is 1925
images, about 6.7 M cells, 7.3% positive.

A random split leaks near-duplicates across train and validation. Instead,
[`build_split`](src/fastdet/dataset.py#L182) hashes every image, groups images
within Hamming distance 8 of one another
([`near_duplicate_groups`](src/fastdet/dataset.py#L122)), and holds out whole
groups ([`grouped_split`](src/fastdet/dataset.py#L158)); the reference split is
1637 train / 288 validation. Cells are labelled and sampled by
[`sample_training_cells`](src/fastdet/dataset.py#L220) and gathered into the
design matrix by [`gather_training_matrix`](src/fastdet/dataset.py#L281).

### The model

The model is a CatBoost
[symmetric (oblivious)](https://catboost.ai/docs/) gradient-boosted tree
ensemble. In a symmetric tree every node at the same depth shares one
`(feature, border)` split, so a depth-7 tree has 2⁷ leaves and only 7 splits.
[`fit_booster`](src/fastdet/training.py#L67) fixes the run:

| Parameter | Value | Role |
| --- | --- | --- |
| `loss_function` | `Logloss` | binary cross-entropy per cell |
| `iterations` | 2400 | number of trees |
| `depth` | 7 | levels per tree, 128 leaves |
| `learning_rate` | 0.1 | shrinkage per tree |
| `border_count` | 15 | split candidates per feature |
| `grow_policy` | `SymmetricTree` | required by the export format |
| `random_seed` | 42 | reproducibility |

Prediction sums one leaf value per tree and applies a sigmoid
([`ImysModel.predict_proba`](src/fastdet/runtime.py#L95)). Training minimizes
Logloss; the reported metric is PR-AUC, which suits the 7% positive rate far
better than accuracy.

### Regularization

The ensemble is regularized by construction and by the usual tree controls:

- **depth 7** limits interaction order to seven splits;
- **learning rate 0.1** shrinks each tree's contribution;
- **L2 leaf regularization** (CatBoost default `l2_leaf_reg = 3`) penalizes leaf
  magnitudes;
- **stochastic bootstrap** (CatBoost default MVS, `subsample = 0.8`) adds row
  noise;
- **`border_count = 15`** coarsens the split points;
- the **symmetric constraint** is itself a strong prior — one split per level,
  shared by every leaf, gives far fewer free parameters than a leaf-wise tree of
  the same depth.

No early stopping is used; the validation plateau was reached by tuning tree
count and learning rate explicitly. See the
[training parameters reference](https://catboost.ai/docs/en/references/training-parameters/).

### Pruning

One full-width training run produces a split-gain ranking; the bundled JSON
stores the descending column names.
[`select_columns`](src/fastdet/training.py#L45) keeps the highest-ranked columns
in canonical order, and [`Detector.prune`](src/fastdet/detector.py#L105) wires
that into the pipeline. Pruning reduces model cost only: the front-end still
computes every column, so it does not speed up `imfeat`.

> **Note.** The bundled ranking was produced against an older front-end whose
> columns differ, so `top_k_features` defaults to `0` (keep everything).
> Regenerate it from a full-width fit with
> [`build_ranking`](src/fastdet/training.py):
>
> ```python
> full = Detector().fit(images_dir, masks_dir)          # top_k_features = 0
> ranking = build_ranking(full.booster, full.feature_names)
> Path("ranks.json").write_text(json.dumps(ranking))
> cfg = Config()
> cfg.train.top_k_features, cfg.train.feature_ranks = 512, "ranks.json"
> det = Detector(cfg).fit(images_dir, masks_dir)
> ```

Column selection defines the matrix the booster is trained on, so `prune()` is a
pre-`fit` step. Calling it on a fitted detector discards the booster and the
runtime rather than leaving the kept columns and the blob's feature indices
describing different matrices; refit before scoring again.

### Export and bundling

[`build_blob`](src/fastdet/exporter.py#L144) reads the JSON that CatBoost writes
with `save_model(format="json")` and re-encodes it without approximation —
borders, split bins, and leaf values are copied verbatim. Two details keep the
blob small and fast:

- a per-feature `level_shift` records how many pyramid levels a column is
  constant across, so coarse cells share bins instead of repeating values;
- version 3 stores one 16-byte table per split,
  `T[v] = (v > bin) ? (1 << d) : 0`
  ([`_shuffle_table_bytes`](src/fastdet/exporter.py#L129)). With 4-bit bins a
  threshold test becomes a single `vpshufb` lookup and OR.

[`ModelArtifact`](src/fastdet/artifact.py#L42) wraps the blob in an `FDT1`
container with the config, the kept feature names, and metadata.
[`parse_blob`](src/fastdet/runtime.py#L112) in Python and `load_imys` in C++
decode the identical bytes; the tests assert bit-for-bit agreement.

### Evaluation

Validation reports pooled PR-AUC over all cells
([`pooled_pr_auc`](src/fastdet/metrics.py#L25)). Model comparisons use a paired
image-level cluster bootstrap: resample validation images, re-pool their cells,
and compare the difference
([`paired_image_bootstrap`](src/fastdet/metrics.py#L37)). This stops one
text-dense image from deciding a verdict. Pooled PR-AUC is optimistic for
per-image thresholding — the reference model's mean per-image average precision
is ≈ 0.74 against a pooled ≈ 0.81 — so report the per-image number when applying
a per-image threshold. See research report §3.6 and
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

### Why inference is fast

Symmetric trees turn scoring into independent work, which the runtime exploits:

- a depth-7 tree is 7 independent comparisons producing a 7-bit leaf index, with
  no dependency chain;
- 4-bit bins make each split a `vpshufb` table lookup instead of
  compare-mask-shift;
- cells are nibble-packed, halving matrix bandwidth;
- the binner is fused over coarse cells because the design matrix repeats
  coarse-level columns about 4.3×.

The layered optimizations and their measurements are in research report §3.8.
The approach follows Aggregated Channel Features (see
[Piotr Dollár's toolbox](https://pdollar.github.io/toolbox/)).

## File format

A saved model is one file with two layers.

### `FDT1` container

Little-endian:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | 4 bytes | magic `FDT1` |
| 4 | uint32 | container version (1) |
| 8 | uint32 | JSON header length |
| 12 | JSON | header, space-padded to an 8-byte boundary |
| 12 + len | bytes | `IMSY` blob, verbatim |

The header holds `config`, `feature_names`, `metadata`, and `blob_bytes`.

### `IMSY` blob (version 3)

Little-endian. A bare blob (no `FDT1` wrapper) is also accepted by both runtimes.

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
| v3 | uint8[] | `shuffle_tables[n_trees * depth * 16]`, `T[v] = (v > bin) ? (1 << d) : 0` |

A split at level `d` (0 = root) sets leaf bit `d` when the feature's byte-space
bin is **strictly greater** than the split bin; the leaf index is the sum of
those bits, so the root split is the lowest bit. Binning is
`searchsorted-left(borders[f], x)`. Version 3 adds 4-bit `vpshufb` tables, so the
C++ path never compares — it ORs a table lookup per split. `border_count` is
capped at `MAX_BORDER_COUNT = 15` because a bin must fit a nibble.

## C++ runtime

`cpp/fastdet_score.cpp` is a dependency-free reader and scorer for the same
file. It ships one SIMD traversal plus a scalar reference path and one binner,
and it self-checks the binner and both traversals against each other.

The scorer uses [Google Highway](https://github.com/google/highway) for SIMD,
exactly as `imfeat` does: static dispatch, with the target chosen by the compiler
flags, so one source runs SSE4/AVX2/AVX-512 on x86 and NEON/SVE on Arm.

The traversal is built on the shape of the features. They form a pyramid, so most
splits test a value that is constant over a block of cells: in a default-size
model about 40% of the splits are constant on any 4×4 block of cells. Cells are
therefore scored in 4×4 **tiles** (16 cells, one 128-bit vector of byte lanes,
two tiles per AVX2 vector), and each tree's splits are divided in two:

- splits on features of side ≤ 16 (and image-wide features) are constant on a
  tile. They are evaluated once per tile, at 16×16 resolution, and only choose
  *which group of leaves* the tile can reach;
- the `v` splits on side-64 and side-32 features vary inside the tile and index
  within that group. Leaves are permuted per tree at load time so these are the
  low index bits. With `v ≤ 4` the group has at most 16 leaves, and the leaf
  values are fetched by four byte shuffles (one per byte of the float32) instead
  of a gather; `v = 5` uses two shuffles and a blend, `v = 0` a broadcast, and
  only `v ≥ 6` falls back to a gather.

The loop order is tree-outer: one tree runs over every pack of tiles with its
dispatch on `v`, its split tables and its plane pointers hoisted, and the running
sums live in a 16 KB buffer rather than in registers (11% faster than the
reverse, and each tree streams its few planes sequentially). Every cell still
adds its trees in order, so the result is bit-identical to the scalar reference.

The binner bins only the features the model uses, each value once, straight into the two layouts the traversal reads (tile-major bytes for
the varying features, one byte per tile for the rest), with the comparison loop
unrolled over the feature's cut count. On a depth-7 × 2400 model fitted to
synthetic text images this measured 2.6 ms per image (0.45 ms binning + 2.1 ms
traversal; AVX2, one thread) against 5.4 ms (0.9 + 4.6) for the per-cell,
gather-based traversal it replaced, and 21 M instead of 39 M instructions per
traversal. The program prints how many trees fall in each `v` class; that
distribution decides the speed, so check it on a real model. CMake fetches
Highway 1.2.0 (the same pin as `imfeat`):

```sh
cmake -S cpp -B build && cmake --build build --config Release --target fastdet_score
```

The default target is the build machine (`-march=native`; `/arch:AVX2` with
MSVC). For a portable or cross build set `-DFASTDET_ARCH_FLAGS=...`. Note that
GCC's `-march=haswell` does not enable AES, which Highway's AVX2 target requires,
so it silently falls back to 16-byte vectors; use
`"-march=haswell -maes -mpclmul"`. To build offline, point CMake at a local
Highway checkout with `-DFETCHCONTENT_SOURCE_DIR_HIGHWAY=/path/to/highway`. The
program prints the Highway target it was built for.

Run:

```bat
fastdet_score model.fdt fixtures.f32 [expected.f32] [iters]
```

`fixtures.f32` holds one image's kept features at their **native resolution** —
exactly what `Detector.native_matrix(image)` returns. Each feature, in model
order, contributes its level's `side × side` little-endian float32 values
row-major, where `side = 64 >> (level_shift / 2)`: 64 for the finest level, 2 for
the coarsest, and 1 for an image-wide global. No dense `4096 × n_features` table
is built on the inference path: a level-8 feature is 64 values rather than 4096
copies of them, and the binner bins each distinct value once. For the default
front-end that is 3.7 MB instead of 19.3 MB per image, and binning is ~6× faster.
(`Detector.design_matrix(image)` still returns the dense matrix; it feeds training
and the Python reference runtime, and expanding every native value over its
block reproduces it exactly.) With `expected.f32` (4096 float32, e.g.
`Detector.predict_proba(image).reshape(-1)`), the program also reports the
maximum absolute probability error and exits non-zero if any gate fails.

**Optional early exit (experimental).** A fifth argument, `"trees:theta,..."`,
adds stages to the same traversal: once `trees` trees have run, a pack of tiles
in which every raw sum is below `theta` is dropped and its cells keep that
partial sum. A pack that survives has run every tree in order, so
its scores are bit-identical to the full evaluation; only confidently negative
tiles are cut short. The program times it next to the full evaluation and prints
how many cells stayed identical and the highest full probability among those
that did not. On the synthetic model above, with thresholds taken from the
training images (the lowest partial sum of any cell that ends at p ≥ 0.05,
minus 2), traversal on the six validation images fell from 2.1 ms to 0.7–1.2 ms
and no cell ending at p ≥ 0.05 was cut short. The thresholds are a property of
the data: calibrate them on yours, and validate PR-AUC, before using this.

## Development

```sh
python -m pytest                    # fit, export one file, reload, score end to end
python -m pytest -m slow             # also build and gate the C++ runtime
python -m ruff check .              # full "ALL" ruleset
python -m black --check .
python -m mypy --strict .
```

The default suite fits a tiny synthetic detector, exports it, reloads it, and
asserts the reloaded model reproduces scores exactly. `tests/test_cpp_runtime.py`
(marked `slow`) additionally builds `cpp/` with CMake (set `FASTDET_HWY_DIR` to a
local Highway checkout to build offline), feeds it the same
artifact plus a fixture built by `Detector.native_matrix` for every image, and
fails if the C++ scorer diverges from the Python one (which scores the dense
matrix, so the two share no feature-layout code) or if its internal
binner/traversal gates fail. A hand-built one-split model pins the image-wide
(global) binning path deterministically. It skips when cmake is unavailable or
the scorer cannot be built.

## Layout

```
fastdet/
  cpp/fastdet_score.cpp      # dependency-free C++ runtime
  src/fastdet/               # config, images, features, dataset, training,
                             # exporter, artifact, runtime, metrics, detector
  src/fastdet/data/          # bundled frozen feature ranking
  tests/                     # end-to-end round-trip + C++ runtime gate
  research_report.md         # full research log and negative results
```
