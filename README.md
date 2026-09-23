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

### Leaf quantisation

Every model ships with its leaves on a low-bit grid, and is *trained for* that grid:

* per tree an `offset` (its smallest leaf), per chunk of `leaf_chunk` trees a
  power-of-two `step`, per leaf a `leaf_bits` code (8 by default, 4 optional):
  `leaf = offset + code * step`. One step per chunk because leaf magnitudes shrink
  as boosting proceeds; a single global step would quantise the later trees into
  noise. Chunks restart at the coarse/fine boundary (below).
* the fit is quantisation-aware: each chunk is fitted on top of the **quantised**
  running score of the previous ones (CatBoost `baseline`), so later trees correct
  the rounding of earlier ones, and the export reproduces exactly the grid the fit
  was steered to (same chunking, same power-of-two steps).
* both runtimes sum the codes as integers (`total += code << shift[chunk]`) and
  turn the integer into a score once, `sum(offsets) + total * 2**e_min`, so the
  Python reference and the C++ scorer agree to the bit, and the scorer's fine-tree
  work is one byte shuffle per nibble plane of codes instead of four per float leaf.

```python
det = Detector(Config()).fit(images_dir, masks_dir)                 # 8-bit, QAT, tiered, exit
det = Detector(Config(model=ModelConfig(leaf_bits=4))).fit(...)     # 4-bit
```

`fit()` reports validation PR-AUC from the exported runtime with early exit on:
the number is the shipped model's, not the float booster's.

### Resolution-tiered boosting

The first `coarse_fraction` of the trees (two thirds by default) may only split on
features that are constant inside a 4×4 cell tile (`coarse_max_side`, 16); the rest
see every feature. A tile-constant tree is evaluated **once per tile** by the scorer
(a fifth of a fine tree, or less), and its tile-level score is what the early exit
gates on. On the synthetic validation set the tiered model scores 0.9012 against
0.9014 for an untiered one.

### Early exit

After the coarse tier, and at `exit_stage_fractions` of the fine tier, a tile whose
cells all score below the stage's threshold stops accumulating and keeps its partial
score. The thresholds are calibrated at fit time on the training images: the lowest
partial score of any cell that ends at or above `exit_keep_prob` (0.05), minus
`exit_margin` (2.0 in raw-score units). They travel in the blob, both runtimes apply
them (`use_exit`), and the C++ scorer additionally bins the side-64 features only
for the tiles still alive after the coarse tier (lazy binning).

The exit is a contract about cells that end above `exit_keep_prob`: on the
calibration images none of them can be stopped, and the margin covers what unseen
images move. Cells that are stopped keep a score below the threshold, so anything a
consumer thresholds at or above `exit_keep_prob` is unaffected; a consumer that
needs exact scores for every cell sets `use_exit=False` (the C++ scorer: no stage
argument, or an empty one).

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

### `IMSY` blob (version 4)

```
+0    "IMSY"                      +24   u32 leaf_bits (4|8)     +40  u32 n_chunks
+4    u32 version = 4             +28   u32 leaf_chunk          +44  u32 n_stages
+8    u32 n_trees                 +32   u32 coarse_trees
+12   u32 n_features              +36   i32 e_min
+16   u32 n_leafs_total
+20   u32 depth
+48   u32 tree_offsets[n_trees+1]   leaf-code start per tree
      u32 tree_base[n_trees+1]      split-byte offset per tree
      split[n_trees*depth]          u16 feature, u8 bin, u8 pad   (root split = index bit 0)
      u8  codes[n_leafs_total]      leaf codes
      f32 offsets[n_trees]
      u8  shifts[n_chunks]          chunk step = 2**(e_min + shift)
      stage[n_stages]               u32 trees, f32 threshold (raw score)
      u32 n_borders[n_features], f32 borders[...], u8 level_shift[n_features]
```

Chunks are `leaf_chunk` consecutive trees, restarting at `coarse_trees`. A cell's
raw score is `sum(offsets) + total * 2**e_min` with `total` the integer sum of
`code << shift`. See `fastdet.runtime` for the reference reader.

## C++ runtime

`cpp/fastdet_score.cpp` is a single-file Highway program that reads the exported
model, bins the native-resolution features, walks the trees and reports timings.
Build it with CMake (Highway is fetched automatically) and run:

```
fastdet_score model.fdt fixture.f32 [expected.f32] [iters] [stages]
```

`fixture.f32` is `Detector.native_matrix(image)` as little-endian float32,
`expected.f32` the Python runtime's full probabilities (`predict_grid(...,
use_exit=False)`). The program gates itself: the tile binner must reproduce the
reference bins exactly and the SIMD traversal must reproduce the scalar integer
reference to the bit, and it exits non-zero otherwise. `stages` overrides the blob's
calibrated exit stages (`trees:theta,...`, applied at chunk ends; an empty string
disables the exit).

What it does, in order:

1. **Binning.** Each feature is binned once per distinct value (a level-L feature
   has L×L values), by comparing against its cut list in SIMD and counting; the
   bins are written as one byte plane per feature in **tile-major** order (a 4×4
   tile is 16 consecutive bytes), coarse features as one byte per tile.
2. **Coarse tier.** Trees that split only on tile-constant features are evaluated
   once per tile: their leaf index is built with byte shuffles over the coarse
   planes and the leaf code gathered per tile.
3. **Early exit and lazy binning.** After the coarse tier's stage, only the packs
   of tiles still alive get their side-64 features binned, and only they are
   scored by the fine trees.
4. **Fine trees.** Splits that vary inside a tile take the low bits of the leaf
   index and are looked up with one byte shuffle each over 32 cells; the
   tile-constant splits select the leaf group per tile. Each nibble plane of the
   group's codes is fetched with one byte shuffle and added into byte-lane sums,
   which are widened once per chunk of trees with the chunk's shift into the
   integer totals. No float arithmetic until the final score.
5. **Score.** `sigmoid(sum(offsets) + total * 2**e_min)` per cell, in row-major order.

Timings printed: `tile binner`, `simd traversal` (every tree on every cell),
`model total` (binning + full traversal) and `model, as shipped` (lazy binning,
coarse tier per tile, early exit) -- the last one is the production number.

## Hyperparameter tuning (`fastdet-tune`)

`fastdet-tune` fits one detector per combination of the knobs you name, scores the
held-out split exactly as the shipped runtime does (quantised leaves, coarse tier,
early exit), and writes a report folder:

* `report.md` -- one row per combination (only the swept knobs appear as columns) with
  PR-AUC (primary), ROC-AUC, best F1 and its threshold, precision / recall / IoU at that
  threshold, fit time, model size and, when the C++ library is built, `predict_proba`
  latency; the framegate text heuristic as a baseline row; the best run's full config.
* `pr.png`, `roc.png` -- precision-recall and ROC curves of the best runs and the baseline.
* `results.json`, `models/<key>.fdt` -- every run's config, metrics and exported model.
  Re-running with the same `--out` resumes: finished combinations are skipped.

```
pip install -e ".[tune]"                     # matplotlib, for the charts
fastdet-tune --images data/images --masks data/masks --out tune_report \
    --n-trees 1200,2400 --leaf-bits 4,8 --coarse-fraction 0.5,0.667
```

That is 2 x 2 x 2 = 8 fits. Every knob takes a comma-separated list and the sweep is
their Cartesian product, so name few knobs at a time; a knob you do not name keeps its
default, which is the tuned production value. `--max-runs 2` smoke-tests a sweep,
`--report-only` rebuilds the report and charts from `results.json`, `--no-baseline`
skips the heuristic. Tuples take `/`: `--levels 64/32/16/8`.

Every field of `ModelConfig` and `TrainConfig` is a knob (`fastdet-tune --help` lists
them with their current defaults). The ones worth sweeping:

| knob | default | what it does |
|---|---|---|
| `--n-trees` | 2400 | boosting iterations; latency grows linearly, quality saturates |
| `--depth` | 7 | tree depth; 7 varying splits is the scorer's limit |
| `--learning-rate` | 0.1 | shrinkage; lower needs more trees |
| `--border-count` | 15 | split candidates per feature (max 15); 7 makes binning ~40% cheaper |
| `--leaf-bits` | 8 | leaf code width, 4 or 8; 4 halves the fine-tree cost (quantisation-aware fit) |
| `--leaf-chunk` | 16 | trees per quantisation step and scorer pass (<= 17) |
| `--coarse-fraction` | 0.667 | share of trees restricted to tile-constant features (evaluated once per tile); 0 disables tiering |
| `--coarse-max-side` | 16 | largest feature grid that counts as tile-constant |
| `--use-exit` | true | early exit on |
| `--exit-keep-prob` | 0.05 | cells ending at or above this probability are never stopped |
| `--exit-margin` | 2.0 | raw-score safety margin under the calibrated thresholds |
| `--exit-stage-fractions` | 0/0.125/0.25/0.5/0.75 | where in the fine tier the stages sit |
| `--top-k-features` | 0 (all) | keep the top-k gain-ranked columns; 512 halves binning |
| `--thumb`, `--stride` | 1024, 4 | front-end resize target and imfeat sampling stride |
| `--levels` | 64/32/16/8/4/2 | pyramid grids (finest must be 64) |
| `--feature-mode`, `--imfeat-space` | raw_plus_global_context_ext, hsv | feature banks and colour space |
| `--gt-cell-thresh` | 0.10 | mask coverage at which a cell is a positive |
| `--val-frac`, `--split-seed` | 0.15, 42 | held-out share of near-duplicate groups and the split seed |
| `--neg-pos-ratio` | none (keep all) | negatives kept per positive in the training sample (`none,3,5`); rebalances the classes by discarding negatives, which shifts the probability scale (use `thr*`) |
| `--scale-pos-weight` | none (1) | CatBoost class weight on positives; the usual alternative to subsampling for boosting, keeps every negative |
| `--max-train-cells` | none | cap on sampled training cells |

Reading the report: PR-AUC is the number to rank by (the positive rate is a few
percent, so ROC-AUC flatters everything); `thr*` is the probability threshold with the
best F1 on the validation split, a reasonable operating point to ship; the latency
column is the whole in-process path (front-end + model) on the machine running the
sweep. Keep `--val-frac` and `--split-seed` fixed across a sweep so every run sees the
same held-out images.

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
