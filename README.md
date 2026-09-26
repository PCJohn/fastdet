# fastdet

A fast per-cell text detector. It classifies every cell of a 64×64 grid as text
or not-text using classical image features and a small gradient-boosted tree
model, then exports the result as **one self-contained file**. A Python runtime
and a dependency-free C++ runtime read the same bytes and produce identical
scores.

The front-end matches [framegate](https://github.com/PCJohn/framegate)'s single
imfeat pass (1024 px square, HSV, stride 1, 64×64 finest grid, six levels), so a
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

Then build the C++ scorer once and install it into the package:

```
fastdet-native-build          # cmake on cpp/, copies fastdet_native into fastdet/_native/
```

Run it from the repository root (it looks for `cpp/` there, or next to an editable
install). A plain `pip install .` replaces the package directory, so repeat the
command after each reinstall -- or use `pip install -e .`, which keeps the built
library across reinstalls. `--build-dir build\pytest-cpp` reuses the test build.
Without the library `Detector` still works but scores with the NumPy runtime, which is
bit-identical and hundreds of times slower; every entry point warns when that happens.
`FASTDET_NATIVE_LIB=path` (or `--native-lib` on the commands) points at a library built
elsewhere. Needs `cmake` and a C++17 compiler (Visual Studio Build Tools on Windows).

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
# det = Detector(depth=5, n_trees=1000)   # override booster fields only
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

## Demo (`fastdet-demo`)

```
fastdet-demo --model model.fdt --source photo.png      # one image
fastdet-demo --model model.fdt --source clip.mp4       # a video, live
fastdet-demo --model model.fdt --source 0              # webcam 0, live
```

Left: the frame with the cell probabilities blended over it (JET colormap, opacity
following the probability). Right: latency -- for an image the two numbers, for a
video or webcam a live time series over the last 240 frames of (1) feature
extraction (resize, colour conversion, imfeat, context banks, packing) and (2) the
model (binning, coarse tier, fine trees, sigmoid), plus the total and the frame rate.
Frames are scored as they arrive; nothing is buffered ahead. The window is
matplotlib's (`pip install -e ".[tune]"`), so it works with `opencv-python-headless`;
OpenCV only decodes, resizes and colours. Keys: `q`/`Esc` quit, `space` pause, `s`
save the figure. `--headless --output out.png` renders without a window;
`--display-width` scales the frame panel; `--native-lib` points at the C++ library if
it is not found automatically (without it the NumPy runtime scores, tens of
milliseconds, and the panel says so). Drawing costs matplotlib a few tens of
milliseconds per frame; the latency numbers exclude it.

## Using fastdet inside a host that already runs imfeat (framegate)

framegate computes the same 1024-px HSV thumbnail and imfeat pyramid for its own
signals, so fastdet must not pay for a second pass. `Detector.front_end_spec` says
exactly what the model was trained on, and `Detector.predict_from_imfeat` scores an
imfeat result the host already has:

```python
det = Detector.load("model.fdt")
spec = det.front_end_spec
# {'thumb': 1024, 'stride': 1, 'resize_interp': 'area', 'space': 'hsv',
#  'levels': [64, 32, 16, 8, 4, 2], 'raw_channels_per_level': 162, 'extra_scales': []}

# host side, once: the FeatureComputer for that spec (imfeat appends its whole-image level)
fc = imfeat.FeatureComputer(shape=(spec["thumb"], spec["thumb"], 3),
                            grid=[(int(np.log2(n)),) * 2 for n in spec["levels"]],
                            stride=spec["stride"], threads=1)
# per frame: the host's own resize (INTER_AREA to thumb x thumb), conversion to spec["space"], imfeat
result = fc.features(hsv_thumbnail)
probabilities = det.predict_from_imfeat(result, image_hw=frame.shape[:2])   # (64, 64)
```

The result must come from a thumbnail and computer matching the spec (size, stride,
colour space, levels; the map shapes are checked); `image_hw` is the original frame's
height and width, which the global feature block records. A model trained with
`extra_scales` needs those passes too: run a computer per `(thumb, stride)` in
`spec["extra_scales"]` on downsized copies of the thumbnail and pass their results as
`extra_results`. The map is the same as `predict_proba(frame)`, through the same
scorer (C++ when the library is built). `FeatureExtractor.run_imfeat` / `.compose` are
the two halves fastdet itself uses, if a host wants to share at a different point.

## How it works

### Features

Each image is squashed to a 1024×1024 thumbnail (`INTER_AREA`; the aspect ratio
is kept as a global feature instead), converted to HSV, and passed to `imfeat`
once at stride 1 with six pyramid levels (64/32/16/8/4/2 cells) — framegate's
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

### Two measurement scales (`extra_scales`)

Every pyramid level is a bigger *window* over the same per-pixel measurements: imfeat's
3x3 kernels run once, at the thumbnail's pixel scale. `extra_scales` adds passes on
downsized copies of the thumbnail (`"256:1"` = a 256-px copy at stride 1), whose kernels
therefore see the image one or two octaves coarser; their 162 raw columns per level are
tagged `64@256px/s1/...` and sit beside the primary ones, so a tree can split on the
fine-scale and the coarse-scale version of the same statistic. On the OCR dataset it is
worth +0.01 to +0.02 PR-AUC on top of stride 1, at the cost of one more imfeat pass per
frame and a host that must compute that pass too (see `front_end_spec`). Off by default
for that reason.

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
| `iterations` | 1000 | number of trees (the first 667 tile-constant, see tiering) |
| `depth` | 5 | levels per tree, 32 leaves |
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

- **depth 5** limits interaction order to five splits;
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
det = Detector(Config()).fit(images_dir, masks_dir)                 # 4-bit, QAT, tiered, exit
det = Detector(Config(model=ModelConfig(leaf_bits=8))).fit(...)     # 8-bit
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

## Training time and memory

The fit is dominated by CatBoost on the training matrix (`n_cells x n_features`
float32: 6.7 M cells x 1178 columns is 32 GiB). What the pipeline does about it, and
the knobs that matter:

* **The matrix is quantised once and released.** `fit()` builds one CatBoost `Pool`,
  quantises it to `border_count` bins (4 bits per value) and drops the float matrix;
  every stage and chunk of the fit reuses it. Peak memory is roughly the float
  matrix plus a quarter, for the moment the pool is built. `fit()` prints the
  matrix size; above 8 GiB it also prints the knobs below. A "CatBoost is using
  more CPU RAM than the limit" warning means the machine is swapping: shrink the
  matrix.
* **Fewer cells** (`TrainConfig`): `neg_pos_ratio=3` keeps three negatives per
  positive (about a quarter of the cells at a 7% positive rate), `max_train_cells`
  caps the total; both are sweepable (`fastdet-tune --neg-pos-ratio none,3,5`), and
  `scale_pos_weight` re-weights positives without discarding data.
* **GPU** (`ModelConfig.task_type="GPU"`): CatBoost trains on any CUDA GPU its wheel
  can see (Windows and Linux, no toolkit install), several times faster on millions
  of cells; the quantised pool must fit in GPU memory (4 bits per value: 6.7 M x 1178
  is ~4 GB, fine for a 6 GB card). The chunked quantisation-aware fit uploads the
  pool once per chunk of `leaf_chunk` trees, so a GPU pays off most for big
  matrices. Inference never uses the GPU.
* **Fewer fits**: the quantisation-aware fit is one chunked fit per 16 trees (63 for
  1000 trees). At 8-bit leaves the rounding is one decision in ~25,000, so `quantisation_aware=False`
  (one fit per stage) is the fast path; keep it on for 4-bit leaves, where it is
  worth a few PR-AUC points.
* `thread_count=-1` already uses every core; the front-end feature pass is a few
  milliseconds per image and is not where the time goes.

## Hyperparameter tuning (`fastdet-tune`)

`fastdet-tune` fits one detector per combination of the knobs you name, scores the
held-out split exactly as the shipped runtime does (quantised leaves, coarse tier,
early exit), and writes a report folder:

* `report.md` -- one row per combination (only the swept knobs appear as columns) with
  PR-AUC (primary), ROC-AUC, best F1 and its threshold, precision / recall / IoU at that
  threshold, fit time, model size and, when the C++ library is built, `predict_proba`
  latency; the framegate text heuristic as a baseline row; the best run's full config.
* `curves.png` -- precision-recall and ROC curves side by side, for the best runs and the baseline.
* `results.json`, `models/<label>-<key>.fdt` -- every run's config, metrics and exported model;
  the file name spells out the swept knobs (`n_trees=1000_stride=2-3f9a...fdt`) and the
  `model` column of the table names it.
  Re-running with the same `--out` resumes: finished combinations are skipped.

```
pip install -e ".[tune]"                     # matplotlib, for the charts
fastdet-tune --images data/images --masks data/masks --out tune_report \
    --n-trees 1000,2000 --leaf-bits 4,8 --coarse-fraction 0.5,0.667
```

That is 2 x 2 x 2 = 8 fits. Every knob takes a comma-separated list and the sweep is
their Cartesian product, so name few knobs at a time; a knob you do not name keeps its
default, which is the tuned production value. `--max-runs 2` smoke-tests a sweep,
`--report-only` rebuilds the report and charts from `results.json`, `--no-baseline`
skips the heuristic, `--native-lib path/to/fastdet_native.{dll,so,dylib}` (or the
`FASTDET_NATIVE_LIB` variable) fills the latency column from the in-process scorer.
Tuples take `/`: `--levels 64/32/16/8`. Runs that share front-end settings reuse the
extracted features and the perceptual-hash split cache, so only the first run of a
sweep pays for extraction.

Every field of `ModelConfig` and `TrainConfig` is a knob (`fastdet-tune --help` lists
them with their current defaults). The ones worth sweeping:

| knob | default | what it does |
|---|---|---|
| `--n-trees` | 1000 | boosting iterations; latency grows linearly, quality saturates (see the trends below) |
| `--depth` | 5 | tree depth; 7 varying splits is the scorer's limit |
| `--learning-rate` | 0.1 | shrinkage; lower needs more trees |
| `--border-count` | 15 | split candidates per feature (max 15); 7 makes binning ~40% cheaper at no measured cost |
| `--leaf-bits` | 4 | leaf code width, 4 or 8; 4 halves the fine-tree cost (quantisation-aware fit, no measured loss) |
| `--leaf-chunk` | 16 | trees per quantisation step and scorer pass (<= 17) |
| `--quantisation-aware` | true | chunked fit on the quantised running score; `false` = one fit per stage (fast path at 8-bit) |
| `--task-type` | AUTO | `GPU` when CatBoost sees a CUDA device, else `CPU`; force either (see Training time and memory) |
| `--scale-pos-weight` | none (1) | CatBoost class weight on positives; the alternative to subsampling that keeps every negative |
| `--coarse-fraction` | 0.667 | share of trees restricted to tile-constant features (evaluated once per tile); 0 disables tiering |
| `--coarse-max-side` | 16 | largest feature grid that counts as tile-constant |
| `--use-exit` | true | early exit on |
| `--exit-keep-prob` | 0.05 | cells ending at or above this probability are never stopped |
| `--exit-margin` | 2.0 | raw-score safety margin under the calibrated thresholds |
| `--exit-stage-fractions` | 0/0.125/0.25/0.5/0.75 | where in the fine tier the stages sit |
| `--thumb` | 1024 | square resize target the feature pyramid is computed on (framegate's) |
| `--stride` | 1 | imfeat sampling stride at that size: 1 = every pixel (16x16 samples per finest cell), 2, 4 |
| `--extra-scales` | none | extra imfeat passes on downsized copies of the thumbnail, `thumb:stride` pairs separated by `;` (`--extra-scales "256:1"`, `--extra-scales "512:2;256:1"`; commas separate sweep values); each adds 162 raw columns per level measured at a coarser scale and one more imfeat pass of latency |
| `--levels` | 64/32/16/8/4/2 | pyramid grids (finest must be 64) |
| `--feature-mode`, `--imfeat-space` | raw_plus_global_context_ext, hsv | feature banks and colour space |
| `--resize-interp` | area | thumbnail resize kernel, `area` or `nearest` |
| `--top-k-features` | 0 (all) | keep the top-k gain-ranked columns; 512 halves binning |
| `--gt-cell-thresh` | 0.10 | mask coverage at which a cell is a positive |
| `--val-frac`, `--split-seed`, `--max-hamming` | 0.15, 42, 8 | held-out share of near-duplicate groups, the split seed, the perceptual-hash distance that makes two images a group |
| `--neg-pos-ratio` | 5 | negatives kept per positive in the training sample (`none` keeps all); rebalances the classes by discarding negatives, which shifts the probability scale (use `thr*`) |
| `--max-train-cells` | 1,000,000 | cap on sampled training cells (RAM: ~4.4 GiB per million cells as float32 before quantisation) |

### Trends on an OCR dataset (1,925 images, 6.8% positive cells)

Recorded from the sweeps that chose the defaults above, so they can be re-checked on a
new dataset with the same commands. Every row is a d5 x 1000, 4-bit, 0.667-coarse model
unless the knob says otherwise; the baseline framegate heuristic is at PR-AUC 0.42–0.45.

| knob | values | validation PR-AUC | note |
|---|---|---|---|
| `--stride` (1024 px, 100k cells) | 4 / 2 / 1 | 0.722 / 0.753 / 0.762 | the largest single effect: more samples per cell. Stride 1 costs ~9 ms of imfeat per frame at 1024 px, stride 2 ~3 ms, stride 4 ~1.6 ms; 512 px at stride 2 matches 1024/2 (0.749) at the cost of 1024/4 |
| `--max-train-cells` | 100k / 1M | 0.76 / 0.77–0.78 | still rising; more data is the next lever |
| `--neg-pos-ratio` | none / 3 / 5 / larger | 0.690 / ≈ / 0.712 / drops mildly | 3 and 5 are equivalent; keeping every negative (none) is worse at a fixed cell budget |
| `--n-trees` | 1000 / 2000 | saturates | at 100k–1M cells 2000 trees is not better than 1000 |
| `--depth` | 3 / 5 / 7 | 5 ≈ 3 > 7 | the signal is low-order; depth 5 costs a third less than 7 in the scorer |
| `--leaf-bits` | 4 / 8 | equal | the quantisation-aware fit closes the gap |
| `--coarse-fraction` | 0.5 / 0.667 / 0.75 | equal | pick for latency |
| `--border-count`, `--leaf-chunk` | 7 / 10 / 15, 8 / 12 / 16 | within run-to-run noise (±0.01) | `border_count 7` is free speed |
| `--learning-rate` | 0.1 / 0.01 | 0.1 better at 1000 trees | 0.01 needs many more trees |
| `--extra-scales` (on top of the defaults) | `256:2` or `256:1` | +0.01 to +0.02 | a second, coarser measurement scale (see Front-end). Not the default: each extra scale is another imfeat pass on the frame; revisit when the front-end can afford it |

Two identical configurations fitted on the GPU differed by 0.011 PR-AUC, so differences
below about 0.01 in a table are noise; re-run a candidate pair before deciding on them.

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
