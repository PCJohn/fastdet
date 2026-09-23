# Fast CPU Text-Region Segmentation with Classical Features and Oblivious Trees

**Project:** `imseg` (training/eval) on top of `imfeat` (single-pass CPU feature extractor)
**Task:** per-cell text/no-text segmentation on a 64×64 grid, as an OCR region proposer
**Period:** Sep 2026
**Status at time of writing:** PR-AUC 0.810–0.829 depending on configuration; feature
extraction 8.2 ms/image.
**Shipped configuration (§7):** `d7 × 2400`, PR-AUC ≈ 0.8106 (mean of 3 seeds), model ≈ 5.7 ms
single-threaded — down from ~302 ms at the start of the inference work.
**Cheaper measured point:** `d7 × 1200` at **3.71 ms** (1.90 traversal + 1.81 binning). The
runtime ladder in §3.8 is reported at `d7 × 1200`; §7.1 records the trade between the two.

---

## 1. Introduction

### 1.1 Setting

`imfeat` is a C++ library that extracts classical computer-vision features in a **single pass
over image pixels**. Its design constraint is that every feature must be expressible as an
accumulator updated from a pixel and a small fixed neighbourhood:

```
accumulators = {}
for pixel in image:
    update_accumulators(pixel, neighbourhood)
final_post_processing(accumulators)
```

It computes intensity moments, structure tensor, orientation histograms, extrema density,
derived summary statistics, and perceptual hashes — per channel, per patch, at multiple patch
sizes, producing a feature pyramid per (channel, feature type). Reference latency from its own
test suite: 256×256×3, 4 levels, stride 2 → ~1.1 ms `features()`, ~2.4–2.7 ms including hashes.

`imseg` trains small models on those features for segmentation. The formulation, inspired by
Aggregated Channel Features (ACF), is:

- resize image to a square thumbnail, convert to a working colour space
- run `imfeat` at several pyramid levels (and, later, several image scales)
- for each cell of the finest 64×64 grid, concatenate features from every level into one
  long vector (plus image-global scalars and local context features)
- classify each of the 4096 cells independently

**Dataset:** 1925 (image, mask) pairs, hand-crafted OCR masks. Group-aware train/val split by
perceptual-hash near-duplicate clusters (leak-free): 1637 train / 288 val. ≈6.7 M cells,
7.29 % positive at `gt_cell_thresh = 0.1` (a cell is positive if ≥10 % of its pixels are text).

**Primary metric:** pooled PR-AUC over all val cells. Secondary: F1, IoU, MCC at the best-F1
threshold. **Baseline:** the `framegate` heuristic text detector, PR-AUC 0.4211 on the same
val split.

### 1.2 Goals

1. Improve detection quality without regressing `imfeat` latency.
2. Identify which *new* features are worth adding to `imfeat` as new accumulators (i.e. still
   single-pass legal — local stencils only, no whole-image statistics, no long delay lines).
3. Reach deployable inference latency on CPU, single-threaded.

### 1.3 Methodology notes (these mattered a lot)

- **Paired image-level bootstrap CI** on ΔPR-AUC for every comparison. Screening-scale CI
  half-widths are ≈ ±0.013; full-scale ≈ ±0.005. Any effect below the CI width is not a result.
- **`split_seed` and `seed` must be separate.** Conflating them once produced a phantom
  ±0.033 swing (the val split was moving, not the model).
- **Keep rule:** keep an addition only if the CI95 lower bound > 0; otherwise prefer the
  cheaper configuration.
- **Latency rule:** all timings single-threaded, warm, p50 over ≥100 reps, same fixture.
  Node-visit counts are reported alongside milliseconds because ms are machine-specific.

---

## 2. Summary of directions tried

### 2.1 Feature / front-end work

| Direction | Outcome |
|---|---|
| **INTER_NEAREST → INTER_AREA thumbnail resize** | **WORKED, biggest single win (~+0.08)**. Nearest-neighbour was aliasing away ~36 % of rows; thumb-scale strokes are only 1–2 px wide. |
| **HSV/V → CIELAB, banks on L\*** | **WORKED (~+0.03)**. V = max(R,G,B) is a poor luminance for text; saturated coloured text on white is near-invisible in V. |
| **Multi-scale: drop 1024 px, use 512/s2 + 256/s1** | **WORKED (~+0.08 and ~40 % faster)**. Pyramid *levels* pool; they do not change gradient scale. Adding true image scales does. |
| **`bard` bar-detector bank (new, single-pass legal)** | **WORKED (+0.0103 @1M, 3/3 seeds CI-clear)**. Multi-lag two-sided saturated differences; 7 pooled stats/level. |
| Exact SWT (Stroke Width Transform) | Strong in the old pipeline (+0.026) but **not portable** — ray marching, not single-pass. |
| `swtp` single-pass SWT proxy | **DID NOT WORK** on top of `bard` (+0.0002, CI straddles 0), and not actually single-pass legal (needs a 97th-percentile whole-image gate, 17×17 opening, σ=8 Gaussian). |
| `bard` at extra scales / on a,b chroma channels | **DID NOT WORK** (+0.0029, +0.0018; CIs straddle 0). |
| `prof` profile-anisotropy bank (row-mean vs col-mean variance) | **DID NOT WORK** (−0.0047). Redundant with the structure tensor / orientation histogram already in `imfeat`. |
| Feature enrichment: squares, cubes, signed roots, log1p, gamma, pairwise products/ratios, curated interactions | **DID NOT WORK** (−0.005 to −0.050). Monotone per-feature transforms are information-free for threshold-split trees. |
| `neighbor_delta` (cell minus 4-neighbour mean) | **DID NOT WORK** (−0.0034). |
| Auto-context v1 (5 stencils over stage-1 score map) | **DID NOT WORK** meaningfully (+0.003–0.004 at ~7× training cost). |
| Hyperparameter tuning of the GBDT | **DID NOT WORK** (0.7514 → 0.7512, flat). The ceiling was the feature set. |

### 2.2 Model / inference work

| Direction | Outcome |
|---|---|
| **Feature pruning to top-512 by split gain** | **WORKED** (PR-AUC tie, −49 % features, −44 % predict time). |
| **Compute-budgeted retraining (fewer/shallower trees)** | **WORKED as a lever**, at a real cost: 1926 → 400 trees cost −0.0175 PR-AUC. |
| **Own C++ tree blob + scorer** | **WORKED**. Self-contained format, no dependencies, bit-exact to sklearn (≤6e-7). |
| **uint8 feature quantization (byte features)** | **WORKED, but only after the binner was fixed.** Naive binning cost more than the traversal saved. |
| **Reduced-cut binning** (only thresholds actually referenced by nodes) | **WORKED, decisive**: binning 17.5 ms → 1.4 ms. Trees used a median of 12 of 255 bins. |
| **Fusing assembly + binning over coarse cells** | **WORKED**: 7.9 ms assembly + binning collapsed to 3.6 ms. The design matrix is 4.27× redundant (coarse-level columns repeat). |
| **Oblivious / symmetric trees (CatBoost) + AVX2** | **WORKED, decisive**: 15.2 → 5.97 ms traversal at quality parity within noise. |
| **4-bit features + `vpshufb` threshold tables** | **WORKED, decisive**: the shuffle *is* the comparison. Traversal → 2.40 ms, then 1.90 ms nibble-packed. |
| **Nibble-packed matrix + fused packed binner** | **WORKED**: 1.90 ms traversal + 1.81 ms binning = **3.71 ms model total**. |
| Soft per-cell cascade (early rejection over trees) | **Partially worked** (~1.4× real on the fast path), rendered unnecessary by the above. |
| Coarse-to-fine pyramid cascade (reject level-8 blocks) | **Mostly DID NOT WORK**: only ~34 % of cells rejected, far below the mask-derived ceiling; oversizing coarse stages broke recall (93.8 %, 973 orphans). |
| Exact bound-based early exit | **DID NOT WORK — structurally**. Bound = Σ max\|leaf\| over remaining trees ≈ tens of logits vs an O(1) decision margin. 13.6 % of trees skipped, no wall-clock win. |
| K-lane scalar interleaved traversal | **DID NOT WORK**. Out-of-order execution already overlapped the independent per-cell walks; forcing fixed depth added 35 % more probes. |
| `max_leaf_nodes` 128 → 63 | **DID NOT WORK**: early stopping bought the capacity back with more trees (1926 → 3738), a 1.6× *increase* in node visits for a quality tie. |
| Non-tree model families (logistic regression, RF/ExtraTrees) | **DID NOT WORK** as replacements (0.59, 0.69–0.73) but useful as frontier anchors. |

### 2.3 High-level takeaways

1. **Front-end bugs dominated everything.** ~95 % of the total quality improvement
   (0.661 → 0.829) came from fixing interpolation, colour space, and image scale — not from
   new features. The single new feature bank contributed +0.0103 of a ~+0.17 total.
   *Lesson: audit the data path before designing features.*
2. **Every negative result predating a front-end fix is void.** The entire "enrichment does
   not help" table had to be re-tested on the corrected front-end; some results changed sign.
3. **Feature engineering hit a hard plateau.** After the front-end fixes, nine separate
   feature/interaction ideas were tested and eight were flat or negative. The 540→996-column
   feature set is at its ceiling for this label set.
4. **Latency was in the model, not the features — by 30×.** The whole project was framed
   around "no `imfeat` latency regression" while the model cost ~302 ms against `imfeat`'s
   8.2 ms. *Lesson: measure the whole pipeline before choosing a constraint.*
5. **Micro-optimisation of the traversal loop was exhausted quickly (~1.2 ns/probe,
   latency-bound); architectural change was worth 8×.** Oblivious trees + nibble features +
   shuffle tables beat every attempt to make the leaf-wise walk faster.
6. **Approximation was never needed.** Soft cascades and coarse-to-fine pyramids cost
   quality, needed threshold calibration and recall budgets, and were made redundant by exact
   changes (quantization, symmetric trees, SIMD).
7. **A straight-line scan beats a bisection on short arrays**, twice over: reduced-cut binning
   (4.68×) and the traversal. Dependent-load chains are the enemy on modern cores.

---

## 3. Detailed logs

### 3.1 Front-end repair (screening scale: 160 images, 24 val, 100k cells)

Starting point (the inherited pipeline): image → 1024×1024 **INTER_NEAREST** → HSV → `imfeat`
38-D/block → per-cell rows; banks computed on the **V** channel.

| front-end config | `raw_plus_global` | `+swtp_context_ext` |
|---|---|---|
| default (nearest / V / HSV) | 0.6414 | 0.6611 |
| `--resize_interp area` | 0.6978 | 0.7400 |
| area + `--imfeat_space lab` | 0.7459 | 0.7641 |
| area + lab + `--bank_gray y` | 0.7459 | 0.7738 |
| area + `--thumb 512 --stride 2` | 0.7143 | 0.7383 |
| area + `--extra_scales 512:2;256:1` | 0.7789 | 0.7816 |
| area + extras + **bard** | 0.7979 | **0.8030** |
| area + extras + auto-context (3 folds) | 0.7819 | 0.7860 |

**Diagnostics that motivated each fix:**

- *Interpolation*: at 1024², mask-derived stroke widths were p50 = 1.9 px, p90 = 3.8 px.
  Nearest-neighbour resampling of a 900×1600 source drops ~36 % of rows, so horizontal strokes
  could vanish entirely before `imfeat` saw them.
- *Colour space*: on a sample image, text-vs-background contrast for red logo text was 0.31 in
  V versus 0.48 in L\*. Hue wraps (logo hue ≈ 174/180) and is noise at low saturation.
- *Scale*: `imfeat`'s pyramid levels pool a fixed 3×3 stencil; they never change gradient
  scale. ACF's gains come substantially from channels computed at multiple *image* scales.

### 3.2 Ablation suite with CIs (screening, 24 val images, paired bootstrap)

| run | base | mode | PR-AUC | `imfeat` ms | vs | ΔPR-AUC CI95 | keep |
|---|---|---|---|---|---|---|---|
| A1 | 1024/s4 + 512:2;256:1 | raw+global | 0.7981 | 13.28 | anchor | — | — |
| A2 | A1 | +bard(L) ctx | 0.8195 | 13.30 | A1 | +0.0214 [+0.004, +0.065] | **✓** |
| A3 | A2 | +swtp | 0.8197 | 13.62 | A2 | +0.0002 [−0.008, +0.006] | ✗ |
| A4 | A2 | bard@extras | 0.8225 | 13.17 | A2 | +0.0029 [−0.006, +0.008] | ✗ |
| A5 | A4 | bard all channels | 0.8243 | 13.10 | A4 | +0.0018 [−0.004, +0.009] | ✗ |
| B1 | 512/s2 + 256:1 | raw+global | 0.7906 | 8.68 | A1 | +0.0075 [−0.007, +0.043] | ≈A1 |
| **B2** | B1 | **+bard(L) ctx** | **0.8082** | **8.41** | B1 | +0.0176 [+0.006, +0.045] | **✓** |

B2 vs A2: +0.0114 [−0.0007, +0.047] — a tie, so the cheaper (B2) wins.

### 3.3 Full-scale validation (1M cells, full 1925-image dataset, 288 val images)

With `split_seed` fixed at 42 and only the cell/model seed varied:

| seed | A1 (raw, 13.3 ms) | B2 (+bard, 8.0 ms) | ΔPR-AUC (B2−A1) | CI95 |
|---|---|---|---|---|
| 42 | 0.8194 | 0.8293 | +0.0100 | [+0.0016, +0.0190] |
| 43 | 0.8190 | 0.8290 | +0.0100 | [+0.0020, +0.0184] |
| 44 | 0.8195 | 0.8305 | +0.0110 | [+0.0043, +0.0185] |
| **mean** | 0.8193 | **0.8296** | **+0.0103** | 3/3 clear of zero |

Lag ablation: `bard_lags` (1,2,4) vs (1,2,4,8) at 1M → −0.0012, CI [−0.0042, +0.0015] → tie,
freeze (1,2,4) (lag-8 spectrum column had zero gain importance).

### 3.4 The frozen feature configuration ("B2")

```
thumb 512, stride 2, extra_scales "256:1", resize_interp area,
imfeat_space lab, bank_gray lstar, bard_channels lum, bard_lags 1,2,4,
levels/combos 64,32,16,8, mode raw_plus_global_bard_context_ext
```

996 columns = 912 raw (4 levels × 114 × 2 scales) + 28 global (7 × 4) + 28 bard (7 × 4)
+ 20 context (5 × 4) + 8 ctx2 (2 × 4). `imfeat` cost 8.2 ms/image single-threaded.

**Superseded (Sep 2026).** Two changes since this section was written. (1) `bard` now
ships inside imfeat's raw block (7 columns per channel, verified to 2.4e-7 against the
numpy bank it replaced), so fastdet computes no image features itself; imfeat also grew a
9-column texture block, for 54 columns per channel. (2) The front-end was switched to
framegate's single imfeat pass — 1024 px square, HSV, stride 4, 64×64 finest grid, six
levels (64…2), no extra scale — so a framegate process can feed fastdet directly. That
front-end is 1178 columns: 972 raw (6 levels × 162), 30 context + 12 ctx2 (6 levels ×
5 and × 2; the coarsest levels are zero where the kernels do not fit), and a 164-wide
global block carried once at the finest level. Note that it reintroduces HSV, which §3.1
measured at ~−0.03 against CIELAB on the old pipeline. No model has been trained on it
yet: every PR-AUC and latency figure in this report applies to earlier front-ends, and the
bundled feature ranking must be regenerated before pruning.

**Top split-gain importances** (500k-cell run, 0.8234 PR-AUC, F1 0.7443, IoU 0.5927,
MCC 0.7272 vs framegate 0.4211 / 0.4285 / 0.2727 / 0.3847):

| rank | feature | gain |
|---|---|---|
| 1 | `64/L/energy` | 22.98 % |
| 2 | `64/L/corner` | 13.38 % |
| 3 | `64@256px/s1/L/var` | 4.99 % |
| 4 | `64/L/var` | 2.51 % |
| 5 | `8/L/std_skew` | 1.80 % |
| 6 | `8/bard/bardL_cover` | 1.46 % |

Top-60 cumulative: 70.5 %.

### 3.5 The `bard` bank (the one new feature worth porting)

For each sampled pixel *c*, lag δ ∈ {1,2,4}, direction *u* ∈ {x,y}, with L and R the pixels at
*p* ∓ δ*u*:

```
dark_δ  = min(sat(L − c), sat(R − c))      # both sides brighter
light_δ = min(sat(c − L), sat(c − R))      # both sides darker
```

Take the max over direction. A pixel contributes only if its peak response over lags and
polarities ≥ τ (τ = 8 grey levels). Accumulate per cell: a gated count, three per-lag sums,
and dark/light maxima → 7 pooled outputs (`bardL_cover`, `spec1..3`, `peak`, `peaked`, `bal`).

Why it is single-pass legal:

- pure uint8 saturated arithmetic (`SaturatedSub`, `Min`, `Max`)
- reads rows *r* ± δ directly from the in-memory image — **no delay line**
- no whole-image statistic — exactly additive, bit-exact under threading

Step edges give zero response (both sides must be brighter, or both darker), so it pairs
opposite edges the way SWT does, but as a local stencil. Response is cumulative in δ: it fires
once δ > w/2, so stroke width appears as the first-difference peak.

### 3.6 Error analysis (unresolved — see §5)

At three operating points on the frozen model (288 val images):

| precision floor | threshold | recall | FP mass at `coverage == 0` | FP isolated | top-20 images' FP share |
|---|---|---|---|---|---|
| 0.5 | 0.026 | 0.884 | 87.7 % | 14.1 % | 28.1 % |
| 0.8 | 0.307 | 0.706 | 80.8 % | 23.1 % | 42.8 % |
| 0.9 | 0.694 | 0.577 | 78.7 % | 27.4 % | 53.6 % |

FP cell mass by connected-blob size at P≥0.8: 18.7 / 26.0 / 27.9 / 11.7 / 11.9 / 3.8 %
(sizes 1 / 2–5 / 6–20 / 21–50 / 51–200 / >200).

Calibration (text-bearing images only, 241/288): pooled PR-AUC 0.8425, mean per-image AP
0.7434, rank-normalised pooled 0.5394. Cross-image score levels carry real signal; there is no
cheap per-image calibration fix. **Note the 0.10 gap between pooled and per-image AP — if the
detector is thresholded per image in deployment, 0.74 is the honest number.**

**The open problem:** 80–88 % of false-positive mass sits in cells whose mask has *zero* text
pixels, yet visual overlays show that mass landing on clearly legible text — engineering part
labels, map place names, "Happy Diwali", a stadium scoreboard. Either the hand-crafted masks
are incomplete, or it is a one-cell halo (the mask covers glyph pixels exactly; an 8 px cell
plus a 3×3 context bank still sees stroke evidence one cell over). **This is an evaluation
defect, not necessarily a model defect, and it is unresolved.**

### 3.7 Model compression

| step | n_trees | mean depth | node visits/cell | PR-AUC | ΔPR-AUC CI95 |
|---|---|---|---|---|---|
| full 996 features | 1716 | — | — | 0.8281 | reference |
| top-512 by gain | 1926 (converged) | 9.08 | 17 485 | 0.8285 | +0.0005 [−0.0020, +0.0032] |
| `lr0.1 × 400 trees × 31 leaves` | 400 | 6.29 | 2 516 | 0.8110 | −0.0175 [−0.023, −0.013] |
| `lr0.2 × 400 × 31` | 400 | 5.43 | 2 171 | 0.8088 | −0.020 [−0.026, −0.014] |
| `lr0.2 × 100 × 31` | 100 | 6.83 | 683 | 0.7926 | −0.036 [−0.044, −0.029] |
| `max_leaf_nodes 63` | 3738 | ~6 | 20 994 | tie | 1.6× *more* visits — rejected |

Frontier anchors: logistic regression 0.59; ExtraTrees/RF 0.69–0.73. Neither is competitive.

**Two structural findings:** 0 of the 512 selected features are unused by the trees (so column
pruning cannot shrink `imfeat`'s work); and the trees reference a median of 12 distinct
thresholds per feature out of 255 bins (~7 % utilisation) — this observation later enabled
both the reduced-cut binner and 4-bit quantization.

### 3.8 Inference engineering (the 302 ms → 3.71 ms path)

**Own blob format and scorer.** sklearn's `_predictors[i][0].nodes` → flat node array,
tree-relative child indices, `baseline_prediction` offset, logit-space threshold. Verified
bit-exact against sklearn (max |Δprob| = 5.96e-7; 0 threshold crossings over all 288 images).
One real bug found: node `left`/`right` are tree-native, so subtracting the blob tree offset
corrupted every tree after the first.

**Traversal history** (4096 cells; the 400-tree model unless noted):

| variant | ms | ns/probe | note |
|---|---|---|---|
| sklearn, 22 threads | 202 | — | OpenMP; not comparable to 1-thread rows |
| sklearn, 1 thread | 3773 | — | contaminated measurement, quoted only for scale |
| C++ cell-major, float | 670 | 84.9 | 1926-tree model |
| C++ tree-major B=64 | 242 | 30.7 | 1926-tree model |
| C++ tree-major colmaj B=512 | 179 | 22.7 | 1926-tree model |
| C++ compact8 colmaj B=512 | 169 | 21.4 | 1926-tree model |
| float compact8 colmaj | 18.0 | 1.43 | 400-tree model |
| byte4 colmaj B=2048 | 15.2 | 1.20 | 400-tree model |
| symmetric scalar | 21.1 | 0.92 | CatBoost d7×800 |
| symmetric AVX2 no-gather | 5.83 | 0.254 | d7×800 |
| symmetric AVX2 gather-leaf | 4.88 | 0.213 | d7×800 |
| symmetric d7×1200, block 1024 | 5.97 | 0.173 | config used for the runtime ladder below |
| + 4-bit `vpshufb` tables | 2.40 | — | `border_count=15` |
| + nibble-packed | **1.90** | — | floor with leaves stubbed: 0.587 ms |

**Assembly and binning history:**

| stage | ms |
|---|---|
| transpose X → colmaj | 9.19 |
| coarse → colmaj expand (assembly) | 7.29–7.91 |
| binning, naive per-value bisection | 39 |
| binning, "SIMD" full-cut scan (O(k) per value — *worse* than bisection) | 17.5–17.8 |
| binning, coarse-only P2 | 7.8–8.3 |
| binning, V-lane interleaved P2 | 4.4 |
| binning, **reduced cuts + SIMD compare-count** | 1.4–1.5 |
| **fused reduced-on-coarse (replaces assembly AND binning)** | **3.60** |
| symmetric, row-outer coarse binner | 2.49 |
| symmetric, `border_count=15` + fused packed | **1.81** |

The design matrix is 4.27× redundant: of 512 columns, 215 are level-8 (64 distinct values
across 4096 cells), 123 level-16, 87 level-32, 87 level-64. Binning per *unique coarse cell*
is 490 688 operations instead of 2 097 152.

**Oblivious (symmetric) trees.** Every node at depth *d* uses the same (feature, threshold), so
a depth-*D* tree is *D* independent comparisons producing a *D*-bit index into a 2^D leaf table.
No dependency chain, no node array, vectorises trivially across cells. CatBoost trains them;
the blob (`IMSY`) and runtime remain first-party. Root split is the **lowest** leaf-index bit.

Extended quality sweep (1M cells, `split_seed` 42, vs HGB reference 0.8110):

| config | PR-AUC | ΔPR | CI95 hi | predicted traversal | verdict |
|---|---|---|---|---|---|
| `d6 × 3200 × lr0.05` | 0.8129 | +0.0018 | +0.0090 | 16.8 ms | over gate |
| `d7 × 3200 × lr0.05` | 0.8127 | +0.0016 | +0.0088 | 19.5 ms | over gate |
| `d7 × 2400 × lr0.1` | 0.8118 | +0.0008 | +0.0077 | 14.7 ms | ties |
| `d7 × 1600 × lr0.1` | 0.8104 | −0.0006 | +0.0057 | 9.77 ms | ties |
| **`d7 × 1200 × lr0.1` (shipped)** | **0.8102** | **−0.0008** | +0.0047 | **7.33 ms** | tie, cheapest |
| `d6 × 1200 × lr0.1` | 0.8042 | −0.0069 | +0.0003 | 6.28 ms | knife-edge |
| `d7 × 1200 × lr0.05` | 0.8032 | −0.0079 | −0.0021 | 7.33 ms | worse |

Seed replication of the shipped config (`split_seed` 42 fixed, model seed varied):

| seed | CatBoost | HGB | Δ (cb − hgb) | CI95 |
|---|---|---|---|---|
| 42 | 0.8102 | 0.8110 | −0.0008 | [−0.0068, +0.0047] |
| 43 | 0.8041 | 0.8105 | −0.0064 | [−0.0154, +0.0019] |
| 44 | 0.8076 | 0.8074 | +0.0002 | [−0.0073, +0.0067] |

**Mean Δ ≈ −0.0023; each CI contains zero but at n = 3 with ±0.007 half-widths the test cannot
distinguish −0.005 from 0. The honest statement is "tie within measurement noise", not "ties".**
Note the HGB reference itself moves 0.003 between seeds.

**4-bit features + `vpshufb`.** With `border_count = 15`, each feature fits in a nibble. For
each split, precompute a 16-byte table `T[v] = (v > bin) ? (1<<d) : 0`; the depth step becomes

```cpp
idx = _mm256_or_si256(idx, _mm256_shuffle_epi8(T, vals));   // 32 cells, 2 uops
```

The shuffle *is* the threshold test — no compare, no mask, no shift. Contrast the original
int32 path: `loadl_epi64 → cvtepu8_epi32 → cmpgt_epi32 → slli → and → or` = 6 uops for 8 cells.

Final lossless ladder (single-thread, idle machine):

| path | traversal | binner | model total |
|---|---|---|---|
| u8-lane + gather-leaf | 2.88 | 2.10 | 4.98 |
| shuffle tables + gather-leaf | 2.40 | 2.10 | 4.50 |
| **nibble-packed + fused packed binner** | **1.90** | **1.81** | **3.71** |

Traversal floor with leaf accumulation stubbed out: 0.587 ms — i.e. ~1.3 ms of the 1.90 ms is
now leaf lookup, one per (cell, tree) = 4.9 M lookups.

### 3.9 Approximation methods (built, measured, ultimately unnecessary)

**Soft per-cell cascade.** Stage boundaries over trees; each stage's reject threshold set on
*training* scores to keep 99.9 % of cells the full model puts above the operating threshold.
Shipped config: stages 20/50/100/200, q=0.001, rejects 80.4 %, visits/cell 2516 → 1080,
PR-AUC 0.8110 → 0.8106. Gains ~2.3× on the slow cell-major path, ~1.4× on the fast path.
Stage 1 (20 trees) rejected 0 % — pure overhead.

**Coarse-to-fine pyramid.** Reject level-8/16/32 blocks before descending, so one rejection
kills 64 finest cells. OR-pooled coarse labels (coverage-pooling was strictly worse: recall
collapsed to 80–90 %).

| chain | visits/cell | est. ms | PR-AUC | fine recall | orphans |
|---|---|---|---|---|---|
| per-cell cascade alone | 1080 | 7.96 | 0.8106 | — | — |
| 16-chain (100×31) | 858 | 6.33 | 0.8106 | 99.9 % | 0 |
| 8→16→32 (100×31) | 880 | 6.49 | 0.8100 | 99.5 % | 3 |
| 8→16→32 oversize (400/200 × 63/31) | 590 | 4.35 | 0.7923 | 93.8 % | 973 |

Only ~34 % of cells were rejected by the coarse chain — well below the mask-derived ceiling —
so ~2700 of 4096 cells still paid the full fine-model cost. Oversizing the coarse stages was
the only way to cut further, and it broke recall.

---

## 4. Summary charts

### 4.1 Quality progression (PR-AUC)

```
framegate heuristic        ████████▍                                  0.421
broken front-end (nearest/V)  ████████████▊                           0.641
  + INTER_AREA             ██████████████                             0.698
  + Lab / L* banks         ███████████████                            0.746
  + multi-scale extras     ████████████████                           0.779
  + bard bank              ████████████████▍                          0.798
B2 frozen, 1M cells        █████████████████                          0.830
budget-swept 400-tree HGB  ████████████████▋                          0.811
shipped symmetric d7x1200  ████████████████▋                          0.810
                           └────┴────┴────┴────┴────┴────┴────┴────┘
                          0.40 0.45 0.50 0.55 0.60 ... 0.80 0.85
```

Most of the distance from 0.42 to 0.83 is front-end repair. The last two rows are the
deliberate quality-for-latency trade (−0.0175 budget sweep, −0.002 symmetric).

### 4.2 Model latency reduction (ms/image, single-thread, 4096 cells)

```
sklearn predict_proba (1716 trees)   ██████████████████████████████  301.6
  + top-512 feature pruning          █████████████████▍              174.2
  + budget sweep (400 trees)         ███▊                             38.1   [cell-major]
  + colmaj / compact8 float          █▊                               18.0   (+7.9 assembly)
  + byte4 quantized traversal        █▌                               15.2   (+3.6 fused)
  + symmetric trees (AVX2)           ▌                                 5.97  (+2.49 bin)
  + 4-bit shuffle tables             ▏                                 2.40  (+2.10 bin)
  + nibble-packed + fused binner     ▏                                 1.90  (+1.81 bin)
                                     MODEL TOTAL: 301.6 → 3.71 ms  (81x)
```

### 4.3 Pipeline composition

```
BEFORE (float HGB, 1926 trees)          AFTER (symmetric, nibble)
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ imfeat          8.2 ms   2.6%│        │ imfeat          8.2 ms  68.9%│
│ assembly        7.9 ms   2.5%│        │ fused asm+bin   1.81 ms 15.2%│
│ model         301.6 ms  94.9%│        │ traversal       1.90 ms 16.0%│
├──────────────────────────────┤        ├──────────────────────────────┤
│ TOTAL         317.7 ms       │        │ TOTAL          11.91 ms      │
└──────────────────────────────┘        └──────────────────────────────┘
```

`imfeat`, which the entire project was originally constrained around, is now 69 % of the cost
and has never been optimised in this campaign.

### 4.4 Quality-vs-latency frontier (model ms, single-thread)

```
PR-AUC
0.830 ┤                                                    ● HGB 1926 trees (302 ms)
0.825 ┤
0.820 ┤
0.815 ┤
0.810 ┤ ● symmetric d7x1200 (3.71)   ● HGB 400 trees (18.9)
0.805 ┤   ● pyramid chain (est ~6)
0.800 ┤
0.795 ┤ ● HGB 100 trees (~7)
      └──┴──────┴──────┴──────┴──────┴──────────────────────────────
         3      6     10     20     40  ...  300   model ms (log-ish)
```

The shipped point (0.810 @ 3.71 ms) is ~81× cheaper than the best-quality point for
−0.019 PR-AUC, of which −0.0175 is the budget sweep and ~−0.002 the symmetric switch.

---

## 5. Explicit worked / did-not-work register

### Worked — keep

| Change | Measured effect | Where it lives |
|---|---|---|
| INTER_AREA resize | ~+0.08 PR-AUC | front-end config |
| Lab colour space, L\* banks | ~+0.03 | front-end config |
| 512/s2 + 256/s1 (drop 1024) | ~+0.08, −40 % `imfeat` ms | front-end config |
| `bard` bank, lum, lags (1,2,4) | +0.0103 @1M, 3/3 seeds | new `imfeat` accumulators |
| Group-aware near-duplicate split | leak-free eval | training harness |
| Paired image-level bootstrap CI | made every later decision honest | eval harness |
| `split_seed` ≠ `seed` | killed a phantom ±0.033 | eval harness |
| Top-512 gain pruning | PR tie, −44 % predict | training |
| Budget-swept 400-tree model | −0.0175 for ~8× | training |
| Own C++ blob + scorer | bit-exact, dependency-free | export + runtime |
| Reduced-cut binning | 17.5 → 1.4 ms | runtime |
| Coarse-cell fused assembly+binning | 7.9 + bin → 3.6 ms | runtime |
| Oblivious trees + AVX2 | 15.2 → 5.97 ms | model family + runtime |
| `border_count=15` + `vpshufb` tables | 5.97 → 2.40 ms | quantization + runtime |
| Nibble packing + fused packed binner | → 1.90 + 1.81 ms | runtime |

### Did not work — do not retry without new information

| Attempt | Measured | Why |
|---|---|---|
| `swtp` proxy SWT on top of bard | +0.0002 [−0.008, +0.006] | redundant with bard; also not single-pass legal |
| bard at extra scales | +0.0029, CI straddles 0 | no extra information |
| bard on a/b chroma | +0.0018, CI straddles 0 | text contrast is luminance-dominated here |
| `prof` anisotropy bank | −0.0047 | duplicates structure tensor + orientation histogram |
| squares/cubes/roots/log1p/gamma | 0.0 to −0.005 | monotone transforms are free for threshold splits |
| pairwise products / ratios | −0.023 / −0.014 | greedy gain favourites do not generalise |
| curated interactions | −0.050 (old FE), ≈0 (new FE) | displaces equally useful base splits |
| `neighbor_delta` | −0.0034 | context banks already cover it |
| auto-context v1 | +0.003 at 7× training cost | 5 stencils, 4 of them half-cell shifted; not a line detector |
| GBDT hyperparameter tuning | 0.7514 → 0.7512 | ceiling is the feature set |
| `max_leaf_nodes` 63 | 1.6× more visits, quality tie | early stopping restores capacity via more trees |
| Logistic regression / RF / ExtraTrees | 0.59 / 0.69–0.73 | not competitive |
| Exact bound-based early exit | 13.6 % trees, no ms win | bound ≫ decision margin for GBDT |
| K-lane scalar interleaved traversal | 23.2 vs 15.2 ms | OoO already overlapped; fixed depth added 35 % probes |
| "SIMD" O(k) full-cut binning | 17.5 ms | asymptotically worse than bisection |
| Coverage-pooled coarse labels | recall 80–90 % | OR-pooling is correct for a rejection cascade |
| Coarse-to-fine pyramid | ~34 % rejection; oversize broke recall | coarse stages could not reject enough |
| Referenced-only border remap (symmetric) | no-op | all 6010 borders were already referenced |

### Open / unresolved

1. **The `coverage == 0` false-positive question (§3.6).** Blocks all further quality work.
   The audit that would settle it: split FP mass into (i) a 1-cell dilation band around GT,
   (ii) cells with 0 < coverage < threshold, (iii) genuine far-FP; then hand-label ~100
   sampled far-FP crops as text / line-work / texture / nothing.
2. **`imfeat` itself (8.2 ms, 69 % of the pipeline)** has never been profiled or optimised.
3. **Deployment metric.** Pooled PR-AUC over images of wildly different text density is a weak
   proxy for a region proposer. Recall at a fixed FP-cell budget per image, scored with an
   ignore band, is probably the right target.
4. **Leaf lookup is the new traversal floor** (~1.3 of 1.90 ms). Int16 leaf quantization with
   integer accumulation, or fewer trees, are the remaining levers.
5. ~~`d7 × 2400` vs `d7 × 1200`~~ — **decided**, see §7: `d7 × 2400` ships. The quality
   difference between the two at `border_count 15` was never measured head to head, so the
   size of the trade is recorded as unverified in §7.1.

---

## 6. Reproduction quick reference

**Frozen front-end** (defaults in `train_classical.py`):
`--thumb 512 --stride 2 --extra_scales "256:1" --resize_interp area --imfeat_space lab
--bank_gray lstar --bard_channels lum --bard_lags 1,2,4 --combos 64,32,16,8`
mode `raw_plus_global_bard_context_ext`, `gt_cell_thresh 0.1`, group-aware split,
`split_seed 42`.

**Model:** CatBoost `SymmetricTree`, Logloss, depth 7, 2400 trees (shipped; 1200 was the earlier
selection), lr 0.1, `border_count 15`,
`l2_leaf_reg 3`, `rsm 1.0`, `boosting Plain`, `bootstrap MVS 0.8`, trained on the top-512
gain-ranked columns of the 996-column matrix, 1M sampled cells.

**Runtime:** `IMSY` blob (magic `IMSY`, version 3) — `n_trees`, `n_features`, `depth`,
`tree_offsets`, `tree_base`, split records `{u16 feat, u8 bin, u8 pad}`, leaf floats,
per-feature borders, `level_shift[]` trailer, and 16-byte `vpshufb` tables per split.
Scoring: nibble-packed colmaj matrix, `_mm256_shuffle_epi8` per depth step, gather-leaf
accumulation, sigmoid at the end.

**Expected numbers, shipped `d7 × 2400`:** PR-AUC ≈ 0.8106 (288-image val, `split_seed` 42,
mean of 3 seeds); model ≈ 5.7 ms; `imfeat` 8.2 ms; pipeline ≈ 13.9 ms.
**Expected numbers, cheaper `d7 × 1200`:** model 3.71 ms (1.90 traversal + 1.81 binning);
pipeline ≈ 11.9 ms. All single-threaded.

**Gates that must pass:** blob decode bit-exact vs the trainer (< 1e-5 on probabilities);
fused/coarse binning identical to the full binner (0 mismatches); every traversal variant
bit-identical to the reference path.

---

## 7. Addendum: final shipped configuration

After the layered runtime work in §3.8 lowered the traversal cost, open question 5 in §5 was
re-decided in favour of the larger model. The shipped package (`fastdet`) freezes:

- **`d7 × 2400`**, `border_count 15`, lr 0.1, `SymmetricTree`, Logloss, seed 42, `split_seed 42`;
- top-512 of the 996 gain-ranked columns, 1M sampled cells, `val_frac 0.15`.

**Measured on the 288-image validation split:**

| Metric | Value |
| --- | --- |
| Pooled PR-AUC (mean of 3 seeds) | **0.8106** |
| Per-seed PR-AUC | 0.8117 / 0.8109 / 0.8092 |
| Model inference, C++ | **≈ 5.7–5.8 ms** single-threaded |
| Design-matrix width | 996 (512 kept) |
| Booster borders / features referenced | 6010 / 506 |

`d7 × 2400` beat the HGB reference in all three seeds (mean +0.0010) and was chosen over
`d7 × 1200`. The Python and C++ runtimes were verified to agree bit-for-bit on a frozen fixture
(scalar vs AVX2 max abs diff 0.0; decoded probabilities vs trainer < 3e-8).

### 7.1 The `d7 × 1200` -> `d7 × 2400` trade, and what is still unmeasured

The stated latency goal for the model was **< 5 ms single-threaded**. The shipped
configuration is ~5.7 ms, i.e. the goal was knowingly traded away for quality. What is on
record:

| config | model ms | PR-AUC | note |
| --- | --- | --- | --- |
| `d7 x 1200`, `border_count 15` | **3.71** | *not measured* | the runtime ladder in §3.8 |
| `d7 x 1200`, `border_count 254` | — | 0.8102 (seed 42) | §3.8 sweep row |
| `d7 x 2400`, `border_count 15` | **≈ 5.7** | 0.8106 (mean of 3 seeds) | **shipped** |
| `d7 x 2400`, `border_count 254` | — | 0.8118 (seed 42) | §3.8 sweep row |

**Two measurements were never taken, so the size of the trade is unverified:**

1. **`d7 x 1200` at `border_count 15`, seeds 42/43/44.** Without it, the quality bought by the
   extra ~2 ms is only inferable from the 254-bin sweep rows (+0.0016 at seed 42), which is
   inside the seed-to-seed spread.
2. **`border_count` 15 vs 254 at a fixed tree count, seeds 42/43/44.** The 4-bit quantization
   is described as quality-neutral throughout, but it was never measured against the 254-bin
   model; the total drift from the HGB float reference (0.8110) is therefore the sum of an
   unverified quantization term and the symmetric-tree term (mean ≈ −0.0023, §3.8).

Anyone continuing this work should run both before treating `d7 x 2400` as settled. If (1)
shows `d7 x 1200` within noise of `d7 x 2400`, the default should move back to the 3.71 ms
configuration and the < 5 ms goal is met without a quality concession.

The model is exported as a single `FDT1` container. Research scripts and the HGB path stay in
the legacy `imseg` tree; `fastdet` holds only the training pipeline, the two runtimes, and the
`FDT1`/`IMSY` serialization described in §6.
