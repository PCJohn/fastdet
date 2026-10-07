# How the detector got fast

An engineering log of the latency work on `fastdet`: where it started, the architecture it
ended with, what each step bought, what was tried and dropped, the lessons, how it was
measured, and the settings to run it with. It is the companion of
[`research_report.md`](../research_report.md), which covers the modelling (features, the
booster, quality) and the first inference ladder; this document carries the engineering
from there on, and refers back rather than repeating it.

The constraints throughout: the exported model scores the same bits in the NumPy
reference, the C++ harness and the in-process scorer, on one thread and on any number of
threads, from a packed matrix and from imfeat's maps read in place; no approximation that
changes a probability; nothing tuned to one machine. Every number below was measured, and
says where: the laptop is the maintainer's 22-thread Windows machine (MSVC, AVX2), the
development VM a shared two-core Linux machine (GCC, AVX-512) whose timings wander by
±5–10% and whose instruction counts do not.

## Starting point

The research phase ([§3.8](../research_report.md)) took the model from sklearn's
histogram gradient boosting (a 1926-tree model at about 302 ms per image) to a CatBoost
*symmetric* (oblivious) ensemble scored by a first-party C++ runtime at 3.71 ms on one thread: a
depth-*d* symmetric tree is *d* independent comparisons producing a *d*-bit leaf index;
with `border_count = 15` a feature fits a nibble and a split becomes one `vpshufb` lookup
over 32 cells; cells are nibble-packed; the binner is fused over coarse cells because the
design matrix repeats coarse-level columns about 4.3×. That is the floor this log starts
from: a d7 × 1200 model over 512 columns, 1.90 ms of traversal and 1.81 ms of binning,
scored from a packed design matrix that Python assembled from the feature maps.

What it did not yet have: leaves on a low-bit grid, a coarse tier, an early exit, an
in-process scorer, threads, a front end shared with a host, or any way to read the
features where imfeat left them.

## Terms

| term | meaning |
|---|---|
| cell | one of the 64×64 output cells; a probability each |
| tile | a 4×4 block of cells (256 per frame); the unit of the coarse tier and the exit |
| pack | `kPack` tiles, one index vector's worth; the unit the fine tier keeps or drops |
| level, side | an imfeat pyramid level and its grid side (64, 32, …, 2; 1 for the image-wide block) |
| column | one feature of the design matrix: a level's map value, a context bank, or a broadcast (image-wide) value |
| bin, cut | a column's value quantised against its `border_count` cuts (4 bits) |
| coarse tree | a tree whose splits are all on columns constant inside a tile (side ≤ 16); evaluated once per tile |
| exit stage | a tree count and a threshold; a tile all of whose cells sit below it stops accumulating |
| packed matrix | the features as one contiguous float32 array in slot order (`native_matrix`); the harness's fixture format |
| source | a base pointer and two strides: a column read where it lies (`fastdet_source`) |
| bank | a derived per-cell map (context, ctx2) the scorer now computes itself |
| model stage | everything per frame after imfeat's pass: handing the maps over, binning, trees, sigmoid |

## Result

The model stage per 720p frame on the laptop, two imfeat threads and two scorer threads,
the frame scored inside framegate's gate (`gate_loop.py`, medians over 100–600 frames):

| state | model stage | of which C++ | note |
|---|---|---|---|
| context banks through OpenCV, packed matrix assembled in Python, scored in process | 0.82 ms | — | the Python assembly dominated |
| context banks in C++, scorer reads the maps in place | 0.47 ms | 0.33 | compose 0.12 ms |
| the banks of every level in one call | 0.45 ms | 0.33 | compose 0.10 ms |
| the banks computed inside the scorer, overlapped with binning | **0.37 ms** | 0.34 | 0.03 ms of Python |
| a model fitted on the current front end, another clip | 0.22 ms | 0.19 | more tiles exit early |

The whole gate frame around it was 2.9 ms on the earlier clip and 2.14 ms (p90 2.38) on
the later one, the imfeat pass being the rest; `examples/visualize.py` reports a higher
model-stage figure (0.58–0.60 ms) because it draws between frames and measures with cold
caches. On the development VM the scorer's own benchmark (`pytest -s tests/test_latency.py`,
a random 1178-column model, 4-bit leaves, 1000 trees of depth 5) puts binning plus full
traversal at about 0.27 ms on one thread and the shipped pass (coarse tier, exit, lazy
binning) at 0.2–0.3 ms, on an AVX-512 build.

Against the research floor (3.71 ms for binning and traversal on one thread): fewer and
shallower trees chosen by the quality sweeps (d5 × 1000 against d7 × 1200), two thirds of
them evaluated once per tile instead of per cell, the early exit, the 4-bit leaf grid, and
no packed matrix at all.

## Architecture

1. **One imfeat pass, shared.** The front end is framegate's: the frame thumbnailed inside
   imfeat's pass (`"pow2-fit"`: 720p → 512×320, 1080p → 1024×576) and converted to HSV
   there, stride 1, six levels from 64×64. A host that runs that pass hands its result to
   `Detector.score_raw` / `predict_from_imfeat`; `predict_proba` runs the pass itself,
   building its imfeat computers on first use.
2. **The maps are the design matrix.** Nothing is packed per frame. The scorer is told once
   where each kept column lives (`MapSources`: a slot per level bank or broadcast vector, an
   index along its last axis) and reads each column through a base pointer and two strides —
   a column of imfeat's cell-major map, a plane of its own bank buffer, an entry of the
   image-wide vector.
3. **The banks are the scorer's.** The context banks (box means and ranges of the cell-mean
   luminance, bit for bit what the OpenCV calls they replaced gave) are computed by the
   calling thread, per level, into buffers the scorer keeps, while the other threads bin the
   columns that do not depend on them.
4. **Three phases, bit-identical at any thread count.** Binning by column from a shared
   pool; the coarse tier by unit of tiles, with its exit stage; the fine tier by block of
   alive packs, with lazy binning of the side-64 columns for those packs only. Every cell's
   score is an integer sum of leaf codes, so the thread split cannot change it.
5. **Leaves on a grid.** Per tree an offset, per chunk of 16 trees a power-of-two step, per
   leaf a 4-bit code; the fit is steered to that grid chunk by chunk; both runtimes sum
   codes as integers and convert once.
6. **One file.** The `IMSY` blob carries the trees, the cuts, the per-column level shift,
   the exit stages and the leaf grid; the NumPy reference and the C++ scorer read the same
   bytes.

## What worked

Chronologically. "Laptop" numbers are the maintainer's measurements; "VM" numbers are the
development machine's.

| step | change | measured effect |
|---|---|---|
| 1 | Symmetric trees, own blob, `vpshufb` splits, nibble-packed cells, binning fused over coarse cells | research phase: about 302 → 3.71 ms per image, the latter on one thread ([§3.8](../research_report.md)) |
| 2 | Leaves on a low-bit grid (4 bits) with a quantisation-aware chunked fit | fine-tree work became one byte shuffle per nibble plane of codes instead of four per float leaf; quality equal to 8-bit leaves (`--leaf-bits 4 / 8` in the tuning table) |
| 3 | Resolution-tiered boosting (two thirds of the trees split only on tile-constant columns, evaluated once per tile), exit stages calibrated at fit time, lazy binning of side-64 columns | the coarse tier costs a fifth of a fine tree or less; tiles that exit skip the fine tier and their side-64 binning; quality 0.9012 against 0.9014 untiered on the synthetic set |
| 4 | The scorer as an in-process extension (nanobind + Highway, built by `pip install`), GIL released during a pass | no subprocess, no fixture file, no copy of the features into another process |
| 5 | Thread pools with parked workers, in imfeat and in the scorer; one wake-up per pass | laptop, 1080p at 1024 px: the pass 11.1 / 5.6 / 3.4 ms on 1 / 2 / 4 threads; the scorer's random-model benchmark 0.45 → 0.35 ms on the VM's two threads |
| 6 | imfeat's computers built lazily, on first use | a detector used through `score_raw` never spawns its own pool |
| 7 | The front end moved into imfeat: BGR → HSV inside the pass, then the `INTER_AREA` thumbnail, then the thumbnail sized from the frame (`"pow2"`, `"pow2-fit"`) | the two OpenCV passes over the frame are gone (imfeat's log has the numbers); a 720p frame runs at 512×320 instead of being upscaled to 1024 |
| 8 | The packed matrix written into a pinned, reused buffer | removed the per-frame allocation on the path that still packed |
| 9 | The context banks in C++ (Highway; one call for every level, reading the mean column where it lies), and the scorer reading imfeat's maps in place | laptop model stage 0.82 → 0.47 ms |
| 10 | Every level's banks in one extension call; the global vector by one extension call; `log1p` cached per frame size | 0.47 → 0.45 ms |
| 11 | The banks computed inside the scorer by the calling thread while the others bin; the columns that read them binned after | 0.45 → 0.37 ms |
| 12 | A separate thread count for the scorer in framegate (`model_threads`) | laptop: 2 threads 0.23 ms, 8 threads 0.31, 16 threads 0.40 for the scorer's pass — a control, not a lever |
| 13 | Training: the pool quantised once and reused by every stage and chunk; CatBoost on the GPU when it sees one | the quantisation-aware fit is 63 chunked fits for 1000 trees; on the GPU each uploads the quantised pool once |

**Step 2, the leaf grid.** Leaf magnitudes shrink as boosting proceeds, so one global step
quantises the late trees into noise; a step per chunk of 16 trees, restarting at the
coarse/fine boundary, keeps every tree resolved (`leaf_grid` and `test_quantisation.py`).
The fit is run chunk by chunk against the *quantised* running score (CatBoost `baseline`),
so later trees correct the rounding of earlier ones and the export reproduces the grid the
fit saw. Both runtimes accumulate `code << shift[chunk]` as integers and apply
`sum(offsets) + total * 2**e_min` once, which is what makes the Python and C++ scores
identical to the bit rather than merely close.

**Step 3, the coarse tier and the exit.** A tree whose splits are all on columns constant
inside a 4×4 tile has one leaf per tile; the scorer builds its index with byte shuffles over
the coarse planes and gathers the code once per tile. Its tile-level partial score is what
the first exit stage gates on: calibrated on the training images as the lowest partial score
of any cell that ends at or above `exit_keep_prob` (0.05), minus a margin of 2.0 in raw
score units, so no cell that matters on the calibration set can be stopped and unseen images
have room to move. The C++ scorer bins the side-64 columns only for the tiles alive after
that stage, reading each alive tile's 16 cells straight from the map. The exit is a contract
about cells above `exit_keep_prob`; a consumer that needs every score exact passes
`use_exit=False`.

**Step 9, reading the maps in place.** The old path packed the features: Python gathered
every kept column of every level into one float32 matrix per frame (level values repeated
over their blocks of cells), and the scorer read that. With `MapSources` the scorer takes one
array per slot and, per column, a slot and an index along its last axis; `fastdet_source`
is a base pointer with a row and a column stride, so a packed fixture (strides 1) and a
cell-major map (column stride = the record width, 162 floats) are one code path, and the
harness scores every fixture both ways and requires the same bytes. Strided column reads
are latency-bound — a side-64 column touches 4096 cache lines — which is why the C++ pass
of the maps costs about 0.1 ms more than the pass of the packed matrix did; what it saves
is the packing itself, which was the larger part of the 0.82 ms. The context banks moved to
C++ at the same time: `cv2.boxFilter`'s double sums with a single rounding and
`dilate`/`erode`'s max and min, written in Highway so every lane does the scalar
operation in the scalar order and the bytes match the OpenCV route on every grid size
(`test_context_banks.py` pins them).

**Step 11, the banks inside the scorer.** With the banks in C++ but computed before the
pass, the scorer waited for them. Now the calling thread computes them into the scorer's own
buffers (`set_bank_slot`: a Python-owned `(7, side, side)` buffer per level whose first
two planes, the cell coordinates, are written once) and then bins the columns that read
them, while the other threads take the remaining columns from a shared atomic pool, eight at
a time; the threads finish together whatever the banks cost. `score_maps` takes `None` for
a bank slot; the raw path (`Detector.score_raw`) hands over imfeat's maps and the global
vector and nothing else per frame.

## What did not work, or was not kept

* **The planar route** (September 2026; not in the tree). Before the strided sources,
  the plan for not packing was to have imfeat lay its maps out one contiguous plane per
  feature (`FeatureComputer(planar=True)`), fastdet's scorer bin each plane in place
  (`score_planes`, per-feature pointers through the C API) and framegate ask imfeat for
  that layout. It worked and measured: the packed copy was 3.7 MB a frame, 0.4 ms on the VM
  and 0.58 ms on the laptop, and the handover became microseconds. It was not kept: the
  layout cost imfeat a second output format and two rewrites of its summary fold for MSVC,
  and the same saving was then had without touching imfeat — a source is a base pointer and
  two strides, and a cell-major map is read where it lies (step 9).
* **Block binning** (`fastdet-3`, shelved). The binner wrote each column's bins to its
  plane; a block binner that staged bins as bytes and binned row bands at a time was built
  to turn the strided column reads into sequential ones. Two attempts were slower live before
  one was not: stores 16 bytes at a time to rows a power of two apart hit one L1 set (fixed
  by completing each row line with four consecutive stores), and planes written 256 bytes at
  a time 4 KB apart paid cold read-for-ownership (fixed by staging bytes and laying planes out
  whole). The third was correct on 16 threads too (a band of one tile row ran the block past
  its end until `nr = min(block_rows, r1 - b0)`), and on the laptop it made no measurable
  difference to the model stage (0.49 against 0.47 ms, within noise). The cold strided reads
  are a memory-latency cost that re-laying the writes does not remove; they would need the
  maps to arrive warm (they are written by imfeat's threads on other cores) or to be read
  by the thread that wrote them.
* **A fused design from everything imfeat computes** (`feature_mode="all"`; reverted).
  The front end was extended to read the channel-pair covariances (`cross`, 6 columns per
  level), the per-level summary of every feature over the level's cells (`summary`, 648 per
  level, broadcast) and the row/column projection profiles cut into the level's bands
  (`profile`, 24 per level), 5252 columns against 1178; every new column equal to imfeat's
  arrays bit for bit, the profile bands by a C++ kernel built with `-ffp-contract=off` so
  MSVC and GCC agree, the broadcast columns binned once per frame, the per-cell ones like
  raw columns. The fit on it was *worse* by a wide margin on the maintainer's validation split, and
  the whole change was reverted. The likely reason is worth keeping: 3888 of the added
  columns were per-image constants (the summaries), on top of the 164 the global block
  already carries, and that many image-level constants let the trees fingerprint individual
  training images. The per-cell families were small (36 cross, 144 profile columns). Nothing
  reads `cross` or `profiles` now, and imfeat stopped computing them (its log, steps 8–9);
  imfeat's `FEATURES.md` keeps their definitions. If a subset is ever retried, the honest
  experiment is one family at a time against the mainline PR-AUC on the same split.
* **The training matrix in Fortran order** (reverted with the item above, but the finding
  stands). CatBoost's `Pool` copies a C-ordered float32 matrix into its own storage and takes
  a Fortran-ordered one as it is: measured on 100k × 5252, `Pool()` 6.1 s and +1.0× the
  matrix in memory against 0.0 s and +0; the quantisation then adds 0.30× (one byte per
  value) and CatBoost keeps the raw matrix referenced for the pool's lifetime either way.
  So a fit peaks at 2.3× the float matrix as the tree stands (46 GiB for 1 M cells × 5252
  columns, 10 GiB at 1178) and would peak at 1.3× with a Fortran-ordered design — assembled
  with the rows grouped by image, because a scattered row write into a Fortran array is ten
  times slower than a slice copy. The README's claim that the matrix is "released" after
  quantisation was wrong and is corrected below. Open: see *Open ideas*.
* **The vectorised staging transpose** inside the block binner and **half-width loads**
  sized to earlier stores — the same lesson as imfeat's: store-forwarding and set conflicts,
  not arithmetic, decided those loops.
* **Measuring on the laptop with the C++ harness.** The harness (`fastdet_score`) gates
  the scorer and prints stage timings, but its warm, fixture-fed numbers are not the gate's
  cold, map-fed ones; the maintainer's rule is that a change counts when `gate_loop.py` or
  `visualize.py` shows it on the laptop, and the harness is for correctness.
* **The approximation methods** of the research phase — a soft per-cell cascade and a
  coarse-to-fine pyramid of rejections ([§3.9](../research_report.md)) — were built and
  measured and are not in the tree: they change probabilities, and the exact exit of step 3
  took their place.
* **`extra_scales`** (a second imfeat pass on a downsized copy of the thumbnail) is worth
  +0.01 to +0.02 PR-AUC and costs a whole extra pass per frame; off by default for that
  reason.
* **float16** was asked about and answered rather than tried: no CPU path on x86 (numpy
  float16 arithmetic is 5–15× slower than float32), and it would change every output.

## Lessons

1. **Deleting the handover beats speeding it up.** The largest single gain (0.82 → 0.47 ms)
   came from not building the packed matrix at all, not from packing faster.
2. **Cold reads are a cost of the layout, not of the code.** The maps are written by
   imfeat's threads and read by the scorer's; a column read is a strided walk over cache
   lines another core wrote. Re-laying the writes (block binning) did not help; overlapping
   the reads with other work (banks inside the scorer) did.
3. **Overlap serial work with the parallel phase.** The banks cost the same either way; the
   threads finishing together is what moved 0.45 → 0.37 ms.
4. **Per-frame numpy calls have a fixed tax.** The first ufunc after a C++ pass costs
   20–30 µs on the VM (cold dispatch); the compose path went from several such calls to one
   extension call per kind of work, and the `log1p` of the frame area is cached per frame
   size.
5. **Exactness comes from integer sums.** Leaf codes summed as integers, converted once,
   are what make three implementations and any thread count agree to the bit; a float
   accumulation per thread would not.
6. **A control is not a lever.** The scorer's pass is a few hundred microseconds; its
   synchronisation costs more than a wider split saves above two threads on the laptop.
   Measure the thread count on the machine, do not raise it.
7. **Width is not free at training time.** The design matrix's width sets the fit's memory
   (1.3–2.3× the float32 matrix) and its CPU-side quantisation time, and more columns do not
   mean a better model (the `all` front end).
8. **Measure the thing the user runs.** The harness, the VM and the laptop disagree in
   level; the gate loop on the laptop is the number of record, and the VM is for shape.

## Measuring

* **The gate loop** (`gate_loop.py`, delivered alongside the patches; `python gate_loop.py
  clip.mp4 --model text.fdt --threads 2 --model-threads 2 --frames 200`): framegate's
  `gate.frame()` per frame with the model loaded, the model stage and `score_maps` timed by
  monkey-patching `ModelBank.maps` and `NativeScorer.score_maps`, medians and p90, the GC
  disabled. The number of record for the model stage.
* **`examples/visualize.py --model text.fdt --model-threads N`** (framegate): the same
  through the demo, which draws between frames; its model-stage median runs higher for that
  reason and is the number to compare thread counts with.
* **`pytest -s tests/test_latency.py`**: the C++ harness on a random model of the shipped
  shape (1000 trees, depth 5, a coarse tier, synthetic exit stages), binning and traversal
  separately, one thread and the default count, plus the front end at three thumbnail sizes
  with the resize and conversion inside imfeat's pass timed against OpenCV's. Random leaves
  cannot be calibrated, so its exit keeps half the tiles; a fitted model's stages depend on
  the frame.
* **`fastdet_score model.fdt fixture.f32 expected.f32`**: the harness on a real model and
  a real fixture (`Detector.native_matrix(image)`), the authoritative traversal number and
  the correctness gate (below).
* **Instruction counts** for anything on the VM: `valgrind --tool=callgrind`, differencing a
  2-frame and a 12-frame run, and summing the self cost of the extension's own object rather
  than the process (a fresh build's first run compiles the package's `.pyc`, which
  contaminates a whole-process difference by a few percent).

## Verifying

* The suite (103 tests on mainline) fits a tiny detector, exports it, reloads it, and
  requires the reloaded model to reproduce the scores exactly; the C++ harness is built with
  CMake and fed the same artifact plus a fixture per image, and must agree with the Python
  runtime (which scores the dense matrix, sharing no layout code), on one thread, on the
  configured count and on 16 threads where some threads own no tiles until the fine tier;
  the maps path is scored plain and with the exit against the packed path; the banks are
  pinned against the OpenCV route on every grid size; the quantisation tests check that the
  blob's codes, offsets and shifts give exactly the leaves the exporter quantised to.
* Every patch was applied to a fresh clone and its suite run there; the maintainer's
  Windows run of the suite is the one that counts.
* ruff (`ALL`), black, mypy strict (including the tests); clang-format and clang-tidy
  (bugprone, performance, clang-analyzer, misc), cppcheck, GCC and clang with `-Wall
  -Wextra -Wshadow` for the C++.

## Recommended configuration

Inference:

* `Detector.load(path, threads=n)` with `n` = 2 when the machine is shared, up to 4 when
  it is not; the gain flattens past that for imfeat and reverses past two for the scorer
  on the laptop. In framegate, `feat_threads=2` and `model_threads` left at 0 (= 2).
* Inside a host that already runs imfeat's pass, `Detector.score_raw(result, (h, w))`
  (what framegate does); the detector then never builds its own imfeat computers. On its
  own, `predict_proba(frame)` with the frame whole — the thumbnail and the HSV conversion
  happen inside the pass.
* Leave the early exit on (`use_exit=True`, the default); turn it off only for a consumer
  that needs exact scores for cells below `exit_keep_prob`.
* Call `Detector.close()` when a stream ends, and before interpreter shutdown on Windows.
* In a frame loop, disable the garbage collector and collect periodically; it is where the
  latency spikes come from, not the steady cost.
* Pruning (`top_k_features`) reduces the model stage only; the front end computes every
  column regardless. Regenerate the ranking from a full-width fit on the current front end
  before pruning (the bundled one is for an older front end).

Training:

* The measured defaults: 1000 trees of depth 5, learning rate 0.1, `border_count` 15
  (7 is free speed, within noise in quality), 4-bit leaves with the quantisation-aware
  fit, `coarse_fraction` 0.667, the exit on. Depth 7 costs a third more in the scorer for
  no quality.
* Keep `feature_mode` at its default (`raw_plus_global_context_ext`, 1178 columns).
* Memory: a fit holds about 2.3× the float32 design matrix at its peak as the tree stands
  (CatBoost copies the C-ordered matrix, then quantises it) and 1.3× for the rest of the
  fit; `max_train_cells` and `neg_pos_ratio` are the knobs (1 M cells × 1178 columns is
  4.4 GiB of matrix, so about 10 GiB at the peak). A "CatBoost is using more CPU RAM than
  the limit" warning means paging.
* `task_type="GPU"` trains several times faster on millions of cells, but nothing touches
  the GPU until CatBoost's CPU-side quantisation of the whole matrix is done — minutes for a
  20 GiB matrix — so an idle `nvidia-smi` early in a fit is normal.
* `quantisation_aware=False` is the fast path for 8-bit leaves (one fit per stage instead
  of 63 chunks); keep it on for 4-bit leaves.

## Open ideas

* **The Fortran-ordered design matrix** (above): halves a fit's peak memory and removes a
  minute of CatBoost's copy per 20 GiB; a one-line allocation change plus assembling rows in
  image order and reordering the labels. Measured, not in the tree.
* **Warm maps.** The 0.1 ms the map path pays over the packed one is cold strided reads
  of maps imfeat's threads wrote on other cores. Binning a level's columns on the thread
  that produced the level, or imfeat handing the scorer the maps band by band, would remove
  it; both cross the library boundary.
* **A model fitted on the current front end** with a regenerated ranking and pruning to
  512 columns would halve the binning (the benchmark's 512-column row).
* **imfeat's remaining exact items** (its log): batched folds with the roll-up fused in,
  and the serial tail at two threads. The pass is 85% of a gate frame; the model stage is
  10%.
