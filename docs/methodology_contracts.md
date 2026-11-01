# Methodology contracts

## Data and split

Original RGB GeoTIFFs stay at native resolution. Inventory preserves the source
CRS, including non-authority WKT, and transforms footprint geometry to EPSG:2180
for metric distances. It does not reproject image or mask pixels.

Scenes are connected when minimum footprint distance is <=256 m; transitive
components are indivisible. Exact duplicate image hashes add grouping edges.
Sorted components are permuted using NumPy PCG64 with seed 2026. The reconstructed
allocator uses floor(0.70 * groups) train, round(0.15 * groups) validation and the
remainder test. Group/scene/image-hash disjointness and minimum cross-partition
distances are checked. A separated footprint is not proof of statistical
independence; shared acquisition identifiers remain unknown.

Nonoverlapping 256x256 windows use stride 256. Out-of-bounds padding is invalid,
with target 255. Background 0 remains supervised/evaluated. RGB is float32 /255
then normalized using mean [.485,.456,.406], std [.229,.224,.225]. Masks are never
interpolated. Fixed subsets use seed 2718 independently of run and condition and
without class information.

## Sparse annotations and loss

Uniform sampling permutes valid flattened coordinates with local PCG64 and takes
a prefix. Budgets 10/50/200 are nested for a fixed run/tile seed. Stratified
sampling creates exactly K cells via horizontal bands and column divisions,
draws one valid point per nonempty cell and fills deficits uniformly without
replacement. No semantic class labels guide point locations. If a tile has fewer
than K valid pixels, every valid pixel is selected.

Annotations are generated in original coordinates and never refreshed by epoch.
D4 transforms jointly move image, targets, point mask and validity mask. The
diagnostic interface may read dense truth; the sparse training interface never
returns unselected truth.

PartialCrossEntropyLoss takes [B,C,H,W] logits and [B,H,W] integer targets/binary
masks. It selects valid, in-range annotated logits before floating-point
arithmetic, including before touching unselected NaNs. It computes logsumexp
minus the selected class score, summed and divided by the total selected count
across the batch. Low-precision arithmetic is promoted to float32. An empty
selection returns an autograd-connected zero. The trainer skips empty batches
before forward propagation, preserving BatchNorm buffers and AdamW state.

## Optimization and selection

Python/NumPy/Torch RNGs and deterministic Torch algorithms are configured.
Shuffling has a separate saved generator. Workers use spawn and a 60-second
timeout. CUDA AMP is permitted only on CUDA and counts scaler-overflow skips
separately from optimizer updates.

Training constructs train/validation datasets only. Highest dense validation
mIoU selects the checkpoint; the earliest exact tie wins. Early stopping is
optional but disabled in the supplied comparison profiles. Resume is at epoch
boundaries, uses the same controls and source/subset provenance, and requires
last.pt plus best.pt in the same output directory. Completed runs are protected.

## Evaluation

One global confusion matrix is accumulated with truth rows and prediction
columns. Class 0 is included. Classes absent from both truth and prediction have
undefined IoU/Dice and are omitted from the corresponding macro average.
False-positive-only classes have score zero and remain included.

Report mIoU, class IoU, mean/class Dice, pixel accuracy, class support, confusion
matrix, valid pixel count and dense CE. Padded pixels do not contribute.
Test inference requires both explicit final-test access and membership of the
checkpoint in a frozen protocol. Preparation may inspect test metadata/label
validity; this is not inference or model selection.

The dense reference uses ordinary CE and more annotation availability. It is
not guaranteed to outperform a sparse model and is not a guaranteed measured
upper bound.

## Phase 4 measurement additions

Training and validation wall times are measured separately for every completed
epoch, including data-loader time and excluding initialization/checkpoint I/O.
CUDA measurements synchronize the device at boundaries. Epoch timing records
are saved in checkpoints and timing.json, so cumulative totals include epochs
completed before a resume. Timing is kept separate from deterministic loss and
metric history.

Spatial coverage is descriptive and never affects point selection. Occupied-cell
fraction uses a fixed 8x8 grid over each tile's valid bounding rectangle; cells
with no valid pixels are excluded from the denominator. Mean nearest-neighbor
distance normalizes each spatial axis to [0,1]. It is undefined for fewer than
two points and is not computed for dense masks with more than 200 points. Run
summaries average defined tile diagnostics and report the number of contributing
tiles. These geometric measurements do not use hidden class labels, and greater
coverage alone does not establish better segmentation.

The experimental labeling unit is a 256x256 training tile, not an entire original
aerial scene. At a fully valid tile, nominal 10/50/200-point fractions are
0.0152587890625%, 0.0762939453125%, and 0.30517578125%. Actual campaign fractions
must use unique annotations divided by pooled valid training pixels, accounting
for edge tiles and budgets clipped to available valid pixels.
