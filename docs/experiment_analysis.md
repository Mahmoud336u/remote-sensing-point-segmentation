# Phase 4: controlled experiments — execution blocked

Assessment date: 2026-11-01 (Africa/Cairo).

**Zero real-data experiments have completed.** The official LandCover.ai download
was attempted and failed with proxy CONNECT HTTP 403. The archive, extracted
rasters, frozen manifest and real-data smoke record are absent. The experiment
runner was invoked and rejected execution at its real-data-smoke preflight gate.
This document reports the observed blocker and completed software checks; it
contains no segmentation-performance findings.

## Preliminary verification

| Check | Observed result |
| --- | --- |
| Current unit/integration suite | 117 passed; one CUDA-only skipped; zero failures/errors |
| Sparse-only batch/loss contract | Passed on synthetic raster fixtures, including excluded logits/labels and empty batches |
| Checkpoint/RNG/optimizer resume | Passed on synthetic fixtures; cumulative epoch timing preserved |
| Configuration snapshots, run IDs, failure handling and final-test gates | Integration tests passed |
| Model and experimental controls | Architecture, split code, sampling algorithms and configurations unchanged from 9282366 |
| Official archive and partitions | Blocked: archive/manifest absent |
| Full real validation/test masks | Unavailable; cannot verify |
| Real-data training and checkpoint reload | Not run; required gate remains unsatisfied |
| GPU memory | No CUDA device available |

Source-bound results are in
[current_suite.json](../results/verification/current_suite.json) and
[phase4_preflight.json](../results/verification/phase4_preflight.json).
The historical smoke scores and old 400-pass test total are not evidence for this
campaign. Synthetic final-test fixtures do not perform LandCover.ai inference.

## Resource budget and fixed controls

The running instance provides a four-core CPU quota and a 32 GiB memory limit.
The preflight JSON records measured free workspace bytes. Only one training
process is planned at a time, using four Torch threads.

The existing cpu_paired profile is the candidate: 10 epochs, batch size two,
zero loader workers, 256x256 native-resolution tiles, train/validation/test caps
128/32/32 and paired seeds 17/29/43. At the training cap, this is at most 640
training batches per run and 7,680 across twelve runs. These are workload counts,
not measured runtime estimates. Wall-time feasibility is unresolved; real-data
calibration is required before selecting how many paired seed blocks are feasible.

Priority remains one completed run each for A1/A2/A3/B2, then additional paired
seeds, then the optional dense reference. Architecture, initialization, split,
optimizer, scheduler, preprocessing and training horizon remain fixed within a
campaign. The optional dense reference is not part of this candidate's required
run list.

## Run status and quantitative results

The labeling unit is a training tile, not an entire original aerial scene.
A2 is reused as B1. All twelve candidate runs are recorded with individual IDs,
seeds and the blocker in [run_status.csv](../results/cpu_paired/run_status.csv)
and [run_status.json](../results/cpu_paired/run_status.json).

| ID | Sampling | Planned points/tile | Completed seeds | mIoU | Mean Dice | Pixel accuracy | Status |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| A1 | Uniform | 10 | 0 | — | — | — | Blocked |
| A2/B1 | Uniform | 50 | 0 | — | — | — | Blocked |
| A3 | Uniform | 200 | 0 | — | — | — | Blocked |
| B2 | Stratified | 50 | 0 | — | — | — | Blocked |
| Optional | Dense | All | 0 | — | — | — | Not scheduled |

A dash means unmeasured, never zero. [results.csv](../results/cpu_paired/results.csv)
contains only its header and [results.json](../results/cpu_paired/results.json)
contains an empty list. Per-class IoU, timing, actual annotation fractions and
seed variability remain unmeasured. No mean or SD is reported with zero runs.

For a fully valid 256x256 tile, the mathematical nominal fractions are
10/65,536 = 0.0152587890625%, 50/65,536 = 0.0762939453125%, and
200/65,536 = 0.30517578125%. These are not measured campaign fractions.
The actual denominator will pool valid pixels from the fixed selected subset;
edge padding and clipped point budgets will be accounted for in annotations.json.

## Measurements implemented for execution

Completed-run CSV/JSON exports now carry experiment IDs, sampling/point budgets,
segmentation metrics, cumulative training/validation duration, actual pooled
annotation fractions and geometric coverage. Timing is measured per completed
epoch and survives resume, with loader time included and checkpoint I/O excluded.

Spatial coverage uses occupied cells in an 8x8 grid over valid tile geometry and
normalized nearest-neighbor point distances. It uses no class labels and cannot
change the selected points. The grid resolution is a diagnostic choice, not a
claim of semantic coverage or performance superiority.

The existing figure commands consume actual completed run artifacts and retain
editable JSON/CSV/NPZ inputs. No campaign plots, qualitative predictions, error
maps, training logs or trained campaign checkpoints exist yet. Notebook/toy-test
figures are not substituted for these deliverables.

## Scientific interpretation and limitations

The annotation-density and stratification hypotheses remain untested. There is
no evidence yet to identify benefiting or difficult classes, rank strategies,
quantify seed stability, identify actual failure cases or establish a useful
supervision threshold. No causal or significance claim is supported.

Class imbalance and scene complexity must be examined through measured class
support, per-class confusion and fixed qualitative examples after data access.
ImageNet-to-aerial-image domain mismatch and the capped CPU subsets limit future
generalization claims. Simulated point labels do not measure human annotation
time, ambiguity, mistakes or behavior. Geographic grouping reduces one leakage
risk but does not establish full statistical independence.

## Concrete prerequisite and continuation

The current recovery environment draft now includes landcover.ai.linuxpolska.com.
The applied policy still blocks it. The configuration tool reports that the user
must review/save this change and publish the current recovery environment to
activate the configuration; saving the draft alone did not apply access.

Once access is active, rerun official download/checksum/extraction and geographic
audit, preserve the generated manifest, verify dense held-out masks, complete
real-data smoke and exact reload, and calibrate compute. Execute the controlled
campaign only after that gate passes. Freeze selected checkpoints before final
test inference, verify saved confusion matrices and figures, and replace this
blocked-status analysis with measured findings.
