# Phase 4 experimental protocol

Question A compares uniform 10, 50 and 200 annotations per valid tile. Question B
compares uniform_50 (reuse A2 as B1) with stratified_50. Dense_reference is optional.

Within a campaign, hold architecture, ImageNet initialization policy, geographic
split, preprocessing, optimizer, scheduler, batch size, selected tiles and
training horizon fixed. All supplied profiles use learning rate .001, weight
decay .0001, cosine minimum .000001 and four Torch threads. Early stopping is
disabled. Primary comparisons use paired seeds [17,29,43] when measured compute
permits.

Original main: 30 epochs, batch 8, four workers, three seeds, no tile caps.
Original CPU pilot: 10 epochs, batch 2, zero workers, seed 17, caps 128/32/32.
Separate cpu_paired: the same pilot controls with three seeds and four sparse
conditions. Smoke: two epochs, 8/4/4 tiles, uniform_50 only, integration evidence.

Execution priority is one completed run per sparse condition, then additional
paired seeds, then optional dense reference. Plan commands do not execute runs.
Preflight requires a source-bound passing suite, official data verification,
frozen manifest, and a completed real-data smoke with exact checkpoint reload.
Do not estimate campaign feasibility solely from the historical smoke duration.

Each run has a distinct profile/condition/seed identifier and immutable config,
structured history, annotation coordinates/cost, provenance and best/last
checkpoints. Record failures and incomplete runs explicitly. Resume only with
matching controls and preserved artifacts.

Select checkpoints by validation mIoU. Freeze the full campaign protocol and
checkpoint hashes before test inference. The final-test flag must be explicit;
test results cannot choose checkpoints or revise controls.

Output actual CSV/JSON rows, training curves, mIoU/Dice versus annotation density,
the 50-point strategy comparison, class IoUs, and fixed qualitative examples/error
maps. Keep editable chart data. Aggregate only completed runs: one seed gets a
single value and no SD; two or more get arithmetic mean and sample SD (ddof=1).
SD is not a confidence interval. Report paired seed differences separately.
Do not claim significance from small changes with few seeds.

Interpret class imbalance, rare-class failures, scene complexity, run variance,
natural-image/aerial-image initialization mismatch, limited subsets and simulated
versus human annotations. A final technical assessment report is outside this
Phase 4 deliverable.
