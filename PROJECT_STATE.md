# Project state

Recovery checkpoint date: 2026-10-31. Reconstruction work: 2026-11-01 (Africa/Cairo).

## Recovery outcome

The accessible workspace and Git remote contained only the initial commit
1f77b1099a5be1cb110ea4a5b5c414121d856eab. No original source, dataset, geographic
manifest, trained checkpoints or historical run records were restored.
The original three-file checkout was copied to /workspace/scratch/recovery-baseline.

All new project code is a documented reconstruction. Historical source hashes,
the 400-pass/4-skip result, and smoke observations are retained as historical
claims in docs/recovery_checkpoint.json, not current verification evidence.

## Current capabilities

- Reconstructed configuration, dataset inventory/grouping, sparse annotation
  sampling, Partial CE, U-Net, training, resume, evaluation and visualization.
- Actual campaign execution, failure reporting, immutable configurations,
  completed-run protection and final-test protocol freezing.
- A complete hashed CPU environment lock and a repeatable local setup script.
- New source-bound test reports and an executed mathematical demo notebook.
- Official ResNet-18 initialization downloaded and verified against its full hash.

The authoritative current test count is results/verification/current_suite.json.
Notebook execution status is results/verification/notebook.json. These checks use
synthetic rasters where indicated and do not replace the required real-data gate.

## Blocking prerequisite

The official LandCover.ai archive is absent. The current proxy returns HTTP 403
for landcover.ai.linuxpolska.com, which is not in the applied allowlist. The user
has said they will enable that domain in the recovery environment. No old
environment configuration was changed or published.

Consequently, official archive verification, real-data inventory and split,
real-data smoke/reload, and Phase 4 remain pending. There are **zero real-data
campaign runs and zero real-data test evaluations** in this reconstructed tree.
Synthetic test fixtures exercise test-evaluation code without touching LandCover.ai.

## Resume in order

1. Verify that the domain is allowed in the applied recovery environment policy.
2. Run the data preparation command in README. Preserve any recovered original
   manifest separately; do not silently overwrite or reinterpret it.
3. Record the new manifest, source hashes, geographic audit and split assignments.
   A reconstructed split/seed implementation cannot claim historical identity.
4. Run scripts.verify_tests again if source or configuration changes.
5. Train a new real-data smoke; verify exact checkpoint reload; render figures.
6. Rerun and inspect the notebook with the new evidence.
7. Measure actual CPU timing, select the documented campaign profile, and pass
   its preflight. Complete one run for each sparse arm, then paired seeds when
   feasible, then the optional dense reference.
8. Freeze the protocol/checkpoints before final test inference. Export actual
   rows, plots and per-class results; write docs/experiment_analysis.md.
9. Update this file and push subsequent reviewed changes to the recovery branch.

The original main and CPU-pilot profiles are preserved; cpu_paired adds three
paired seeds in a separate file. No smoke score may be used as a campaign score.
No final technical assessment report has been written.
