# Final engineering review

Review date: 2026-11-01. Scope: the reconstructed repository and its current
artifacts, plus the user-uploaded original assessment. This is a separate review
pass by the preparing assistant, not an external hiring decision or independent
third-party certification.

## Findings fixed before submission packaging

| Finding | Consequence | Repair and regression evidence |
| --- | --- | --- |
| Confusion inputs silently cast to int64 | Fractional counts could become plausible but incorrect metrics | Require nonnegative integer counts and safe arithmetic; tests/test_metrics.py |
| Saved metrics trusted during collection | Edited scalar scores could enter charts despite conflicting confusion counts | Recompute IoU, Dice, accuracy, support and totals; validate finite dense CE; tests/test_metrics.py and test_campaign_execution.py |
| Saved best selection checked only against its own prefix | A stale best-validation file could disagree with later history | Check contiguous full history, earliest maximum and saved selected metrics before export |
| Summary/checkpoint binding not enforced at export | Different model bytes could be associated with saved scores | Validate summary checkpoint SHA-256, run identity and configuration snapshot |
| Final-test cache skipped without checking content | Stale or inconsistent test results could be reused | Validate frozen configuration/run list/checkpoint bytes and cached result provenance/metrics before any new inference |
| Nonfinite valid evaluation logits could yield meaningless predictions | Argmax could populate a matrix before a later serialization failure | Fail explicitly at valid logits and restore the model's prior training mode |

Twenty focused regression checks passed, then the complete source-bound suite
passed: **131 passed, 1 CUDA-only skipped, zero failures/errors**. The authoritative
JSON and JUnit XML are in results/verification. The model, dataset splitting,
point sampler and experiment YAML files were not modified by this review.

## Mathematical and computer-vision review

- PCE selects valid annotated logits before arithmetic, computes stable
  log-sum-exp minus target score and averages over the batch's selected count.
  Empty selections remain connected to autograd; empty batches skip forward
  and optimizer operations. Tests check CE equivalence, NaN exclusion, gradients,
  invalid labels, batch weighting, precision and state preservation.
- Dataset batches expose sparse targets only for sparse training. Point locations
  depend on valid geometry and local seeds, not dense class values. D4 transforms
  align all tensors. Fixed points do not accumulate supervision across epochs.
- Image/mask dimensions, georeferencing and labels are checked. Pixels are not
  reprojected; geographic grouping precedes tiling. Duplicate-image and proximity
  constraints are tested on generated rasters. Actual LandCover.ai separation
  cannot be verified without its missing rasters and manifest.
- U-Net raw-logit output shape, arbitrary spatial padding/cropping, parameter
  count and backward pass are tested. ImageNet initialization uses its verified
  checksum and explicit failure behavior. The actual U-Net is also exercised in
  synthetic training/checkpoint tests; faster runner tests use a tiny CNN.
- Evaluation pools confusion counts before calculating metrics, includes class 0,
  excludes padding and handles absent/false-positive-only classes explicitly.
  Training creates no test dataset. Test inference requires protocol membership;
  test fixtures do not constitute real test access.

## Reproducibility and engineering review

Seeds, initialization policy, loader RNG, deterministic operations, annotations,
subsets, configuration/source hashes and optimizer/scheduler/scaler states are
recorded. Epoch-boundary resume is tested against uninterrupted training on
synthetic fixtures. Completed outputs and immutable controls are protected.
The complete 77-package hashed CPU lock was installed in a separate clean
environment and passed dependency/import/model-forward checks. The setup script
uses a writable cache and retains TLS/hash verification.

The notebook executes ten code cells with one embedded geometric sampling figure.
Its real-data, checkpoint, evaluation and experiment sections explicitly remain
blocked. `real_data_assessment_complete` is false despite successful execution.
The report uses the actual 131-pass count and zero campaign rows, with no historical
smoke metrics in performance tables. Original requirements are quoted with the
uploaded document hash. Report citations are identifiable scientific references;
repository metadata retrieval and remaining source-access limits are recorded.

## Residual limitations and completion gates

1. Official data, the recovered frozen split and trained checkpoints are absent.
   Real-data labels, partitions, training/reload and performance remain unverified.
2. The full CPU campaign's wall time is unknown. No claim of feasible three-seed
   runtime follows from tiny synthetic checks or historical smoke timing.
3. CUDA/AMP behavior is unverified. Bitwise portability across devices or library
   versions is not promised. Resume also requires the original absolute output
   location; a moved archive can run new experiments but cannot directly resume
   a lost checkpoint at a different path.
4. Numerical consistency checks cannot detect deliberately fabricated coherent
   artifacts; raw data checksums and independently repeated runs are still needed.
5. Original uncommitted code, exact point seeds and manifest were not recovered.
   The behavioral reconstruction must not be described as the identical earlier
   experiment. A new split requires explicit provenance and audit.

No remaining verified defect requires a model rewrite. Missing empirical evidence
is a mandatory assessment gap, not a cosmetic documentation issue.
