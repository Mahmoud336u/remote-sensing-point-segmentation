# Results and verification

Only new executed checks appear in results/verification. The full suite binds its
result to current source/test/config fingerprints. Notebook status covers the
executed demo and does not imply real-data verification. ImageNet initialization
has a separate full-artifact hash record.

Runs save run.json, config.json, annotations.json, history.json, summary.json,
best_validation.json, checkpoints/best.pt and checkpoints/last.pt. A verified
real-data smoke additionally saves reloaded_validation.json and verification.json.
Figures include editable JSON/CSV/NPZ inputs and PNG/PDF exports.

Phase 4 timing.json separates measured training/validation duration per completed
epoch. Summary/export durations accumulate across resumes. Annotation records
include geometric coverage diagnostics without semantic class information.
run_status.csv/json records blocked or missing runs separately from results;
results.csv/json includes only actually completed measured runs.

Each campaign writes campaign_config.json, preflight.json, campaign_status.json,
failures.json and actual completed results.csv/results.json. Final-test inference
requires frozen_protocol.json and explicit authorization. Scores for incomplete,
failed or unrun conditions are never filled in.

Historical results from the failed environment are not present. Their hashes and
reported observations remain historical in docs/recovery_checkpoint.json.
Generated model/data artifacts stay ignored. Preserve them separately during
migration. Do not overwrite completed runs.
