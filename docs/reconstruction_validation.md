# Reconstruction verification

The current source-bound suite completed with 114 passed, one CUDA-only skipped,
zero failures and zero errors. The report is results/verification/reconstruction-pytest.xml;
its fingerprints and exact counts are in current_suite.json. Synthetic raster
fixtures exercise full U-Net training/reload, multiprocessing and campaign gates.
No LandCover.ai campaign or final-test inference has executed.

The full 77-package CPU lock was installed into a fresh isolated virtual
environment with hash verification. Package consistency, imports and a padded
U-Net forward pass passed. The saved setup script also ran successfully.
Official ImageNet weights match the recorded full SHA-256.

The notebook executed all five code cells and retained its sampling figure. The
embedded plot and generated synthetic-run loss/qualitative figures were visually
inspected. A full notebook-page viewer was unavailable; notebook structure,
execution counts, text outputs and embedded plot content were checked. For a
full-page visual review, open notebooks/assessment_demo.ipynb in JupyterLab using
the project environment.

Official archive verification, real-data split/audit, real-data smoke/reload and
Phase 4 are unrun because dataset access is blocked. Historical verification is
not substituted. Earlier phase locks were generated constrained to the full
locked environment; only the complete Phase 3 environment received an independent
fresh-install verification.
