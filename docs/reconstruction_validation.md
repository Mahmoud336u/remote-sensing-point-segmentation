# Reconstruction verification

The initial recovery suite at commit 9282366 completed with 114 passed and one
CUDA-only skipped. The Phase 4 suite completed with 117 passed and one skip.
The final Phase 5 suite completes with **131 passed, one CUDA-only skipped,
zero failures and zero errors**. The report is results/verification/reconstruction-pytest.xml;
its fingerprints and exact counts are in current_suite.json. Synthetic raster
fixtures exercise full U-Net training/reload, multiprocessing and campaign gates.
No LandCover.ai campaign or final-test inference has executed.

The full 77-package CPU lock was installed into a fresh isolated virtual
environment with hash verification. Package consistency, imports and a padded
U-Net forward pass passed. The saved setup script also ran successfully.
Official ImageNet weights match the recorded full SHA-256.

The final notebook executes ten code cells and retains its labelled synthetic
sampling figure. Structure, all execution outputs and the embedded figure were
inspected. The offline HTML is self-contained; full-page Chromium screenshot
attempts timed out and are not claimed as completed visual validation. Open the
notebook in JupyterLab for a full-page interactive review. Real-data sections
explicitly report missing artifacts. Earlier temporary synthetic loss/qualitative
figures exercised plotting code but are not campaign evidence or package results.

The seven-page technical report was rendered with Pandoc/XeLaTeX. All pages,
equations, table alignment and the conceptual workflow figure were visually
reviewed; text bounding boxes lie within each page. Report rendering evidence is
results/verification/report_render.json. Submission integrity checks are separate
from mandatory scientific completion.

Official archive verification, real-data split/audit, real-data smoke/reload and
Phase 4 are unrun because dataset access is blocked. Historical verification is
not substituted. Earlier phase locks were generated constrained to the full
locked environment; only the complete Phase 3 environment received an independent
fresh-install verification.
