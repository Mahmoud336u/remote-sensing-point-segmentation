# Verification

Run python -m scripts.verify_tests from the project root using .venv. This runs
pytest, preserves its exit status, rejects zero successful tests, and saves JUnit
plus a summary bound to source/test/config hashes.

Coverage includes full/sparse/empty Partial CE, low-precision arithmetic,
unselected NaNs/gradients, nested budgets, stratified cell deficits, D4 alignment,
GeoTIFF inventory/checksums/geographic separation, padding, sparse-only targets,
class-blind fixed subsets, global metrics, configurable model shape and backward
passes, empty-batch optimizer/BatchNorm safety, RNG restoration, exact interruption
and resume, real U-Net round trip on synthetic rasters, two spawn workers, and
campaign failure/protocol/overwrite controls.

Fixtures are synthetic rasters under temporary directories. Synthetic final-test
gate tests do not perform LandCover.ai inference. CUDA checks skip explicitly
without hardware. Local IPC is needed for the two-worker test and Jupyter.

The old 400 passed/4 skipped total is historical, not the current suite. Read the
current JSON/JUnit outputs for actual counts. A real-data smoke is a separate
required gate and cannot be certified by these synthetic fixtures.
