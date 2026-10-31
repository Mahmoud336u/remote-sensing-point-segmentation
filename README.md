# Point-supervised remote-sensing segmentation

Reproducible LandCover.ai experiments with fixed sparse annotations, Partial
Cross-Entropy and a U-Net with a ResNet-18 encoder.

**Recovery status:** the previous uncommitted implementation and data were not
restored. This implementation was reconstructed from the recovery checkpoint
supplied on 2026-10-31. It has new verification evidence; it is not a byte-identical
recovery. See [PROJECT_STATE.md](PROJECT_STATE.md) and [recovery notes](docs/recovery.md).
Historical smoke scores are not campaign results.

## Install and validate

Use Python 3.12 on Linux x86-64. The full CPU environment is pinned with artifact
hashes in [requirements-phase3-cpu.lock](requirements-phase3-cpu.lock).

    bash scripts/setup.sh
    source .venv/bin/activate
    python -m scripts.inspect_environment
    python -m scripts.verify_tests
    python scripts/validate_notebook.py

Setup uses a writable cache at /workspace/scratch/uv-cache. The tested environment
has four CPU cores and 32 GiB RAM, without CUDA. Multiprocessing tests and Jupyter
require local IPC; a command sandbox that blocks sockets may need the platform's
normal reviewed execution route. No long-lived service is required. Cloud tasks
already use isolated environments: use the existing checkout; do not create a
worktree unless explicitly requested.

The lock contains CPU PyTorch packages from download.pytorch.org and other
packages from PyPI. TLS and package-hash verification remain enabled. The project
does not need an API token for public data or initialization weights.

## Data and real-data gate

Dataset: [LandCover.ai v1](https://landcover.ai.linuxpolska.com/), original aerial
RGB GeoTIFFs and dense masks. The dataset is **CC BY-NC-SA 4.0**, separately from
this repository's MIT code license. Imagery attribution: Head Office of Geodesy
and Cartography, Poland.

    python -m scripts.prepare_data --config configs/smoke.yaml --download
    python -m src.train --config configs/smoke.yaml --condition uniform_50 --seed 17
    python -m scripts.verify_smoke --run results/reconstructed_smoke/uniform_50/seed_17
    python -m src.visualization --run results/reconstructed_smoke/uniform_50/seed_17

Download requires access to landcover.ai.linuxpolska.com. A proxy denial must be
resolved in the recovery environment's network settings; it is not a reason to
disable TLS or bypass the proxy. The archive's full SHA-256 is checked before
extraction. Existing data directories and manifests are protected against
accidental replacement.

The original frozen manifest is missing. A newly generated manifest is explicitly
marked reconstructed and may assign different scenes than the historical split.
Keep it with the experiment artifacts. Do not claim its hash, annotation
coordinates or trained weights match the lost originals.

A deliberate second smoke run needs a new output directory:

    python -m src.train --config configs/smoke.yaml --condition uniform_50 --seed 17 \
      --output-dir results/reconstructed_smoke_rerun/uniform_50/seed_17

## Campaigns

| Profile | Epochs | Batch | Workers | Seeds | Train/val/test tile caps |
| --- | ---: | ---: | ---: | --- | --- |
| default | 30 | 8 | 4 | 17, 29, 43 | none |
| cpu_pilot | 10 | 2 | 0 | 17 | 128 / 32 / 32 |
| cpu_paired | 10 | 2 | 0 | 17, 29, 43 | 128 / 32 / 32 |
| smoke | 2 | 2 | 0 | 17 | 8 / 4 / 4 |

cpu_paired is a separate profile. The original main and CPU-pilot controls are
preserved. It includes the four sparse conditions; dense_reference remains an
optional, separately planned comparison.

    python -m experiments.run_experiments --config configs/cpu_paired.yaml
    python -m experiments.run_experiments --config configs/cpu_paired.yaml --preflight
    python -m experiments.run_experiments --config configs/cpu_paired.yaml --execute

The default command only plans. Execution requires current passing tests and a
verified real-data smoke/checkpoint reload. It prioritizes one seed for all sparse
conditions, then additional paired seeds, then any configured dense references.
Each run saves immutable configuration, fixed annotation coordinates, source and
subset provenance, count-weighted losses, dense validation metrics, optimizer/RNG
state and best/last checkpoints. Completed runs are protected. Interrupted runs
require the same configuration, code, manifest, subsets, output directory and
preserved checkpoints.

    python -m src.train --config configs/cpu_paired.yaml --condition uniform_50 \
      --seed 17 --resume

After all planned runs complete, freeze the protocol and selected checkpoints
before final test inference:

    python -m experiments.run_experiments --config configs/cpu_paired.yaml --freeze
    python -m experiments.run_experiments --config configs/cpu_paired.yaml \
      --final-test --allow-final-test
    python -m experiments.analyze_results --config configs/cpu_paired.yaml --split test

Result CSV/JSON, sample SDs where at least two runs exist, paired differences and
PNG/PDF figures are written beneath the profile's output root. Each plot has
editable JSON/CSV data. Per-run qualitative predictions and error maps use the
tile with the largest valid area in the frozen subset (first tile breaks ties),
selected before inference:

    python -m src.visualization --run results/cpu_paired/uniform_50/seed_17

## Methodology and limits

- Original scenes are grouped before tiling using footprint distance <=256 m,
  including transitive connections and identical image hashes.
- Native 256-pixel windows use RGB/255 and ImageNet normalization. Background
  class 0 is valid; padding and unselected labels use 255.
- Uniform 10/50/200 annotations are nested for a fixed seed. Stratified sampling
  uses spatial cells without looking at semantic labels. Points remain fixed
  across epochs; D4 augmentation moves all aligned arrays together.
- Sparse training exposes only selected target labels. The dense diagnostic
  interface and dense_reference are explicit.
- Partial CE selects annotated logits before arithmetic and averages over the
  total valid point count. Empty batches skip both forward and optimizer steps.
- Selection uses dense validation mIoU, earliest exact tie. Test evaluation is
  explicitly gated. Metrics pool one confusion matrix over all valid pixels.

See [architecture](docs/architecture.md), [methodology](docs/methodology_contracts.md),
[experimental plan](docs/experimental_plan.md), [data notes](data/README.md), and
[assessment notebook](notebooks/assessment_demo.ipynb). A final technical
assessment report is outside Phase 4 and has not been written.
