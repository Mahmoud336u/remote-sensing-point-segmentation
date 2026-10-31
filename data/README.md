# Data storage

Raw archive: data/raw/landcover.ai.v1.zip.
Verified extraction: data/landcover.ai.v1/.
Frozen scene split: data/manifests/splits.json.
Verified ImageNet encoder weights: data/cache/resnet18-f37072fd.pth.

These paths are Git-ignored to keep dataset and model binaries out of source
delivery. Preserve the exact manifest, archive/source checksums and run records
in a migration/snapshot even though they are ignored. A fresh clone does not
recover them. Never overwrite a recovered manifest to make its schema fit;
preserve it and explicitly migrate to a new file if necessary.

The original manifest with SHA-256
fcf16f88860546b54321840247e7c7c32a2db6306bc059df374f7374f5aad055
was not recovered. A reconstructed manifest must have its own recorded hash.

LandCover.ai data uses CC BY-NC-SA 4.0, separate from the code's MIT license.
Credit the dataset authors and Head Office of Geodesy and Cartography, Poland.
See docs/dataset_selection.md for citation and original sources.
