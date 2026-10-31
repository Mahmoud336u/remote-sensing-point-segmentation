# Architecture

The pipeline consists of an immutable scene inventory, geographic grouping,
class-blind window selection, fixed sparse annotations, a raster-window dataset,
U-Net training, dense validation selection, checkpoint restoration and gated test
evaluation.

The model is segmentation_models_pytorch 0.6.0 U-Net with a ResNet-18 encoder,
decoder widths [256,128,64,32,16], RGB input and five raw logits per pixel.
The standard configuration has 14,328,789 parameters. The factory supports other
input/output channel counts for isolated tests; LandCover.ai configurations
require RGB/five classes. Inputs are padded to a multiple of 32 and logits are
cropped without interpolating labels.

U-Net skip connections retain spatial detail. ResNet-18 is an established,
moderate-size encoder suitable for limited compute. ImageNet initialization is
useful for RGB imagery but comes from natural images and has an aerial-image
domain mismatch. Requested initialization failures raise an explicit error.
Checkpoint restoration constructs the architecture without downloading weights.

The sparse training batch contains image, target, point mask, valid mask and tile
ID. Unselected targets are 255. Dense references are exposed only through the
explicit diagnostic interface or dense_reference condition. Validation/test
datasets provide complete masks without augmentation.

AdamW and a cosine schedule are fixed within a comparison. Loader shuffling uses
its own reproducible generator; augmentation depends on epoch and tile identity.
Checkpoints hold model/optimizer/scheduler/scaler/RNG state plus configuration,
source fingerprints, subset identity, history and selected validation epoch.

The experiment runner separates planning, preflight, execution, freezing,
final-test access and export. Failed/incomplete runs are reported, never assigned
invented scores. A full final assessment report is a separate deliverable.
