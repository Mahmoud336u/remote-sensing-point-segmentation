"""U-Net/ResNet-18 with explicit, checksum-verified ImageNet initialization."""
from __future__ import annotations

from pathlib import Path
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F

from .config import file_sha256, resolve_path

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 5,
                 decoder_channels: tuple = (256, 128, 64, 32, 16)):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                            in_channels=in_channels, classes=classes,
                            decoder_channels=tuple(decoder_channels), activation=None)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("Expected [B,C,H,W] images")
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, (-width) % 32, 0, (-height) % 32))
        return self.unet(padded)[..., :height, :width]


def build_model(cfg: dict, *, initialize: bool = True) -> SegmentationModel:
    mc = cfg["model"]
    if mc.get("encoder_name", "resnet18") != "resnet18":
        raise ValueError("Only the frozen ResNet-18 architecture is supported")
    model = SegmentationModel(mc["in_channels"], mc["classes"], tuple(mc["decoder_channels"]))
    if initialize and mc["pretrained"]:
        path = resolve_path(cfg, "data/cache/resnet18-f37072fd.pth")
        try:
            from scripts.prepare_data import download_verified
            download_verified(WEIGHTS_URL, path, WEIGHTS_SHA256)
            if mc["in_channels"] != 3:
                raise ValueError("ImageNet loading requires three input channels")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.unet.encoder.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Requested ImageNet initialization could not be loaded: {exc}") from exc
    return model
