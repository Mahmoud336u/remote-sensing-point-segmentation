import pytest
import torch
from src.model import SegmentationModel, build_model


@pytest.mark.parametrize("shape", [(64, 64), (65, 97), (256, 256)])
def test_shape_and_parameter_count(shape):
    model = SegmentationModel().eval()
    assert sum(p.numel() for p in model.parameters()) == 14328789
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, *shape))
    assert output.shape == (1, 5, *shape)


def test_channels_and_backward():
    model = SegmentationModel(in_channels=4, classes=3)
    x = torch.randn(2, 4, 64, 64)
    output = model(x)
    output.square().mean().backward()
    assert output.shape == (2, 3, 64, 64)
    assert model.unet.segmentation_head[0].weight.grad is not None


def test_pretrained_failure_is_explicit(cfg, monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("unavailable")
    monkeypatch.setattr("scripts.prepare_data.download_verified", fail)
    cfg["model"]["pretrained"] = True
    with pytest.raises(RuntimeError, match="Requested ImageNet"):
        build_model(cfg)
    # Checkpoint reconstruction must never touch the initialization source.
    build_model(cfg, initialize=False)
