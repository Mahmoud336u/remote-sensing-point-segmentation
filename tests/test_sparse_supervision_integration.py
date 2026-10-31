import copy
import torch
from torch import nn
from src.train import train_epoch


def test_empty_batch_preserves_buffers_optimizer_and_weights():
    model = nn.Sequential(nn.Conv2d(3, 5, 1), nn.BatchNorm2d(5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {"image": torch.randn(2, 3, 4, 4), "target": torch.zeros(2, 4, 4, dtype=torch.long),
             "point_mask": torch.ones(2, 4, 4, dtype=torch.bool)}
    train_epoch(model, [batch], optimizer, scaler, "cpu", 5)
    before = copy.deepcopy(model.state_dict())
    optim_before = copy.deepcopy(optimizer.state_dict())
    batch["point_mask"].zero_()
    batch["image"].fill_(float("nan"))
    metrics = train_epoch(model, [batch], optimizer, scaler, "cpu", 5)
    assert metrics["empty_batches"] == 1 and metrics["optimizer_updates"] == 0
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)
    for key, values in optim_before["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
