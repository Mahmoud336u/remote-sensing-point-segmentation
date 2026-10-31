"""Partial cross entropy with selection before floating-point arithmetic."""
from __future__ import annotations

import torch
from torch import nn


def supervised_mask(targets: torch.Tensor, point_mask: torch.Tensor,
                    classes: int, ignore_index: int = 255) -> torch.Tensor:
    if targets.shape != point_mask.shape or targets.ndim != 3:
        raise ValueError("targets and point_mask must have matching [B,H,W] shape")
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("targets must contain integer labels")
    if point_mask.dtype != torch.bool and not torch.all((point_mask == 0) | (point_mask == 1)):
        raise ValueError("point_mask must be binary")
    return point_mask.bool() & (targets != ignore_index) & (targets >= 0) & (targets < classes)


class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                point_mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
            raise ValueError("Expected logits [B,C,H,W] and targets/mask [B,H,W]")
        valid = supervised_mask(targets, point_mask, logits.shape[1], self.ignore_index)
        selected = logits.permute(0, 2, 3, 1)[valid]
        if selected.dtype in (torch.float16, torch.bfloat16):
            selected = selected.float()
        if not selected.numel():
            # The empty selection stays connected to logits without reading NaNs.
            return selected.sum()
        labels = targets[valid].long()
        return (torch.logsumexp(selected, dim=1) - selected.gather(1, labels[:, None])[:, 0]).mean()
