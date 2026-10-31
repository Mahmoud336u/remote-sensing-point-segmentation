"""Global-confusion evaluation on complete held-out masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, seed_everything
from .config import canonical_hash, file_sha256, resolve_path, write_json
from .dataset import SegmentationDataset
from .model import build_model

CLASS_NAMES = ["background", "building", "woodland", "water", "road"]


def metrics_from_confusion(confusion: np.ndarray | torch.Tensor) -> dict:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or np.any(cm < 0):
        raise ValueError("Expected square nonnegative confusion matrix")
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    union = support + predicted - tp
    dice_denominator = support + predicted
    iou = [float(tp[i] / union[i]) if union[i] else None for i in range(len(cm))]
    dice = [float(2 * tp[i] / dice_denominator[i]) if dice_denominator[i] else None
            for i in range(len(cm))]
    valid_iou = [v for v in iou if v is not None]
    valid_dice = [v for v in dice if v is not None]
    return {"miou": float(np.mean(valid_iou)) if valid_iou else None,
            "mean_dice": float(np.mean(valid_dice)) if valid_dice else None,
            "per_class_iou": iou, "per_class_dice": dice,
            "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None,
            "class_support": support.tolist(), "predicted_support": predicted.tolist(),
            "confusion_matrix": cm.tolist(), "valid_pixels": int(cm.sum()),
            "class_names": CLASS_NAMES[:len(cm)]}


def confusion_update(confusion: torch.Tensor, prediction: torch.Tensor,
                     target: torch.Tensor, valid: torch.Tensor) -> None:
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Prediction, target and validity shapes must agree")
    n = confusion.shape[0]
    selected = valid.bool() & (target >= 0) & (target < n)
    pred = prediction[selected]
    if torch.any((pred < 0) | (pred >= n)):
        raise ValueError("Predicted labels out of range")
    counts = torch.bincount((target[selected] * n + pred).long(), minlength=n * n)
    confusion += counts.reshape(n, n).to(confusion.device)


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device | str, classes: int = 5) -> dict:
    was_training = model.training
    model.eval()
    cm = torch.zeros((classes, classes), dtype=torch.int64)
    ce_sum, count = 0.0, 0
    try:
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            valid = batch["valid_mask"].to(device).bool() & (targets >= 0) & (targets < classes)
            logits = model(images)
            selected_logits = logits.permute(0, 2, 3, 1)[valid].float()
            selected_targets = targets[valid].long()
            if len(selected_targets):
                ce_sum += F.cross_entropy(selected_logits, selected_targets, reduction="sum").item()
                count += len(selected_targets)
            confusion_update(cm, logits.argmax(1).cpu(), targets.cpu(), valid.cpu())
    finally:
        model.train(was_training)
    metrics = metrics_from_confusion(cm.numpy())
    metrics["dense_ce"] = ce_sum / count if count else None
    return metrics


def evaluate_checkpoint(checkpoint: str | Path, split: str = "val", device: str = "cpu",
                        *, allow_final_test: bool = False, protocol: str | Path | None = None) -> dict:
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    cfg = state["config"]
    if canonical_hash(cfg) != state["config_hash"]:
        raise ValueError("Checkpoint configuration hash mismatch")
    if split == "test":
        if not allow_final_test or protocol is None:
            raise PermissionError("Test evaluation requires --allow-final-test and a frozen protocol")
        freeze = json.loads(Path(protocol).read_text())
        entry = [r for r in freeze["runs"] if r["checkpoint_sha256"] == file_sha256(checkpoint)]
        if (not entry or freeze["config_hash"] != state["config_hash"] or
                freeze["manifest_sha256"] != state["provenance"]["manifest_sha256"]):
            raise ValueError("Checkpoint/configuration is not in the frozen protocol")
    seed_everything(state["seed"], cfg["training"]["threads"])
    manifest = resolve_path(cfg, cfg["data"]["manifest"])
    if file_sha256(manifest) != state["provenance"]["manifest_sha256"]:
        raise ValueError("Manifest changed since training")
    dataset = SegmentationDataset(cfg, split, state["condition"], state["seed"],
                                  allow_test=allow_final_test)
    if split == "val" and dataset.subset_hash != state["provenance"]["val"]["subset_hash"]:
        raise ValueError("Validation subset changed")
    model = build_model(cfg, initialize=False).to(device)
    model.load_state_dict(state["model"])
    generator = torch.Generator().manual_seed(state["seed"])
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=0,
                        shuffle=False, generator=generator)
    metrics = evaluate_model(model, loader, device, cfg["model"]["classes"])
    return {"split": split, "condition": state["condition"], "seed": state["seed"],
            "epoch": state["epoch"], "checkpoint_sha256": file_sha256(checkpoint),
            "manifest_sha256": state["provenance"]["manifest_sha256"],
            "subset": dataset.provenance(), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a new metrics output path")
    result = evaluate_checkpoint(args.checkpoint, args.split, args.device,
                                 allow_final_test=args.allow_final_test, protocol=args.protocol)
    write_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
