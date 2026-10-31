"""Editable chart data and PNG/PDF figures from actual run artifacts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/workspace/scratch/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from .checkpoint import load_checkpoint, seed_everything
from .config import file_sha256, write_json
from .dataset import SegmentationDataset
from .evaluate import CLASS_NAMES
from .model import build_model

COLORS = ["#6b7280", "#c67715", "#7c8528", "#3177ad", "#b65a89"]
CLASS_CMAP = ListedColormap(COLORS)
CLASS_CMAP.set_bad("#ffffff")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.titlesize": 12, "axes.labelsize": 10,
         "savefig.dpi": 180, "figure.facecolor": "white"}


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def training_curves(run: str | Path):
    run = Path(run)
    history = json.loads((run / "history.json").read_text())
    if not history:
        raise ValueError("No executed epochs to visualize")
    data = [{"epoch": h["epoch"], "training_loss": h["train"]["loss"],
             "validation_dense_ce": h["val"]["dense_ce"], "validation_miou": h["val"]["miou"]}
            for h in history]
    figures = run / "figures"
    write_json(figures / "training_curves.json", data)
    with (figures / "training_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    metadata = json.loads((run / "run.json").read_text())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        epochs = [r["epoch"] for r in data]
        axes[0].plot(epochs, [r["training_loss"] for r in data], "o-", color="#3177ad",
                     label="Training (supervised pixels)")
        axes[0].plot(epochs, [r["validation_dense_ce"] for r in data], "s--", color="#c67715",
                     label="Validation (all valid pixels)")
        axes[0].set(xlabel="Epoch", ylabel="Cross entropy", title="Training and validation loss")
        axes[0].legend(fontsize=8)
        axes[1].plot(epochs, [100 * r["validation_miou"] for r in data], "o-", color="#3177ad")
        axes[1].set(xlabel="Epoch", ylabel="mIoU (%)", ylim=(0, 100), title="Dense validation mIoU")
        for ax in axes:
            ax.set_xticks(epochs if len(epochs) <= 10 else [epochs[0], *epochs[4::5], epochs[-1]])
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(metadata["run_id"])
        save_figure(fig, figures / "training_curves")


@torch.inference_mode()
def examples(run: str | Path):
    run = Path(run)
    checkpoint = run / "checkpoints/best.pt"
    state = load_checkpoint(checkpoint)
    cfg = copy.deepcopy(state["config"])
    cfg["data"]["augment_d4"] = False
    seed_everything(state["seed"], cfg["training"]["threads"])
    model = build_model(cfg, initialize=False).eval()
    model.load_state_dict(state["model"])
    records = []
    for split in ("train", "val"):
        ds = SegmentationDataset(cfg, split, state["condition"], state["seed"])
        # Choose by valid area before inference, never by class labels or model performance.
        def valid_area(index):
            tile = ds.tiles[index]
            scene = ds.scenes[tile["scene"]]
            return min(ds.size, scene["height"] - tile["row"]) * min(ds.size, scene["width"] - tile["col"])
        index = max(range(len(ds)), key=valid_area)
        batch = ds[index]
        diagnostic = ds.diagnostic_dense(index)
        prediction = model(batch["image"][None]).argmax(1)[0].numpy()
        truth = diagnostic["dense_target"]
        valid = diagnostic["valid_mask"]
        error = np.ma.array((prediction != truth).astype(float), mask=~valid)
        panels = [diagnostic["rgb"].transpose(1, 2, 0),
                  np.ma.array(truth, mask=~valid), np.ma.array(prediction, mask=~valid), error]
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 4, figsize=(12, 4), layout="constrained")
            for i, (ax, panel, title) in enumerate(zip(
                    axes, panels, ["RGB / annotation points", "Dense diagnostic truth", "Prediction",
                                   "Error (orange = incorrect)"])):
                if i == 0:
                    ax.imshow(panel)
                    if split == "train" and state["condition"] != "dense_reference":
                        yy, xx = np.nonzero(batch["point_mask"].numpy())
                        ax.scatter(xx, yy, s=9, facecolors="none", edgecolors="white", linewidths=.6)
                elif i < 3:
                    ax.imshow(panel, cmap=CLASS_CMAP, vmin=0, vmax=4, interpolation="nearest")
                else:
                    ax.imshow(panel, cmap=ListedColormap(["#eeeeee", "#c67715"]), vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{split}: {batch['tile_id']} — {state['condition']}, seed {state['seed']}",
                         fontsize=11)
            fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)],
                       loc="outside lower center", ncol=5, frameon=False)
            save_figure(fig, run / "figures" / f"{split}_example")
        np.savez_compressed(run / "figures" / f"{split}_example.npz",
                            rgb=diagnostic["rgb"], target=truth, valid=valid,
                            prediction=prediction, points=batch["point_mask"].numpy())
        records.append({"split": split, "tile": diagnostic["tile"],
                        "selection_rule": "largest valid area in frozen subset; first tile breaks ties; chosen before inference",
                        "checkpoint_sha256": file_sha256(checkpoint)})
    write_json(run / "figures/examples.json",
               {"examples": records, "test_inference": False,
                "attribution": "LandCover.ai; imagery: Head Office of Geodesy and Cartography, Poland",
                "data_license": "CC BY-NC-SA 4.0"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    training_curves(args.run)
    examples(args.run)
