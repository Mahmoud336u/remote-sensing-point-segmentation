"""Summarize completed runs without treating smoke scores as campaign evidence."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config, resolve_path, write_json
from src.evaluate import CLASS_NAMES
from src.visualization import STYLE, save_figure
from .run_experiments import export

ORDER = ["uniform_10", "uniform_50", "uniform_200", "stratified_50", "dense_reference"]
COLORS = {"uniform_10": "#3177ad", "uniform_50": "#3177ad", "uniform_200": "#3177ad",
          "stratified_50": "#c67715", "dense_reference": "#6b7280"}


def summarize(rows: list[dict], split: str) -> list[dict]:
    summary = []
    for condition in ORDER:
        group = [r for r in rows if r["condition"] == condition and r["split"] == split]
        if not group:
            continue
        entry = {"condition": condition, "split": split, "n": len(group),
                 "seeds": [r["seed"] for r in group],
                 "annotation_fraction": statistics.mean(r["annotation_fraction"] for r in group)}
        for key in ["miou", "mean_dice", "pixel_accuracy", *[f"iou_class_{i}" for i in range(5)]]:
            values = [r[key] for r in group if r[key] is not None]
            entry[key] = statistics.mean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) >= 2 else None
            entry[f"{key}_n"] = len(values)
        summary.append(entry)
    return summary


def analyze(cfg: dict, split: str = "val") -> dict:
    if "smoke" in cfg["profile"]:
        raise ValueError("Smoke observations cannot be analyzed as campaign results")
    result = export(cfg)
    rows = [r for r in result["completed_rows"] if r["split"] == split]
    if not rows:
        raise ValueError(f"No completed {split} campaign rows; no figures or findings generated")
    summary = summarize(rows, split)
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    figures = root / "figures"
    write_json(root / f"summary_{split}.json", summary)
    with (root / f"summary_{split}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    for seed in sorted({r["seed"] for r in rows}):
        pair = {r["condition"]: r for r in rows if r["seed"] == seed}
        if "uniform_50" in pair and "stratified_50" in pair:
            paired.append({"seed": seed, "stratified_minus_uniform_miou":
                           pair["stratified_50"]["miou"] - pair["uniform_50"]["miou"]})
    write_json(root / f"paired_differences_{split}.json", paired)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, key, title in zip(axes, ["miou", "mean_dice"], ["mIoU", "Mean Dice"]):
            for r in summary:
                if r["condition"] == "dense_reference":
                    continue
                error = r[f"{key}_std"]
                ax.errorbar(r["annotation_fraction"] * 100, r[key] * 100,
                            yerr=None if error is None else 100 * error,
                            marker="s" if r["condition"] == "stratified_50" else "o",
                            color=COLORS[r["condition"]], capsize=4,
                            label=f"{r['condition']} (n={r['n']})")
            ax.set(xlabel="Unique annotated / valid training pixels (%)", ylabel=f"{title} (%)",
                   title=f"{split}: {title} vs annotation density", ylim=(0, 100))
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(fontsize=8)
        fig.supxlabel("Error bars show sample SD across completed seeds; no bar for one seed.",
                      fontsize=8)
        save_figure(fig, figures / f"annotation_density_{split}")
        pair_summary = [r for r in summary if r["condition"] in ("uniform_50", "stratified_50")]
        if pair_summary:
            fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
            for i, r in enumerate(pair_summary):
                sd = r["miou_std"]
                ax.bar(i, r["miou"] * 100, color=COLORS[r["condition"]],
                       yerr=None if sd is None else sd * 100, capsize=4,
                       hatch="//" if r["condition"] == "stratified_50" else None)
            ax.set_xticks(range(len(pair_summary)),
                          [f"{r['condition']}\nn={r['n']}" for r in pair_summary])
            ax.set(ylabel="mIoU (%)", ylim=(0, 100), title=f"{split}: sampling strategy at 50 points")
            save_figure(fig, figures / f"sampling_strategy_{split}")
        fig, axes = plt.subplots(1, 5, figsize=(13, 4), sharey=True, layout="constrained")
        for i, ax in enumerate(axes):
            for j, r in enumerate(summary):
                value, sd = r[f"iou_class_{i}"], r[f"iou_class_{i}_std"]
                if value is not None:
                    ax.errorbar(j, value * 100, yerr=None if sd is None else sd * 100,
                                fmt="s" if "stratified" in r["condition"] else "o",
                                color=COLORS[r["condition"]], capsize=3)
            ax.set_title(CLASS_NAMES[i])
            ax.set_xticks(range(len(summary)),
                          [r["condition"].replace("uniform_", "U").replace("stratified_", "S")
                           .replace("dense_reference", "Dense") for r in summary], rotation=45)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Class IoU (%)")
        fig.suptitle(f"{split}: per-class IoU; undefined classes omitted")
        save_figure(fig, figures / f"per_class_iou_{split}")
    return {"summary": summary, "paired_differences": paired,
            "incomplete_runs": [r for r in result["runs"] if r["status"] != "completed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    print(json.dumps(analyze(load_config(args.config), args.split), indent=2))
