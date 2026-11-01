"""Deterministic sparse/dense training with safe, epoch-boundary resume."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint, seed_everything
from .config import CONDITIONS, canonical_hash, file_sha256, load_config, resolve_path, write_json
from .dataset import SegmentationDataset
from .evaluate import evaluate_model
from .losses import PartialCrossEntropyLoss, supervised_mask
from .model import build_model


def worker_seed(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def performance_clock(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def make_loader(dataset, cfg: dict, generator: torch.Generator, shuffle: bool):
    workers = cfg["training"]["num_workers"]
    kwargs = {"multiprocessing_context": "spawn", "timeout": 60} if workers else {}
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                      shuffle=shuffle, num_workers=workers, generator=generator,
                      worker_init_fn=worker_seed, persistent_workers=False, **kwargs)


def train_epoch(model, loader, optimizer, scaler, device, classes: int, *, dense: bool = False,
                amp: bool = False) -> dict:
    model.train()
    criterion = PartialCrossEntropyLoss()
    weighted_loss, annotated, updates, skipped, overflow = 0., 0, 0, 0, 0
    for batch in loader:
        targets = batch["target"].to(device)
        points = batch["point_mask"].to(device)
        count = int(supervised_mask(targets, points, classes).sum())
        if count == 0:
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(device).type, enabled=amp):
            logits = model(batch["image"].to(device))
            if dense:
                loss = F.cross_entropy(logits, targets, ignore_index=255)
            else:
                loss = criterion(logits, targets, points)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite selected training loss")
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            overflow += 1
        else:
            updates += 1
        weighted_loss += loss.item() * count
        annotated += count
    return {"loss": weighted_loss / annotated if annotated else None,
            "annotated_pixels_seen": annotated, "optimizer_updates": updates,
            "empty_batches": skipped, "amp_overflow_batches": overflow}


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this run") from exc
        yield


def source_fingerprints(cfg: dict) -> dict:
    root = Path(cfg["_repo_root"])
    paths = [p for sub in ("src", "experiments") for p in sorted((root / sub).glob("*.py"))]
    paths += [root / "requirements-phase3-cpu.lock"]
    return {str(p.relative_to(root)): file_sha256(p) for p in paths}


def train(cfg: dict, condition: str, seed: int, *, output_dir: str | Path | None = None,
          resume: bool = False, stop_after_epochs: int | None = None) -> Path:
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    directory = Path(output_dir) if output_dir else resolve_path(
        cfg, cfg["experiment"]["output_root"]) / condition / f"seed_{seed}"
    with run_lock(directory):
        return _train(cfg, condition, seed, directory, resume, stop_after_epochs)


def _train(cfg, condition, seed, directory, resume, stop_after_epochs):
    run_path = directory / "run.json"
    controls = {"config": cfg, "condition": condition, "seed": seed}
    config_hash = canonical_hash(cfg)
    if run_path.exists():
        prior = json.loads(run_path.read_text())
        if prior["status"] == "completed":
            raise FileExistsError("Completed output directory is protected")
        if not resume:
            raise FileExistsError("Existing run requires explicit --resume")
        if prior["controls_hash"] != canonical_hash(controls):
            raise ValueError("Resume requires identical configuration, condition and seed")
        if prior["output_directory"] != str(directory.resolve()):
            raise ValueError("Resume requires the original output directory")
        for name in ("last.pt", "best.pt"):
            if not (directory / "checkpoints" / name).exists():
                raise FileNotFoundError("Resume requires last.pt and preserved best.pt")
    elif resume:
        raise FileNotFoundError("Cannot resume a new directory")
    elif any(p.name != ".run.lock" for p in directory.iterdir()):
        raise FileExistsError("Output directory contains unrelated files")
    seed_everything(seed, cfg["training"]["threads"])
    requested = cfg["training"]["device"]
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime = {"device": str(device), "torch": str(torch.__version__),
               "numpy": str(np.__version__), "threads": cfg["training"]["threads"]}
    amp = bool(cfg["training"]["amp"])
    if amp and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA")
    train_data = SegmentationDataset(cfg, "train", condition, seed)
    val_data = SegmentationDataset(cfg, "val", condition, seed)
    provenance = {"manifest_sha256": file_sha256(train_data.manifest_path),
                  "train": train_data.provenance(), "val": val_data.provenance(),
                  "source_fingerprints": source_fingerprints(cfg)}
    last = load_checkpoint(directory / "checkpoints/last.pt") if resume else None
    if last and last["provenance"] != provenance:
        raise ValueError("Resume source/subset/manifest provenance changed")
    if last and (last["config_hash"] != config_hash or last["condition"] != condition
                 or last["seed"] != seed or last["output_directory"] != str(directory.resolve())):
        raise ValueError("Checkpoint resume controls differ from the requested run")
    if last and last["runtime"] != runtime:
        raise ValueError("Exact resume requires the same device and numerical runtime")
    model = build_model(cfg, initialize=not resume).to(device)
    t = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["epochs"], eta_min=t["min_lr"]) if t["scheduler"] == "cosine"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    loader = make_loader(train_data, cfg, generator, True)
    val_loader = make_loader(val_data, cfg, val_generator, False)
    history, timings, best, best_epoch, start = [], [], -1., 0, 1
    if last:
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        restore_rng(last["rng"], generator)
        history, best, best_epoch = last["history"], last["best_metric"], last["best_epoch"]
        timings = last.get("timings", [])
        start = last["epoch"] + 1
    else:
        write_json(directory / "config.json", cfg)
        records = [train_data.annotation_record(i) for i in range(len(train_data))]
        cost = sum(r["annotated_pixels"] for r in records)
        pixels = sum(r["valid_pixels"] for r in records)
        coverage = {}
        for key in ("occupied_cell_fraction", "mean_nearest_neighbor_normalized"):
            values = [r["spatial_coverage"][key] for r in records
                      if r["spatial_coverage"][key] is not None]
            coverage[key] = float(np.mean(values)) if values else None
            coverage[f"{key}_tiles"] = len(values)
        write_json(directory / "annotations.json",
                   {"records": records, "unique_annotations": cost, "valid_training_pixels": pixels,
                    "annotation_fraction": cost / pixels,
                    "spatial_coverage_mean_over_tiles": coverage,
                    "seed_algorithm": train_data.provenance()["seed_algorithm"]})
    run = {"run_id": f"{cfg['profile']}/{condition}/seed_{seed}",
           "status": "running", "controls_hash": canonical_hash(controls), "config_hash": config_hash,
           "condition": condition, "seed": seed, "profile": cfg["profile"],
           "output_directory": str(directory.resolve()),
           "runtime": runtime,
           "device": str(device), "provenance": provenance}
    write_json(run_path, run)
    began = time.perf_counter()
    try:
        end = min(t["epochs"], stop_after_epochs) if stop_after_epochs else t["epochs"]
        stopped_early = False
        for epoch in range(start, end + 1):
            train_data.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]
            train_began = performance_clock(device)
            train_metrics = train_epoch(model, loader, optimizer, scaler, device,
                                        cfg["model"]["classes"], dense=condition == "dense_reference",
                                        amp=amp)
            train_ended = performance_clock(device)
            val_metrics = evaluate_model(model, val_loader, device, cfg["model"]["classes"])
            validation_ended = performance_clock(device)
            if val_metrics["miou"] is None:
                raise RuntimeError("Validation contains no valid pixels")
            history.append({"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics})
            timings.append({"epoch": epoch, "training_seconds": train_ended - train_began,
                            "validation_seconds": validation_ended - train_ended})
            improved = val_metrics["miou"] > best
            if improved:
                best, best_epoch = val_metrics["miou"], epoch
            if train_metrics["optimizer_updates"]:
                scheduler.step()
            state = {"schema_version": 1, "epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "rng": capture_rng(generator), "history": history,
                     "best_metric": best, "best_epoch": best_epoch, "config": cfg,
                     "config_hash": config_hash, "condition": condition, "seed": seed,
                     "output_directory": str(directory.resolve()),
                     "runtime": runtime,
                     "timings": timings,
                     "provenance": provenance}
            if improved:
                save_checkpoint(directory / "checkpoints/best.pt", state)
                write_json(directory / "best_validation.json", {"epoch": epoch, "metrics": val_metrics})
            save_checkpoint(directory / "checkpoints/last.pt", state)
            write_json(directory / "history.json", history)
            write_json(directory / "timing.json", timings)
            print(json.dumps({"run": run["run_id"], "epoch": epoch,
                              "training_loss": train_metrics["loss"], "val_miou": val_metrics["miou"]}),
                  flush=True)
            patience = t.get("early_stopping_patience")
            if patience and epoch - best_epoch >= patience:
                stopped_early = True
                break
        completed = bool(history) and (history[-1]["epoch"] == t["epochs"] or stopped_early)
        run["status"] = "completed" if completed else "incomplete"
        run["best_epoch"] = best_epoch
        run["best_validation_miou"] = best
        run["session_training_seconds"] = time.perf_counter() - began
        run["training_seconds"] = sum(row["training_seconds"] for row in timings)
        run["validation_seconds"] = sum(row["validation_seconds"] for row in timings)
        run["timing_scope"] = "completed epochs across resumes; includes loader time; excludes initialization/checkpoint I/O"
        write_json(run_path, run)
        write_json(directory / "summary.json",
                   {**run, "epochs_completed": history[-1]["epoch"] if history else 0,
                    "optimizer_updates": sum(h["train"]["optimizer_updates"] for h in history),
                    "checkpoint_sha256": file_sha256(directory / "checkpoints/best.pt"),
                    "stopped_early": stopped_early})
    except BaseException as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(run_path, run)
        raise
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(train(load_config(args.config), args.condition, args.seed,
                output_dir=args.output_dir, resume=args.resume))


if __name__ == "__main__":
    main()
