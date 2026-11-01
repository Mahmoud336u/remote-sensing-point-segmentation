"""Plan, execute, resume and freeze controlled segmentation campaigns."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import torch

from src.checkpoint import load_checkpoint
from src.config import CONDITIONS, canonical_hash, file_sha256, load_config, resolve_path, write_json
from src.data_manifest import load_manifest
from src.evaluate import evaluate_checkpoint
from src.train import source_fingerprints, train
from scripts.verify_tests import verification_fingerprints

EXPERIMENT_IDS = {"uniform_10": "A1", "uniform_50": "A2/B1", "uniform_200": "A3",
                  "stratified_50": "B2", "dense_reference": "Dense"}


def experiment_fields(condition: str) -> dict:
    sampling, points = CONDITIONS[condition]
    return {"experiment_id": EXPERIMENT_IDS[condition], "sampling": sampling,
            "points_per_tile": points if points is not None else "all"}


def plan(cfg: dict) -> list[dict]:
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    conditions, seeds = cfg["experiment"]["conditions"], cfg["experiment"]["seeds"]
    sparse = [c for c in conditions if c != "dense_reference"]
    order = [(s, c) for s in seeds for c in sparse]
    if "dense_reference" in conditions:
        order += [(s, "dense_reference") for s in seeds]
    return [{"run_id": f"{cfg['profile']}/{c}/seed_{s}", "condition": c, "seed": s,
             **experiment_fields(c),
             "directory": str(root / c / f"seed_{s}")} for s, c in order]


def preflight(cfg: dict) -> dict:
    repo = Path(cfg["_repo_root"])
    tests = json.loads((repo / "results/verification/current_suite.json").read_text())
    if tests["status"] != "passed" or tests["fingerprints"] != verification_fingerprints(repo):
        raise RuntimeError("A passing current-source test suite is required")
    if tests["junit_sha256"] != file_sha256(repo / "results/verification/reconstruction-pytest.xml"):
        raise RuntimeError("Test evidence changed")
    smoke = json.loads((repo / "results/verification/real_data_smoke.json").read_text())
    if smoke["status"] != "passed" or not smoke["real_data"] or not smoke["exact_reload"]:
        raise RuntimeError("Verified real-data training and checkpoint reload are required")
    if smoke["source_fingerprints"] != source_fingerprints(cfg):
        raise RuntimeError("Source changed after real-data smoke")
    manifest_path = resolve_path(cfg, cfg["data"]["manifest"])
    manifest = load_manifest(manifest_path, resolve_path(cfg, cfg["data"]["root"]), verify_hashes=True)
    if file_sha256(manifest_path) != smoke["manifest_sha256"]:
        raise RuntimeError("Manifest differs from verified smoke")
    if manifest["source"]["sha256"] != cfg["data"]["archive_sha256"]:
        raise RuntimeError("Campaign data source differs")
    return {"status": "passed", "config_hash": canonical_hash(cfg),
            "manifest_sha256": file_sha256(manifest_path),
            "source_fingerprints": source_fingerprints(cfg), "smoke": smoke,
            "planned_runs": plan(cfg)}


def collect(cfg: dict) -> dict:
    rows, statuses = [], []
    failure_path = resolve_path(cfg, cfg["experiment"]["output_root"]) / "failures.json"
    failures = {f["run_id"]: f["error"] for f in json.loads(failure_path.read_text())} if failure_path.exists() else {}
    for job in plan(cfg):
        directory = Path(job["directory"])
        path = directory / "run.json"
        run = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
        if job["run_id"] in failures and run["status"] != "completed":
            run = {**run, "status": "failed", "error": failures[job["run_id"]]}
        statuses.append({**job, "status": run["status"], "error": run.get("error")})
        if run["status"] != "completed":
            continue
        if run["config_hash"] != canonical_hash(cfg):
            raise ValueError("Completed run has a different campaign configuration")
        checkpoint = directory / "checkpoints/best.pt"
        best = json.loads((directory / "best_validation.json").read_text())
        annotation = json.loads((directory / "annotations.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        for split, result in [("val", best)]:
            row = {"run_id": job["run_id"], "condition": job["condition"], "seed": job["seed"],
                   **experiment_fields(job["condition"]),
                   "split": split, "best_epoch": best["epoch"],
                   "annotation_fraction": annotation["annotation_fraction"],
                   "unique_annotations": annotation["unique_annotations"],
                   "valid_training_pixels": annotation["valid_training_pixels"],
                   "training_seconds": summary.get("training_seconds"),
                   "validation_seconds": summary.get("validation_seconds"),
                   "occupied_cell_fraction": annotation.get("spatial_coverage_mean_over_tiles", {}).get("occupied_cell_fraction"),
                   "mean_nearest_neighbor_normalized": annotation.get("spatial_coverage_mean_over_tiles", {}).get("mean_nearest_neighbor_normalized"),
                   "checkpoint_sha256": file_sha256(checkpoint)}
            row.update({k: result["metrics"][k] for k in ["miou", "mean_dice", "pixel_accuracy", "dense_ce"]})
            for i, value in enumerate(result["metrics"]["per_class_iou"]):
                row[f"iou_class_{i}"] = value
            rows.append(row)
        test_path = directory / "final_test.json"
        if test_path.exists():
            result = json.loads(test_path.read_text())
            if result["checkpoint_sha256"] != file_sha256(checkpoint):
                raise ValueError("Final-test metrics refer to a different checkpoint")
            row = {**row, "split": "test"}
            row.update({k: result["metrics"][k] for k in ["miou", "mean_dice", "pixel_accuracy", "dense_ce"]})
            row.update({f"iou_class_{i}": v for i, v in enumerate(result["metrics"]["per_class_iou"])})
            rows.append(row)
    return {"config_hash": canonical_hash(cfg), "runs": statuses, "completed_rows": rows}


def export(cfg: dict) -> dict:
    result = collect(cfg)
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    write_json(root / "campaign_status.json", result)
    write_json(root / "run_status.json", result["runs"])
    with (root / "run_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(result["runs"])
    if result["completed_rows"]:
        write_json(root / "results.json", result["completed_rows"])
        with (root / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["completed_rows"][0]))
            writer.writeheader()
            writer.writerows(result["completed_rows"])
    return result


def execute(cfg: dict) -> dict:
    checks = preflight(cfg)
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    snapshot = root / "campaign_config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise FileExistsError("Campaign configuration is immutable")
    write_json(snapshot, cfg)
    write_json(root / "preflight.json", checks)
    failures = []
    write_json(root / "failures.json", failures)
    for job in plan(cfg):
        path = Path(job["directory"]) / "run.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous["status"] == "completed":
            if previous["config_hash"] != canonical_hash(cfg):
                raise ValueError("Existing completed run has different controls")
            continue
        try:
            train(cfg, job["condition"], job["seed"], output_dir=job["directory"],
                  resume=previous is not None)
        except Exception as exc:
            failures.append({**job, "error": f"{type(exc).__name__}: {exc}"})
            write_json(root / "failures.json", failures)
        export(cfg)
    write_json(root / "failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed; see {root / 'failures.json'}")
    return export(cfg)


def freeze(cfg: dict) -> Path:
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    records = collect(cfg)
    if any(r["status"] != "completed" for r in records["runs"]):
        raise RuntimeError("All planned runs must complete before freezing final-test access")
    preflight(cfg)
    runs = []
    for job in plan(cfg):
        checkpoint = Path(job["directory"]) / "checkpoints/best.pt"
        state = load_checkpoint(checkpoint)
        if state["config_hash"] != canonical_hash(cfg):
            raise ValueError("Checkpoint configuration differs")
        selected = max(state["history"], key=lambda h: h["val"]["miou"])
        if selected["epoch"] != state["epoch"] or state["epoch"] != state["best_epoch"]:
            raise ValueError("Checkpoint does not follow the frozen selection rule")
        runs.append({**job, "checkpoint": str(checkpoint),
                     "checkpoint_sha256": file_sha256(checkpoint),
                     "selected_epoch": state["epoch"], "validation_miou": state["best_metric"]})
    protocol = {"schema_version": 1, "config_hash": canonical_hash(cfg),
                "config": cfg, "runs": runs,
                "selection_rule": "maximum dense validation mIoU; earliest exact tie",
                "manifest_sha256": file_sha256(resolve_path(cfg, cfg["data"]["manifest"])),
                "test_subset_seed": cfg["data"]["subset_seed"],
                "test_tile_cap": cfg["data"]["tile_caps"]["test"]}
    path = root / "frozen_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise FileExistsError("Frozen final-test protocol cannot be changed")
    else:
        write_json(path, protocol)
    return path


def final_test(cfg: dict):
    root = resolve_path(cfg, cfg["experiment"]["output_root"])
    protocol = root / "frozen_protocol.json"
    if not protocol.exists():
        raise FileNotFoundError("Freeze the protocol before test evaluation")
    for job in plan(cfg):
        directory = Path(job["directory"])
        output = directory / "final_test.json"
        if output.exists():
            continue
        requested = cfg["training"]["device"]
        device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        result = evaluate_checkpoint(directory / "checkpoints/best.pt", "test", device,
                                     allow_final_test=True, protocol=protocol)
        write_json(output, result)
    return export(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cpu_paired.yaml")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--final-test", action="store_true")
    action.add_argument("--export", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.execute:
        result = execute(cfg)
    elif args.preflight:
        result = preflight(cfg)
    elif args.freeze:
        result = {"frozen_protocol": str(freeze(cfg))}
    elif args.final_test:
        if not args.allow_final_test:
            parser.error("--final-test also requires --allow-final-test")
        result = final_test(cfg)
    elif args.export:
        result = export(cfg)
    else:
        result = {"planning_only": True, "runs": plan(cfg)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
