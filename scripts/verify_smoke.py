"""Verify real-data smoke provenance, annotations and exact checkpoint reload."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.checkpoint import load_checkpoint
from src.config import file_sha256, resolve_path, write_json
from src.data_manifest import load_manifest
from src.dataset import SegmentationDataset
from src.evaluate import evaluate_checkpoint
from src.train import source_fingerprints


def verify(run: str | Path) -> dict:
    run = Path(run)
    checkpoint = run / "checkpoints/best.pt"
    state = load_checkpoint(checkpoint)
    cfg = state["config"]
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "completed" or summary["optimizer_updates"] < 1:
        raise ValueError("Smoke training is incomplete or made no optimizer updates")
    manifest_path = resolve_path(cfg, cfg["data"]["manifest"])
    manifest = load_manifest(manifest_path, resolve_path(cfg, cfg["data"]["root"]), verify_hashes=True)
    if len(manifest["scenes"]) != 41 or manifest["source"]["sha256"] != cfg["data"]["archive_sha256"]:
        raise ValueError("This gate requires all 41 original official LandCover.ai scenes")
    marker = resolve_path(cfg, cfg["data"]["root"]) / ".verified-archive.json"
    if json.loads(marker.read_text())["sha256"] != cfg["data"]["archive_sha256"]:
        raise ValueError("Official archive verification marker is missing or mismatched")
    if state["provenance"]["source_fingerprints"] != source_fingerprints(cfg):
        raise ValueError("Source changed since smoke training")
    if file_sha256(manifest_path) != state["provenance"]["manifest_sha256"]:
        raise ValueError("Smoke manifest changed")
    original = json.loads((run / "best_validation.json").read_text())
    reloaded = evaluate_checkpoint(checkpoint, "val", "cpu")
    if reloaded["metrics"] != original["metrics"]:
        raise AssertionError("Reloaded validation metrics/confusion differ")
    ds = SegmentationDataset(cfg, "train", state["condition"], state["seed"])
    saved = json.loads((run / "annotations.json").read_text())
    if saved["records"] != [ds.annotation_record(i) for i in range(len(ds))]:
        raise AssertionError("Saved annotations do not match fixed source coordinates")
    write_json(run / "reloaded_validation.json", reloaded)
    result = {"status": "passed", "real_data": True, "test_inference": False,
              "scope": "integration smoke only; not campaign evidence",
              "run": str(run.resolve()), "manifest_sha256": file_sha256(manifest_path),
              "checkpoint_sha256": file_sha256(checkpoint),
              "annotations_sha256": file_sha256(run / "annotations.json"),
              "source_fingerprints": source_fingerprints(cfg),
              "optimizer_updates": summary["optimizer_updates"],
              "exact_reload": True, "annotation_fraction": saved["annotation_fraction"]}
    write_json(run / "verification.json", result)
    write_json("results/verification/real_data_smoke.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
