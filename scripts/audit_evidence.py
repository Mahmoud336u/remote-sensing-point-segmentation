"""Inventory available assessment evidence; never replace missing experiment scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import file_sha256, load_config, write_json
from src.evaluate import validate_saved_metrics
from scripts.verify_tests import verification_fingerprints

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    suite = read("results/verification/current_suite.json")
    assert suite["status"] == "passed"
    assert suite["fingerprints"] == verification_fingerprints(ROOT), "Test evidence is stale"
    assert suite["junit_sha256"] == file_sha256(ROOT / "results/verification/reconstruction-pytest.xml")
    notebook = read("results/verification/notebook.json")
    assert notebook["sha256"] == file_sha256(ROOT / "notebooks/assessment_demo.ipynb")
    rows = read("results/cpu_paired/results.json")
    statuses = read("results/cpu_paired/run_status.json")
    with (ROOT / "results/cpu_paired/results.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    run_files = sorted((ROOT / "results").rglob("run.json"))
    inspected_runs = []
    for path in run_files:
        run = json.loads(path.read_text())
        for metric_path in path.parent.glob("*validation.json"):
            validate_saved_metrics(json.loads(metric_path.read_text())["metrics"])
        inspected_runs.append({"path": str(path.relative_to(ROOT)), "status": run["status"],
                               "profile": run["profile"], "sha256": file_sha256(path)})
    cfg = load_config(ROOT / "configs/cpu_paired.yaml")
    claims = [
        {"id": "E01", "claim": "Current software tests", "status": "verified_synthetic_scope",
         "value": {k: suite[k] for k in ("passed", "skipped", "failures", "errors")},
         "sources": ["results/verification/current_suite.json", "results/verification/reconstruction-pytest.xml"]},
        {"id": "E02", "claim": "Locked CPU dependency install", "status": "verified",
         "value": read("results/verification/lock_install.json"),
         "sources": ["requirements-phase3-cpu.lock", "results/verification/lock_install.json"]},
        {"id": "E03", "claim": "U-Net parameter count and ImageNet initialization", "status": "verified",
         "value": read("results/verification/reconstructed_initialization.json"),
         "sources": ["src/model.py", "results/verification/reconstructed_initialization.json"]},
        {"id": "E04", "claim": "Original dataset inventory and split statistics", "status": "historical_unrecovered",
         "value": None, "sources": ["docs/recovery_checkpoint.json", "docs/source_evidence.json"]},
        {"id": "E05", "claim": "Measured campaign score rows", "status": "verified_artifact_count",
         "value": len(rows), "sources": ["results/cpu_paired/results.json", "results/cpu_paired/results.csv"]},
        {"id": "E06", "claim": "Real-data partition and smoke verification", "status": "blocked_missing_artifacts",
         "value": {"manifest_available": (ROOT / cfg["data"]["manifest"]).exists(),
                   "real_data_smoke_available": (ROOT / "results/verification/real_data_smoke.json").exists()},
         "sources": ["results/verification/phase4_preflight.json", "docs/experiment_analysis.md"]},
        {"id": "E07", "claim": "Sequential notebook execution", "status": "verified_execution_incomplete_real_data",
         "value": notebook, "sources": ["notebooks/assessment_demo.ipynb", "results/verification/notebook.json"]},
        {"id": "E08", "claim": "Candidate CPU campaign controls", "status": "planned_not_executed",
         "value": {"epochs": 10, "seeds": [17, 29, 43], "tile_caps": cfg["data"]["tile_caps"]},
         "sources": ["configs/cpu_paired.yaml", "results/cpu_paired/campaign_config.json"]},
        {"id": "E09", "claim": "Nominal annotation percent on fully valid 256-square tiles",
         "status": "arithmetic_not_measured_subset", "value": {str(k): 100*k/256**2 for k in (10, 50, 200)},
         "sources": ["configs/cpu_paired.yaml", "src/dataset.py"]},
        {"id": "E10", "claim": "Original three assessment requirements reviewed", "status": "verified_user_document",
         "value": read("docs/assessment_source.json"), "sources": ["docs/assessment_requirements.md", "docs/assessment_source.json"]},
    ]
    for claim in claims:
        claim["source_sha256"] = {p: file_sha256(ROOT / p) for p in claim["sources"]}
    experimental_claims = []
    for run in statuses:
        matches = [r for r in rows if r["run_id"] == run["run_id"]]
        experimental_claims.append({"run_id": run["run_id"], "condition": run["condition"],
                                   "seed": run["seed"], "status": run["status"],
                                   "configuration": "configs/cpu_paired.yaml",
                                   "configuration_sha256": file_sha256(ROOT / "configs/cpu_paired.yaml"),
                                   "metric_files": [str(Path(run["directory"]) / "best_validation.json")] if matches else [],
                                   "measured_result_rows": matches,
                                   "supported_performance_claims": bool(matches)})
    inventory = []
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and path.name not in {"phase5_evidence.json", "submission_validation.json", "report_render.json"}:
            inventory.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": file_sha256(path)})
    figures = [p["path"] for p in inventory if Path(p["path"]).suffix in (".png", ".pdf", ".svg")]
    result = {"schema_version": 1, "claims": claims, "experimental_claims": experimental_claims,
              "inspected_run_files": inspected_runs, "result_file_inventory": inventory,
              "campaign_figure_files": figures,
              "excluded_evidence": ["historical unrecovered smoke scores", "temporary synthetic test fixtures outside results"],
              "assessment_complete": False}
    write_json(ROOT / "report/evidence_inventory.json", result)
    with (ROOT / "report/evidence_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "claim", "status", "sources", "value"])
        writer.writeheader()
        for c in claims:
            writer.writerow({**{k: c[k] for k in ("id", "claim", "status")},
                             "sources": "; ".join(c["sources"]), "value": json.dumps(c["value"], sort_keys=True)})
    write_json(ROOT / "results/verification/phase5_evidence.json",
               {"status": "inventory_verified", "completed_campaign_rows": len(rows),
                "candidate_runs": len(statuses), "campaign_figures": len(figures),
                "assessment_complete": False,
                "inventory_sha256": file_sha256(ROOT / "report/evidence_inventory.json")})
    print(json.dumps({"rows": len(rows), "candidate_runs": len(statuses), "figures": len(figures)}))


if __name__ == "__main__":
    main()
