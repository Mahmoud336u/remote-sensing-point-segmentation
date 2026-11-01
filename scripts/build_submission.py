"""Build and verify a portable source/evidence archive with per-file SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from src.config import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
NAME = "remote_sensing_assessment_submission.zip"
PREFIX = "remote_sensing_assessment/"
FOLDERS = ("src", "scripts", "experiments", "configs", "tests", "notebooks", "docs", "report")
SUFFIXES = {".py", ".md", ".json", ".csv", ".yaml", ".ipynb", ".pdf", ".png", ".svg", ".npz",
            ".tex", ".bib", ".dot", ".html", ".css", ".xml", ".sh"}


def payload():
    paths = {ROOT / n for n in ("README.md", "PROJECT_STATE.md", "LICENSE", "pyproject.toml", ".gitignore", ".gitattributes")}
    paths.update(ROOT.glob("requirements*"))
    for folder in FOLDERS:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in SUFFIXES
                     and "__pycache__" not in p.parts)
    paths.add(ROOT / "data/README.md")
    paths.update(p for p in (ROOT / "results").rglob("*") if p.is_file() and p.suffix in SUFFIXES
                 and p.name != "submission_validation.json")
    for p in paths:
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f"Unsafe package path: {p}")
        if p.stat().st_size > 20 * 1024**2:
            raise ValueError(f"Large artifact requires deliberate review: {p}")
    return sorted(paths)


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            parts = Path(name).parts
            if not name.startswith(PREFIX) or ".." in parts or any(
                    part in {".git", ".venv", "__pycache__", ".aws", ".env", "checkpoints"} for part in parts):
                raise ValueError(f"Forbidden archive member: {name}")
        manifest = json.loads(archive.read(PREFIX + "SUBMISSION_MANIFEST.json"))
        if set(names) != {PREFIX + e["path"] for e in manifest["files"]} | {PREFIX + "SUBMISSION_MANIFEST.json"}:
            raise ValueError("Manifest membership differs from ZIP")
        for entry in manifest["files"]:
            content = archive.read(PREFIX + entry["path"])
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Integrity failure: {entry['path']}")
        required = {"report/technical_report.pdf", "report/technical_report.md", "report/reviewer_audit.md",
                    "report/evidence_inventory.json", "notebooks/assessment_demo.ipynb", "README.md",
                    "results/cpu_paired/results.json", "results/verification/current_suite.json"}
        if not required <= {e["path"] for e in manifest["files"]}:
            raise ValueError("Required submission artifact missing")
    return {"status": "passed", "archive": path.name, "members": len(names),
            "bytes": path.stat().st_size, "sha256": file_sha256(path),
            "checks": ["CRC", "membership", "SHA256 per file", "required artifacts", "excluded paths"],
            "assessment_complete": False, "trained_checkpoint_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / NAME
    if not args.verify_only:
        files = payload()
        entries = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files]
        manifest = {"schema_version": 1, "hash_policy": "All members except this manifest; archive hash is external",
                    "assessment_complete": False, "files": entries}
        temporary = output.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry, path in zip(entries, files):
                info = zipfile.ZipInfo(PREFIX + entry["path"], date_time=(2026, 11, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
            info = zipfile.ZipInfo(PREFIX + "SUBMISSION_MANIFEST.json", date_time=(2026, 11, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    result = verify(output)
    output.with_suffix(".zip.sha256").write_text(f"{result['sha256']}  {output.name}\n")
    write_json(ROOT / "results/verification/submission_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
