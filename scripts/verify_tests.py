"""Run the current suite and bind its result to source and test fingerprints."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from src.config import file_sha256, write_json


def verification_fingerprints(root: Path) -> dict:
    paths = [p for folder in ("src", "scripts", "experiments", "tests")
             for p in sorted((root / folder).glob("*.py"))]
    paths += sorted((root / "configs").glob("*.yaml"))
    return {str(p.relative_to(root)): file_sha256(p) for p in paths}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/verification"
    output.mkdir(parents=True, exist_ok=True)
    report = output / "reconstruction-pytest.xml"
    before = verification_fingerprints(root)
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"]
    result = subprocess.run(command, cwd=root)
    if result.returncode:
        write_json(output / "current_suite.json", {"status": "failed", "returncode": result.returncode})
        raise SystemExit(result.returncode)
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
              for k in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if counts["passed"] < 1 or counts["failures"] or counts["errors"]:
        raise RuntimeError("No successful, complete test run")
    after = verification_fingerprints(root)
    if before != after:
        raise RuntimeError("Source changed during verification")
    write_json(output / "current_suite.json",
               {"status": "passed", **counts, "fingerprints": after,
                "junit_sha256": file_sha256(report), "command": command,
                "scope": "reconstructed code; synthetic raster integration, not real-data smoke"})


if __name__ == "__main__":
    main()
