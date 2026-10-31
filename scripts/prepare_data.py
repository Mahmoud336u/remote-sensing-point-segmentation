"""Download, verify and inventory original LandCover.ai data without resampling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import ssl
import urllib.request
import zipfile

from src.config import file_sha256, load_config, resolve_path, write_json
from src.data_manifest import build_manifest, load_manifest


def download_verified(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        with urllib.request.urlopen(url, context=context, timeout=60) as source, partial.open("wb") as dest:
            shutil.copyfileobj(source, dest, length=1 << 20)
        if file_sha256(partial) != expected_sha256:
            raise ValueError(f"Downloaded archive checksum mismatch; quarantined at {partial}")
        partial.replace(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Existing archive checksum mismatch: {path}")


def extract_verified(archive: Path, root: Path, expected_sha256: str) -> None:
    if file_sha256(archive) != expected_sha256:
        raise ValueError("Refusing extraction of an unverified archive")
    marker = root / ".verified-archive.json"
    if root.exists():
        if marker.is_file() and json.loads(marker.read_text())["sha256"] == expected_sha256:
            return
        raise FileExistsError("Data directory exists without a matching verification marker; preserve it")
    staging = root.with_name(root.name + ".extracting")
    if staging.exists():
        raise FileExistsError(f"Preserve or inspect incomplete extraction: {staging}")
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (staging / info.filename).resolve()
            if not target.is_relative_to(staging.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive entry")
        z.extractall(staging)
    write_json(staging / ".verified-archive.json", {"sha256": expected_sha256})
    staging.rename(root)


def prepare(cfg: dict, download: bool = False) -> dict:
    d = cfg["data"]
    archive, root, manifest_path = (resolve_path(cfg, d[k]) for k in ("archive", "root", "manifest"))
    if download:
        download_verified(d["archive_url"], archive, d["archive_sha256"])
    if not root.exists():
        extract_verified(archive, root, d["archive_sha256"])
    marker = root / ".verified-archive.json"
    if not marker.is_file() or json.loads(marker.read_text())["sha256"] != d["archive_sha256"]:
        raise ValueError("Verified official archive extraction is required")
    if manifest_path.exists():
        manifest = load_manifest(manifest_path, root, verify_hashes=True)
    else:
        manifest = build_manifest(root, manifest_path, d)
    return {"manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "audit": manifest["audit"], "source": manifest["source"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = prepare(load_config(args.config), args.download)
    write_json("results/verification/data_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
