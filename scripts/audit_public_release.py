#!/usr/bin/env python3
"""Fail closed if a proposed public release contains restricted data or secrets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "PUBLIC_RELEASE_AUDIT.json"
MANIFEST = ROOT / "SHA256SUMS.txt"

FORBIDDEN_PARTS = {
    ".env", ".venv", "__pycache__", "data", "physionet.org", "fundus_photos",
    "r0_audit", "r6_frozen_split", "r7_artifacts", "r7_results", "r8_analysis",
    "private_artifacts", "predictions", "embeddings", "checkpoints",
}
FORBIDDEN_SUFFIXES = {
    ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".npy", ".npz", ".pt",
    ".pth", ".pkl", ".joblib", ".p12", ".pfx", ".key", ".pem",
}
SECRET_PATTERNS = {
    "github_token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "absolute_user_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
}
PATIENT_LEVEL_COLUMNS = {"patient_id", "image_id", "exam_eye", "true_label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[dict[str, str]] = []
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
        and path not in {REPORT, MANIFEST}
    )
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            failures.append({"path": str(relative), "reason": "forbidden path component"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append({"path": str(relative), "reason": "forbidden binary/data suffix"})
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append({"path": str(relative), "reason": "file exceeds 10 MiB release limit"})

        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream), [])
            restricted = sorted(PATIENT_LEVEL_COLUMNS.intersection(header))
            if restricted:
                failures.append({
                    "path": str(relative),
                    "reason": "patient-level CSV columns: " + ",".join(restricted),
                })

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append({"path": str(relative), "reason": name})

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_ignores = [
        ".env", ".venv/", "data/", "r6_frozen_split/", "r7_artifacts/",
        "r7_results/", "*.npy", "*.pth", "*.joblib",
    ]
    missing_ignores = [item for item in required_ignores if item not in gitignore]
    failures.extend(
        {"path": ".gitignore", "reason": f"missing rule: {item}"}
        for item in missing_ignores
    )

    manifest_rows = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in files
    ]
    status = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": "public-release-audit-v1",
        "status": status,
        "files_scanned": len(files),
        "restricted_brset_content_found": any(
            "forbidden" in failure["reason"] or "patient-level" in failure["reason"]
            for failure in failures
        ),
        "secret_or_local_path_found": any(
            failure["reason"] in SECRET_PATTERNS for failure in failures
        ),
        "failures": failures,
        "audit_boundaries": [
            "No raw retinal image formats",
            "No BRSET source metadata or patient-level CSV columns",
            "No partitions, predictions, embeddings, fitted models, or checkpoints",
            "No common secret patterns or absolute workstation paths",
            "Required defensive gitignore rules present",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        if MANIFEST.exists():
            MANIFEST.unlink()
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    MANIFEST.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
