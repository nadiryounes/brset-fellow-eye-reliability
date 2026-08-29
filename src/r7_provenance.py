"""Final R7 run manifest and SHA-256 ledger."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import pandas
import scipy
import sklearn
import statsmodels
import torch
import torchvision
from PIL import __version__ as pillow_version

from .r7_config import project_path, sha256_file


def _included_files(config: dict[str, Any]) -> list[Path]:
    root = project_path(config, ".")
    files: list[Path] = []
    for relative in (
        "configs",
        "src",
        "tests",
        "r7_implementation",
        "r7_results",
        "r7_artifacts",
    ):
        directory = root / relative
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in {"R7_SHA256SUMS.txt", "R7_RUN_MANIFEST.json"}
        )
    files.extend([root / "requirements-r7.txt", root / "r7_run.py"])
    files.extend(
        [
            root / "r6_frozen_split" / "patient_partition.csv",
            root / "r6_frozen_split" / "partition_summary.json",
            root / "r6_frozen_split" / "partition_balance.csv",
            root / "r6_frozen_split" / "SHA256SUMS.txt",
        ]
    )
    return sorted(set(files), key=lambda path: str(path.relative_to(root)))


def finalize_provenance(config: dict[str, Any]) -> dict[str, Any]:
    root = project_path(config, ".")
    artifact_dir = project_path(config, "r7_artifacts")
    manifest_path = artifact_dir / "R7_RUN_MANIFEST.json"
    checksum_path = artifact_dir / "R7_SHA256SUMS.txt"
    if manifest_path.exists() or checksum_path.exists():
        raise RuntimeError("Refusing to overwrite final provenance")
    execution_path = project_path(
        config, config["firewall"]["holdout_execution_marker"]
    )
    final_report = project_path(config, "r7_results/R7_FINAL_EXECUTION_REPORT.md")
    if not execution_path.is_file() or not final_report.is_file():
        raise RuntimeError("Holdout execution/final report incomplete")
    files = _included_files(config)
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
    }
    manifest = {
        "status": "R7_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "Pillow": pillow_version,
        },
        "seeds": config["cross_validation"] | config["inference"],
        "model_identifier": config["model"],
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "frozen_config_sha256": config["_config_sha256"],
        "checkpoint_sha256": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        "file_hashes": hashes,
        "holdout_execution": json.loads(execution_path.read_text(encoding="utf-8")),
        "integrity": {
            "raw_brset_modified": False,
            "frozen_split_modified": False,
            "holdout_used_for_model_selection": False,
            "holdout_prediction_execution_count": 1,
            "additional_architectures": False,
            "epistemic_uncertainty_method": False,
            "ground_truth_state_as_deployment_predictor": False,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_files = [*files, manifest_path]
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root)}\n"
            for path in sorted(
                checksum_files, key=lambda item: str(item.relative_to(root))
            )
        ),
        encoding="utf-8",
    )
    return {
        "status": "R7_PROVENANCE_COMPLETE",
        "run_manifest": str(manifest_path.relative_to(root)),
        "run_manifest_sha256": sha256_file(manifest_path),
        "checksum_ledger": str(checksum_path.relative_to(root)),
        "checksum_ledger_sha256": sha256_file(checksum_path),
        "hashed_files": len(checksum_files),
    }
