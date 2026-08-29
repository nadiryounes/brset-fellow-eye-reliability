"""Programmatic separation of development and final-holdout access."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .r7_config import project_path, sha256_file


DEVELOPMENT = "development"
FINAL_HOLDOUT = "final_holdout"
ALLOWED_STAGES = {DEVELOPMENT, FINAL_HOLDOUT}


class FirewallViolation(RuntimeError):
    """Raised when code attempts prohibited holdout access."""


@dataclass(frozen=True)
class PartitionIntegrity:
    manifest_sha256: str
    development_ids: frozenset[str]
    holdout_ids: frozenset[str]


def verify_partition(config: dict[str, Any]) -> PartitionIntegrity:
    manifest = project_path(config, config["partition"]["manifest_path"])
    observed_hash = sha256_file(manifest)
    expected_hash = config["partition"]["manifest_sha256"]
    if observed_hash != expected_hash:
        raise FirewallViolation(
            f"Frozen manifest hash mismatch: {observed_hash} != {expected_hash}"
        )

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["patient_id", "partition"]:
            raise FirewallViolation("Frozen manifest schema changed")
        rows = list(reader)

    patient_ids = [row["patient_id"] for row in rows]
    if len(patient_ids) != len(set(patient_ids)):
        raise FirewallViolation("A patient appears more than once in the manifest")
    development_ids = frozenset(
        row["patient_id"]
        for row in rows
        if row["partition"] == config["partition"]["development_label"]
    )
    holdout_ids = frozenset(
        row["patient_id"]
        for row in rows
        if row["partition"] == config["partition"]["holdout_label"]
    )
    known = development_ids | holdout_ids
    if len(known) != len(rows):
        raise FirewallViolation("Manifest contains an unknown partition value")
    if development_ids & holdout_ids:
        raise FirewallViolation("Development/holdout patient overlap detected")
    if len(development_ids) != config["partition"]["development_patients"]:
        raise FirewallViolation("Development patient count changed")
    if len(holdout_ids) != config["partition"]["final_holdout_patients"]:
        raise FirewallViolation("Holdout patient count changed")
    return PartitionIntegrity(observed_hash, development_ids, holdout_ids)


def _load_release_marker(config: dict[str, Any]) -> dict[str, Any]:
    marker_path = project_path(
        config, config["firewall"]["holdout_release_marker"]
    )
    if not marker_path.is_file():
        raise FirewallViolation(
            "Final-holdout access denied: HOLDOUT_RELEASED.json does not exist"
        )
    with marker_path.open(encoding="utf-8") as handle:
        marker = json.load(handle)
    required = {
        "status": "HOLDOUT_RELEASED",
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "frozen_config_sha256": config["_config_sha256"],
        "scientific_specification_frozen": True,
        "pre_holdout_validation": "PASS",
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise FirewallViolation(f"Invalid release marker field: {key}")
    for relative_path, expected_hash in marker.get("sealed_file_hashes", {}).items():
        actual = sha256_file(project_path(config, relative_path))
        if actual != expected_hash:
            raise FirewallViolation(
                f"Sealed development file changed after release: {relative_path}"
            )
    return marker


def authorize_stage(
    config: dict[str, Any], stage: str, *, purpose: str
) -> PartitionIntegrity:
    if stage not in ALLOWED_STAGES:
        raise FirewallViolation(f"Unknown execution stage: {stage}")
    integrity = verify_partition(config)
    if stage == FINAL_HOLDOUT:
        _load_release_marker(config)
        execution_marker = project_path(
            config, config["firewall"]["holdout_execution_marker"]
        )
        if execution_marker.exists() and purpose in {
            "embedding_extraction",
            "prediction_generation",
        }:
            raise FirewallViolation(
                "Final holdout is write-once and has already been executed"
            )
    return integrity


def allowed_patient_ids(
    config: dict[str, Any], stage: str, *, purpose: str
) -> frozenset[str]:
    integrity = authorize_stage(config, stage, purpose=purpose)
    return (
        integrity.development_ids if stage == DEVELOPMENT else integrity.holdout_ids
    )


def stream_stage_metadata(
    config: dict[str, Any],
    stage: str,
    *,
    columns: Iterable[str],
    purpose: str,
) -> list[dict[str, str]]:
    """Read only authorized rows and requested columns.

    The patient ID is checked against the authorized set before a retained
    dictionary is constructed. Development code therefore never materializes
    holdout labels or other holdout metadata values.
    """

    requested = tuple(dict.fromkeys(("patient_id", *columns)))
    allowed = allowed_patient_ids(config, stage, purpose=purpose)
    metadata_path = Path(config["source"]["metadata_path"])
    if sha256_file(metadata_path) != config["source"]["metadata_sha256"]:
        raise FirewallViolation("Raw metadata hash mismatch")

    records: list[dict[str, str]] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in requested}
        patient_position = positions["patient_id"]
        for raw_row in reader:
            patient_id = raw_row[patient_position]
            if patient_id not in allowed:
                continue
            records.append({name: raw_row[positions[name]] for name in requested})
    expected_rows = 2 * len(allowed)
    if len(records) != expected_rows:
        raise FirewallViolation(
            f"Authorized stage expected {expected_rows} image rows, got {len(records)}"
        )
    if {record["patient_id"] for record in records} != set(allowed):
        raise FirewallViolation("Authorized metadata IDs do not match manifest IDs")
    return records


def verify_stage_image_files(config: dict[str, Any], stage: str) -> dict[str, Any]:
    records = stream_stage_metadata(
        config,
        stage,
        columns=("image_id", "exam_eye"),
        purpose="file_integrity",
    )
    image_dir = Path(config["source"]["image_dir"])
    missing = [
        record["image_id"]
        for record in records
        if not (image_dir / f"{record['image_id']}.jpg").is_file()
    ]
    if missing:
        raise FirewallViolation(f"Missing authorized image files: {len(missing)}")
    return {
        "stage": stage,
        "patients": len({record["patient_id"] for record in records}),
        "images": len(records),
        "missing_images": 0,
    }


def create_holdout_release_marker(
    config: dict[str, Any], sealed_relative_paths: Iterable[str]
) -> Path:
    """Create the one-way release marker after a PASS validation report."""

    verify_partition(config)
    completion_path = project_path(
        config, config["firewall"]["development_completion_marker"]
    )
    validation_path = project_path(
        config, config["firewall"]["pre_holdout_report"]
    )
    if not completion_path.is_file() or not validation_path.is_file():
        raise FirewallViolation("Development completion/validation files are missing")
    validation_text = validation_path.read_text(encoding="utf-8")
    if "`PASS`" not in validation_text and "# PASS" not in validation_text:
        raise FirewallViolation("Pre-holdout validation does not declare PASS")

    marker_path = project_path(
        config, config["firewall"]["holdout_release_marker"]
    )
    if marker_path.exists():
        raise FirewallViolation("Holdout release marker already exists")
    sealed = {
        relative: sha256_file(project_path(config, relative))
        for relative in sorted(set(sealed_relative_paths))
    }
    marker = {
        "status": "HOLDOUT_RELEASED",
        "released_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "frozen_config_sha256": config["_config_sha256"],
        "scientific_specification_frozen": True,
        "pre_holdout_validation": "PASS",
        "development_completion_sha256": sha256_file(completion_path),
        "pre_holdout_validation_sha256": sha256_file(validation_path),
        "sealed_file_hashes": sealed,
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker_path
