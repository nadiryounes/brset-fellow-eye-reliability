#!/usr/bin/env python3
"""Create the frozen R6 patient-only BRSET partition without model outputs.

This script reads BRSET v1.0.2 metadata, reconstructs the R0 clean bilateral
cohort, applies the prespecified R6 allocation rule, and writes only the four
authorized split artifacts. It never opens image pixels and never computes a
model quantity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get("BRSET_ROOT", PROJECT_ROOT / "data" / "BRSET_v1.0.2")
).expanduser()
METADATA = SOURCE_ROOT / "label_brset.csv"
IMAGE_DIR = SOURCE_ROOT / "fundus_photos"
EXPECTED_METADATA_SHA256 = (
    "4cd000ec1f651deb5cfb4f018ed1dec7a874da59a20c47f1e59a421d4a112adf"
)
OUTPUT_DIR = Path(
    os.environ.get("BRSET_SPLIT_DIR", PROJECT_ROOT / "r6_frozen_split")
).expanduser()

SEED = 20260829
TARGET_HOLDOUT = 1924
EXPECTED_ELIGIBLE = 7695
ENDPOINTS = ("drusen", "increased_cup_disc", "diabetic_retinopathy")
ENDPOINT_DISPLAY = {
    "drusen": "drusen_state",
    "increased_cup_disc": "increased_cup_disc_state",
    "diabetic_retinopathy": "diabetic_retinopathy_state",
}
EXPECTED_STATES = {
    "drusen": {"00": 6002, "11": 984, "10": 363, "01": 346},
    "increased_cup_disc": {"00": 5829, "11": 1171, "10": 399, "01": 296},
    "diabetic_retinopathy": {"00": 7153, "11": 440, "10": 60, "01": 42},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_sort_key(patient_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(patient_id))
    except ValueError:
        return (1, patient_id)


def assignment_digest(patient_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{patient_id}".encode("utf-8")).hexdigest()


def camera_pair(right: dict[str, str], left: dict[str, str]) -> str:
    cameras = (right["camera"], left["camera"])
    if cameras == ("Canon CR", "Canon CR"):
        return "Canon/Canon"
    if cameras == ("NIKON NF5050", "NIKON NF5050"):
        return "Nikon/Nikon"
    return "mixed"


def state(right: dict[str, str], left: dict[str, str], endpoint: str) -> str:
    return f"{right[endpoint]}{left[endpoint]}"


def quality_pair(right_value: str, left_value: str) -> str:
    labels = {"1": "normal", "2": "abnormal"}
    right = labels.get(right_value, "unevaluable")
    left = labels.get(left_value, "unevaluable")
    return f"R_{right}__L_{left}"


def patient_sex(right: dict[str, str], left: dict[str, str]) -> str:
    if right["patient_sex"] != left["patient_sex"]:
        return "inconsistent"
    return {"1": "male", "2": "female"}.get(
        right["patient_sex"], "missing_or_unexpected"
    )


def patient_age(right: dict[str, str], left: dict[str, str]) -> float | None:
    if right["patient_age"] != left["patient_age"]:
        return None
    try:
        value = float(right["patient_age"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def age_bin(age: float | None) -> str:
    if age is None:
        return "missing_or_inconsistent"
    if age < 40:
        return "under_40"
    if age < 60:
        return "40_to_59"
    if age < 80:
        return "60_to_79"
    return "80_or_older"


def hamilton(
    counts: dict[str, int], target: int, lexical_order: bool = True
) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0 or target < 0 or target > total:
        raise ValueError("Invalid Hamilton allocation inputs")
    quotas = {key: count * target / total for key, count in counts.items()}
    allocated = {key: math.floor(quota) for key, quota in quotas.items()}
    remainder = target - sum(allocated.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(quotas[key] - allocated[key]),
            counts[key],
            key if lexical_order else "",
        ),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    if sum(allocated.values()) != target:
        raise AssertionError("Hamilton allocation did not reach target")
    return allocated


def load_cohort() -> tuple[list[dict[str, object]], dict[str, object]]:
    if sha256_file(METADATA) != EXPECTED_METADATA_SHA256:
        raise RuntimeError("Source metadata SHA-256 differs from the R0-verified file")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with METADATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        grouped[row["patient_id"]].append(row)

    eligible: list[dict[str, object]] = []
    for patient_id, patient_rows in grouped.items():
        if len(patient_rows) != 2:
            continue
        by_eye = {row["exam_eye"]: row for row in patient_rows}
        if set(by_eye) != {"1", "2"}:
            continue
        right, left = by_eye["1"], by_eye["2"]
        if any(right[name] not in {"0", "1"} or left[name] not in {"0", "1"}
               for name in ENDPOINTS):
            continue
        if not (IMAGE_DIR / f"{right['image_id']}.jpg").is_file():
            raise RuntimeError(f"Missing right-eye image for patient {patient_id}")
        if not (IMAGE_DIR / f"{left['image_id']}.jpg").is_file():
            raise RuntimeError(f"Missing left-eye image for patient {patient_id}")

        age = patient_age(right, left)
        record: dict[str, object] = {
            "patient_id": patient_id,
            "camera_pair": camera_pair(right, left),
            "image_field_pair": quality_pair(
                right["image_field"], left["image_field"]
            ),
            "focus_pair": quality_pair(right["focus"], left["focus"]),
            "sex": patient_sex(right, left),
            "age": age,
            "age_bin": age_bin(age),
        }
        for endpoint in ENDPOINTS:
            record[ENDPOINT_DISPLAY[endpoint]] = state(right, left, endpoint)
        eligible.append(record)

    eligible.sort(key=lambda item: patient_sort_key(str(item["patient_id"])))
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE} eligible patients, found {len(eligible)}"
        )

    for endpoint in ENDPOINTS:
        observed = Counter(
            str(item[ENDPOINT_DISPLAY[endpoint]]) for item in eligible
        )
        if dict(observed) != EXPECTED_STATES[endpoint]:
            raise RuntimeError(
                f"{endpoint} states differ from R0: {dict(observed)}"
            )

    duplicate_eye_patients = sum(
        len({row["exam_eye"] for row in patient_rows}) < len(patient_rows)
        for patient_rows in grouped.values()
    )
    exclusions = {
        "all_metadata_rows": len(rows),
        "all_patients": len(grouped),
        "eligible_clean_bilateral_patients": len(eligible),
        "ineligible_patients_reconciled": len(grouped) - len(eligible),
        "single_image_patients": sum(len(value) == 1 for value in grouped.values()),
        "single_laterality_patients_nonexclusive": sum(
            len({row["exam_eye"] for row in value}) == 1
            for value in grouped.values()
        ),
        "duplicate_eye_patients_nonexclusive": duplicate_eye_patients,
        "more_than_two_image_patients_nonexclusive": sum(
            len(value) > 2 for value in grouped.values()
        ),
        "multi_camera_patients_all_nonexclusive": sum(
            len({row["camera"] for row in value}) > 1 for value in grouped.values()
        ),
        "invalid_focus_rows_all": sum(row["focus"] not in {"1", "2"} for row in rows),
    }
    return eligible, exclusions


def make_assignment(
    eligible: list[dict[str, object]],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    acquisition_counts = Counter(str(item["camera_pair"]) for item in eligible)
    acquisition_targets = hamilton(dict(acquisition_counts), TARGET_HOLDOUT)

    cell_targets: dict[tuple[str, str], int] = {}
    for acquisition in sorted(acquisition_counts):
        state_counts = Counter(
            str(item["drusen_state"])
            for item in eligible
            if item["camera_pair"] == acquisition
        )
        allocated = hamilton(dict(state_counts), acquisition_targets[acquisition])
        for drusen_state, count in allocated.items():
            cell_targets[(acquisition, drusen_state)] = count

    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in eligible:
        cells[(str(item["camera_pair"]), str(item["drusen_state"]))].append(item)

    holdout: set[str] = set()
    for cell, members in sorted(cells.items()):
        members.sort(
            key=lambda item: (
                assignment_digest(str(item["patient_id"])),
                patient_sort_key(str(item["patient_id"])),
            )
        )
        holdout.update(
            str(item["patient_id"]) for item in members[: cell_targets[cell]]
        )

    if len(holdout) != TARGET_HOLDOUT:
        raise AssertionError("Holdout size differs from frozen target")
    assignment = {
        str(item["patient_id"]): (
            "final_holdout" if str(item["patient_id"]) in holdout else "development"
        )
        for item in eligible
    }
    allocation = {
        "source_acquisition_counts": dict(sorted(acquisition_counts.items())),
        "holdout_acquisition_targets": dict(sorted(acquisition_targets.items())),
        "source_cell_counts": {
            f"{acquisition}|{drusen_state}": len(members)
            for (acquisition, drusen_state), members in sorted(cells.items())
        },
        "holdout_cell_targets": {
            f"{acquisition}|{drusen_state}": count
            for (acquisition, drusen_state), count in sorted(cell_targets.items())
        },
    }
    return assignment, allocation


def manifest_bytes(assignment: dict[str, str]) -> bytes:
    lines = ["patient_id,partition\n"]
    for patient_id in sorted(assignment, key=patient_sort_key):
        lines.append(f"{patient_id},{assignment[patient_id]}\n")
    return "".join(lines).encode("utf-8")


def aggregate_counts(
    eligible: list[dict[str, object]], assignment: dict[str, str], variable: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    levels = sorted({str(item[variable]) for item in eligible})
    for partition in ("development", "final_holdout"):
        counter = Counter(
            str(item[variable])
            for item in eligible
            if assignment[str(item["patient_id"])] == partition
        )
        result[partition] = {level: counter[level] for level in levels}
    return result


def build_balance_rows(
    eligible: list[dict[str, object]], assignment: dict[str, str]
) -> list[dict[str, str]]:
    variables = (
        "drusen_state",
        "increased_cup_disc_state",
        "diabetic_retinopathy_state",
        "camera_pair",
        "image_field_pair",
        "focus_pair",
        "sex",
        "age_bin",
    )
    denominators = Counter(assignment.values())
    rows: list[dict[str, str]] = []
    for variable in variables:
        counts = aggregate_counts(eligible, assignment, variable)
        levels = sorted(counts["development"])
        for level in levels:
            development_n = counts["development"][level]
            holdout_n = counts["final_holdout"][level]
            development_fraction = development_n / denominators["development"]
            holdout_fraction = holdout_n / denominators["final_holdout"]
            rows.append(
                {
                    "variable": variable,
                    "level": level,
                    "development_n": str(development_n),
                    "development_fraction": f"{development_fraction:.8f}",
                    "final_holdout_n": str(holdout_n),
                    "final_holdout_fraction": f"{holdout_fraction:.8f}",
                    "holdout_minus_development_fraction": (
                        f"{holdout_fraction - development_fraction:.8f}"
                    ),
                    "role": (
                        "allocation" if variable in {"drusen_state", "camera_pair"}
                        else "descriptive_balance_only"
                    ),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            f"Refusing to overwrite immutable split directory: {OUTPUT_DIR}"
        )

    eligible, exclusions = load_cohort()
    assignment, allocation = make_assignment(eligible)
    manifest = manifest_bytes(assignment)
    manifest_sha = hashlib.sha256(manifest).hexdigest()

    eligible_ids = "".join(
        f"{item['patient_id']}\n" for item in eligible
    ).encode("utf-8")
    eligible_set_sha = hashlib.sha256(eligible_ids).hexdigest()
    balance_rows = build_balance_rows(eligible, assignment)

    endpoint_counts = {
        endpoint: aggregate_counts(
            eligible, assignment, ENDPOINT_DISPLAY[endpoint]
        )
        for endpoint in ENDPOINTS
    }
    quality_counts = {
        variable: aggregate_counts(eligible, assignment, variable)
        for variable in ("image_field_pair", "focus_pair")
    }
    valid_ages = {
        partition: [
            float(item["age"])
            for item in eligible
            if assignment[str(item["patient_id"])] == partition
            and item["age"] is not None
        ]
        for partition in ("development", "final_holdout")
    }
    age_summary = {
        partition: {
            "evaluable_n": len(values),
            "mean": round(statistics.fmean(values), 6) if values else None,
            "median": round(statistics.median(values), 6) if values else None,
        }
        for partition, values in valid_ages.items()
    }

    summary = {
        "schema_version": "R6-frozen-partition-v1",
        "freeze_date": "2026-08-29",
        "source_release": "BRSET v1.0.2",
        "source_metadata": "label_brset.csv",
        "source_metadata_sha256": EXPECTED_METADATA_SHA256,
        "source_metadata_sha256_verified_before_and_after": True,
        "cohort_definition": (
            "Exactly two rows, one exam_eye=1 and one exam_eye=2, all three "
            "endpoint labels binary, and both referenced image files present."
        ),
        "cohort_audit": exclusions,
        "eligible_patient_set_sha256": eligible_set_sha,
        "partition_seed": SEED,
        "assignment_key": "SHA256(UTF-8 '20260829|patient_id')",
        "allocation_algorithm": (
            "Two-stage Hamilton largest remainder: acquisition-pair totals, then "
            "drusen state within acquisition; deterministic tie rules in protocol."
        ),
        "target_final_holdout_fraction": 0.25,
        "eligible_patients": len(eligible),
        "development_patients": sum(
            value == "development" for value in assignment.values()
        ),
        "final_holdout_patients": sum(
            value == "final_holdout" for value in assignment.values()
        ),
        "realized_final_holdout_fraction": round(TARGET_HOLDOUT / len(eligible), 12),
        "allocation": allocation,
        "endpoint_state_counts": endpoint_counts,
        "quality_pair_counts": quality_counts,
        "sex_counts": aggregate_counts(eligible, assignment, "sex"),
        "age_bin_counts": aggregate_counts(eligible, assignment, "age_bin"),
        "age_summary": age_summary,
        "manifest_columns": ["patient_id", "partition"],
        "manifest_sha256": manifest_sha,
        "holdout_used_for_model_selection": False,
        "holdout_performance_inspected": False,
        "model_predictions_present": False,
    }

    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    manifest_path = OUTPUT_DIR / "patient_partition.csv"
    manifest_path.write_bytes(manifest)

    balance_path = OUTPUT_DIR / "partition_balance.csv"
    write_csv(
        balance_path,
        [
            "variable",
            "level",
            "development_n",
            "development_fraction",
            "final_holdout_n",
            "final_holdout_fraction",
            "holdout_minus_development_fraction",
            "role",
        ],
        balance_rows,
    )

    summary_path = OUTPUT_DIR / "partition_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if sha256_file(METADATA) != EXPECTED_METADATA_SHA256:
        raise RuntimeError("Source metadata changed during partition generation")

    checksum_paths = (balance_path, manifest_path, summary_path)
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n"
        for path in sorted(checksum_paths, key=lambda item: item.name)
    )
    (OUTPUT_DIR / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")


if __name__ == "__main__":
    main()
