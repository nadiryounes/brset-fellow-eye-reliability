#!/usr/bin/env python3
"""R0 BRSET v1.0.2 dataset-integrity and scientific-feasibility audit.

This program reads metadata and image files only. It performs no image decoding,
feature extraction, dataset splitting for model development, or model training.

Usage:
    .venv/bin/python r0_brset_audit.py \
        --root /path/to/brazilian-ophthalmological/1.0.2 \
        --out r0_audit --hash-images
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_VERSION = "1.0.2"
EXPECTED_N_IMAGES = 16_266
EXPECTED_N_PATIENTS = 8_524
RIGHT_EYE = 1
LEFT_EYE = 2

QUALITY_COLS = ["focus", "illumination", "image_field", "artifacts"]
ANATOMY_COLS = ["optic_disc", "vessels", "macula"]
BINARY_LABELS = [
    "diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd",
    "vascular_occlusion", "hypertensive_retinopathy", "drusen", "hemorrhage",
    "retinal_detachment", "myopic_fundus", "increased_cup_disc", "other",
]
ORDINAL_LABELS = ["DR_ICDR", "DR_SDRG"]

EXPECTED_COLUMNS = [
    "image_id", "patient_id", "camera", "patient_age", "comorbidities",
    "diabetes_time_y", "insulin", "patient_sex", "exam_eye", "diabetes",
    "nationality", "optic_disc", "vessels", "macula", "DR_SDRG", "DR_ICDR",
    "focus", "illumination", "image_field", "artifacts", *BINARY_LABELS, "quality",
]

LEGACY_TYPO_COLS = {
    "illuminaton": "illumination", "Illuminaton": "illumination",
    "drusens": "drusen", "insuline": "insulin",
}

VARIABLE_GUIDANCE = {
    "image_id": (
        "Sequential de-identified image identifier used to join metadata to files.",
        "available", "low if used only as an identifier; high if modeled",
        "Sequential order may proxy acquisition workflow or time; identifiers must not be predictors.",
        "file join and audit only",
    ),
    "patient_id": (
        "De-identified patient grouping identifier.",
        "available", "high if modeled; essential for leakage control",
        "Directly links fellow-eye and repeated records from the same patient.",
        "grouping, patient-level splitting, and clustered analysis only",
    ),
    "camera": (
        "Retinal camera recorded at acquisition.",
        "available at acquisition", "context dependent",
        "May proxy site, period, operator, population, or workflow; it is not an isolated device effect.",
        "stratification, shift description, sensitivity analysis; predictor only if deployment supplies it",
    ),
    "patient_age": (
        "Age in years from the clinical record.",
        "workflow dependent", "context dependent",
        "Available in many workflows but not an image-only signal; some patients have inconsistent row values.",
        "descriptive covariate or prespecified multimodal predictor",
    ),
    "patient_sex": (
        "Recorded sex (1 male, 2 female per documentation).",
        "workflow dependent", "context dependent",
        "Available in many workflows but not an image-only signal; some patients have inconsistent row values.",
        "descriptive covariate or prespecified multimodal predictor",
    ),
    "nationality": (
        "Recorded nationality.", "workflow dependent", "low in this dataset",
        "Constant in the observed metadata and therefore non-informative within BRSET.",
        "cohort description only",
    ),
    "comorbidities": (
        "Free-text, self-reported clinical antecedents.",
        "workflow dependent", "potential leakage",
        "Contains disease terms and heterogeneous free text that may reveal target-relevant diagnoses.",
        "descriptive analysis; do not use as a disease predictor without a separate leakage review",
    ),
    "diabetes_time_y": (
        "Self-reported time since diabetes diagnosis.",
        "only when diabetes history is collected", "potential leakage",
        "Directly encodes disease history relevant to diabetic-retinopathy outcomes.",
        "descriptive/stratification variable; not an image-only predictor",
    ),
    "insulin": (
        "Self-reported insulin use.",
        "only when medication history is collected", "potential leakage",
        "Treatment history is strongly related to diabetes status and severity.",
        "descriptive/stratification variable; not an image-only predictor",
    ),
    "diabetes": (
        "Recorded diabetes diagnosis.", "only when diagnosis is known", "potential leakage",
        "Directly reveals a major target-related clinical condition.",
        "cohort definition or stratification; not a predictor for diabetes-related image labels",
    ),
    "optic_disc": (
        "Specialist anatomical assessment of the optic disc.",
        "post-image specialist assessment", "post-diagnostic",
        "Human annotation overlaps retinal phenotype and may directly reveal disease-related abnormalities.",
        "auxiliary outcome or descriptive label; not a disease predictor",
    ),
    "vessels": (
        "Specialist anatomical assessment of retinal vessels.",
        "post-image specialist assessment", "post-diagnostic",
        "Human annotation overlaps retinal phenotype and may directly reveal disease-related abnormalities.",
        "auxiliary outcome or descriptive label; not a disease predictor",
    ),
    "macula": (
        "Specialist anatomical assessment of the macula.",
        "post-image specialist assessment", "post-diagnostic",
        "Human annotation overlaps retinal phenotype and may directly reveal disease-related abnormalities.",
        "auxiliary outcome or descriptive label; not a disease predictor",
    ),
    "DR_ICDR": (
        "Specialist ICDR diabetic-retinopathy grade (0-4).",
        "post-diagnostic", "post-diagnostic/direct target leakage",
        "Directly encodes diabetic-retinopathy severity.", "outcome only",
    ),
    "DR_SDRG": (
        "Specialist SDRG diabetic-retinopathy grade (0-4).",
        "post-diagnostic", "post-diagnostic/direct target leakage",
        "Directly encodes diabetic-retinopathy severity.", "outcome only",
    ),
    "quality": (
        "Aggregate human image-quality label.",
        "post-image human assessment unless separately automated", "uncertain/potential leakage",
        "Not automatically available to a deployed image-only model and may be derived from component annotations.",
        "quality outcome, stratification, or separately generated signal; not an assumed inference-time predictor",
    ),
}

for _quality_col in QUALITY_COLS:
    VARIABLE_GUIDANCE[_quality_col] = (
        f"Human assessment of image {_quality_col.replace('_', ' ')} (1 normal, 2 abnormal).",
        "post-image human assessment unless separately automated", "uncertain/potential leakage",
        "The supplied annotation is not automatically available at inference and must not be conflated with model uncertainty.",
        "quality outcome/stratification or target for a separately validated quality mechanism",
    )

for _label_col in BINARY_LABELS:
    VARIABLE_GUIDANCE[_label_col] = (
        f"Specialist binary pathology label: {_label_col.replace('_', ' ')}.",
        "post-diagnostic", "post-diagnostic/direct target leakage",
        "This is a study outcome and cannot be supplied as a predictor for retinal pathology.",
        "outcome only",
    )


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def markdown_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows; the applicable audit found zero records._"
    return frame.head(max_rows).to_markdown(index=False)


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def find_labels(root: Path) -> Path:
    preferred = [root / "label_brset.csv", root / "labels.csv"]
    direct = [path for path in preferred if path.is_file()]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise RuntimeError(f"Ambiguous metadata files at dataset root: {direct}")
    candidates = sorted(
        [*root.rglob("label_brset.csv"), *root.rglob("labels.csv")],
        key=lambda path: (len(path.relative_to(root).parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError(f"Neither label_brset.csv nor labels.csv was found below {root}")
    shallowest = len(candidates[0].relative_to(root).parts)
    tied = [p for p in candidates if len(p.relative_to(root).parts) == shallowest]
    if len(tied) != 1:
        raise RuntimeError(f"Ambiguous metadata files below dataset root: {tied}")
    return tied[0]


def find_images(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def parse_sha256sums(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    if not path.is_file():
        return expected
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                raise ValueError(f"Malformed SHA256SUMS line {line_number}: {text[:100]}")
            relative_path = parts[1].lstrip("*")
            if relative_path in expected:
                raise ValueError(f"Duplicate path in SHA256SUMS: {relative_path}")
            expected[relative_path] = parts[0].lower()
    return expected


def allowed_value_audit(
    frame: pd.DataFrame, column: str, allowed: set[int],
) -> tuple[pd.Series, list[dict[str, object]]]:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    invalid_mask = frame[column].notna() & ~numeric.isin(allowed)
    anomalies: list[dict[str, object]] = []
    if invalid_mask.any():
        counts = frame.loc[invalid_mask, column].astype(str).value_counts(dropna=False)
        anomalies = [
            {"column": column, "unexpected_value": value, "count": int(count)}
            for value, count in counts.items()
        ]
    return numeric, anomalies


def code_summary(series: pd.Series, allowed: set[int]) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & ~numeric.isin(allowed)
    return {
        "nonmissing_n": int(series.notna().sum()),
        "missing_n": int(series.isna().sum()),
        "evaluable_n": int((series.notna() & ~invalid).sum()),
        "unexpected_n": int(invalid.sum()),
        "unexpected_values": "|".join(sorted(series.loc[invalid].astype(str).unique())),
    }


def patient_structure_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, group in frame.groupby("patient_id", dropna=False, sort=True):
        right_n = int((group["exam_eye"] == RIGHT_EYE).sum())
        left_n = int((group["exam_eye"] == LEFT_EYE).sum())
        invalid_eye_n = int((~group["exam_eye"].isin([RIGHT_EYE, LEFT_EYE]) & group["exam_eye"].notna()).sum())
        eye_pattern = (
            "OD+OS" if right_n and left_n else
            "OD-only" if right_n else
            "OS-only" if left_n else "no valid eye code"
        )
        rows.append({
            "patient_id": patient_id,
            "n_images": int(len(group)),
            "od_rows": right_n,
            "os_rows": left_n,
            "invalid_eye_rows": invalid_eye_n,
            "n_unique_valid_eyes": int(group.loc[group["exam_eye"].isin([1, 2]), "exam_eye"].nunique()),
            "eye_pattern": eye_pattern,
            "has_both_eye_codes": int(right_n > 0 and left_n > 0),
            "valid_unique_od_os_pair": int(right_n == 1 and left_n == 1 and invalid_eye_n == 0),
            "has_duplicate_eye_record": int(right_n > 1 or left_n > 1),
            "n_cameras": int(group["camera"].dropna().nunique()),
            "cameras": "|".join(sorted(group["camera"].dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def build_variable_leakage_table(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        description, availability, leakage, reason, use = VARIABLE_GUIDANCE.get(
            column,
            (
                "No field-specific definition was encoded in this audit.", "uncertain", "uncertain",
                "Inference-time availability and target relationship require protocol-specific review.",
                "do not model until adjudicated",
            ),
        )
        rows.append({
            "variable": column, "description": description,
            "availability_at_inference": availability, "potential_leakage": leakage,
            "reason": reason, "recommended_use": use,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="BRSET v1.0.2 dataset root")
    parser.add_argument("--out", default="r0_audit", help="Audit output directory")
    parser.add_argument(
        "--hash-images", action="store_true",
        help="Hash every image, verify SHA256SUMS, and detect exact duplicate bytes",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    out.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, str]] = []

    def flag(level: str, code: str, message: str) -> None:
        findings.append({"level": level, "code": code, "message": message})

    labels_path = find_labels(root)
    frame = pd.read_csv(labels_path)
    label_sha256 = sha256_file(labels_path)

    # Version and schema verification.
    if root.name != EXPECTED_VERSION:
        flag("YELLOW", "VERSION_PATH", f"Dataset directory is named {root.name!r}, not {EXPECTED_VERSION!r}.")
    legacy_found = [column for column in LEGACY_TYPO_COLS if column in frame.columns]
    if legacy_found:
        flag("RED", "LEGACY_SCHEMA", f"Legacy columns found: {legacy_found}")
    missing_expected_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    unexpected_columns = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing_expected_columns:
        flag("RED", "MISSING_EXPECTED_COLUMNS", f"Missing expected v1.0.2 columns: {missing_expected_columns}")
    if unexpected_columns:
        flag("YELLOW", "UNEXPECTED_COLUMNS", f"Columns outside the audited v1.0.2 schema: {unexpected_columns}")
    if labels_path.name == "label_brset.csv":
        flag(
            "INFO", "LOCAL_METADATA_FILENAME",
            "The downloaded v1.0.2 file is label_brset.csv; the PhysioNet narrative still calls it labels.csv.",
        )
    for absent_doc in ["README.txt", "RECORDS"]:
        if not (root / absent_doc).exists():
            flag("INFO", "LOCAL_FILE_ABSENT", f"{absent_doc} is not present in the local v1.0.2 release directory.")

    n_rows = int(len(frame))
    n_image_ids = int(frame["image_id"].nunique(dropna=True))
    n_patients = int(frame["patient_id"].nunique(dropna=True))
    if n_rows != EXPECTED_N_IMAGES:
        flag("RED", "ROW_COUNT", f"Observed {n_rows} metadata rows; expected {EXPECTED_N_IMAGES}.")
    if n_patients != EXPECTED_N_PATIENTS:
        flag("RED", "PATIENT_COUNT", f"Observed {n_patients} patients; expected {EXPECTED_N_PATIENTS}.")

    # Value-domain audit without rewriting or silently coercing the source.
    value_anomalies: list[dict[str, object]] = []
    allowed_domains = {
        "patient_sex": {1, 2}, "exam_eye": {RIGHT_EYE, LEFT_EYE},
        **{column: {1, 2} for column in QUALITY_COLS + ANATOMY_COLS},
        **{column: {0, 1} for column in BINARY_LABELS},
        **{column: {0, 1, 2, 3, 4} for column in ORDINAL_LABELS},
    }
    numeric_views: dict[str, pd.Series] = {}
    for column, allowed in allowed_domains.items():
        if column not in frame.columns:
            continue
        numeric, anomalies = allowed_value_audit(frame, column, allowed)
        numeric_views[column] = numeric
        value_anomalies.extend(anomalies)
        if anomalies:
            flag("RED", "UNEXPECTED_CODE", f"{column} contains values outside {sorted(allowed)}: {anomalies}")
    write_csv(
        pd.DataFrame(value_anomalies), out / "value_anomalies.csv",
        ["column", "unexpected_value", "count"],
    )

    # Metadata/file integrity.
    images = find_images(root)
    relative_images = [path.relative_to(root).as_posix() for path in images]
    image_file_manifest = pd.DataFrame({
        "relative_path": relative_images, "filename": [path.name for path in images],
        "stem": [path.stem for path in images], "extension": [path.suffix.lower() for path in images],
        "bytes": [path.stat().st_size for path in images],
    })
    write_csv(
        image_file_manifest, out / "image_file_manifest.csv",
        ["relative_path", "filename", "stem", "extension", "bytes"],
    )

    label_ids = set(frame["image_id"].dropna().astype(str))
    file_ids = set(image_file_manifest["stem"].astype(str)) if not image_file_manifest.empty else set()
    missing_image_ids = sorted(label_ids - file_ids)
    orphan_file_stems = sorted(file_ids - label_ids)
    write_csv(pd.DataFrame({"image_id": missing_image_ids}), out / "missing_image_files.csv", ["image_id"])
    write_csv(pd.DataFrame({"file_stem": orphan_file_stems}), out / "orphan_image_files.csv", ["file_stem"])
    if missing_image_ids:
        flag("RED", "MISSING_IMAGES", f"{len(missing_image_ids)} labeled image identifiers lack a file.")
    if orphan_file_stems:
        flag("RED", "ORPHAN_IMAGES", f"{len(orphan_file_stems)} image files lack a metadata row.")
    if len(images) != EXPECTED_N_IMAGES:
        flag("RED", "IMAGE_FILE_COUNT", f"Observed {len(images)} image files; expected {EXPECTED_N_IMAGES}.")

    duplicate_image_id_mask = frame.duplicated(["image_id"], keep=False)
    duplicate_patient_eye_sizes = frame.groupby(["patient_id", "exam_eye"], dropna=False).size()
    duplicate_patient_eye = duplicate_patient_eye_sizes[duplicate_patient_eye_sizes > 1].rename("row_count").reset_index()
    full_duplicate_mask = frame.duplicated(keep=False)
    write_csv(
        frame.loc[duplicate_image_id_mask].sort_values("image_id"),
        out / "duplicate_image_id_rows.csv", list(frame.columns),
    )
    write_csv(
        duplicate_patient_eye, out / "duplicate_patient_eye_combinations.csv",
        ["patient_id", "exam_eye", "row_count"],
    )
    write_csv(frame.loc[full_duplicate_mask], out / "duplicate_metadata_rows.csv", list(frame.columns))
    if duplicate_image_id_mask.any():
        flag("RED", "DUPLICATE_IMAGE_ID", f"{int(duplicate_image_id_mask.sum())} rows have duplicated image_id values.")
    if not duplicate_patient_eye.empty:
        flag(
            "YELLOW", "DUPLICATE_PATIENT_EYE",
            f"{len(duplicate_patient_eye)} patient-eye combinations ({int(duplicate_patient_eye.row_count.sum())} rows) are repeated.",
        )
    if full_duplicate_mask.any():
        flag("RED", "DUPLICATE_METADATA_ROW", f"{int(full_duplicate_mask.sum())} rows are exact metadata duplicates.")

    checksum_path = root / "SHA256SUMS.txt"
    published_checksums = parse_sha256sums(checksum_path)
    metadata_paths = [
        labels_path, root / "LICENSE.txt", checksum_path, root / "index.html",
        root / "README.txt", root / "RECORDS",
    ]
    input_rows = []
    for path in metadata_paths:
        exists = path.is_file()
        relative_path = path.relative_to(root).as_posix()
        actual_hash = sha256_file(path) if exists else None
        expected_hash = published_checksums.get(relative_path)
        input_rows.append({
            "role": "label_metadata" if path == labels_path else "release_metadata",
            "relative_path": relative_path, "exists": bool(exists),
            "bytes": path.stat().st_size if exists else None, "sha256": actual_hash,
            "published_sha256": expected_hash,
            "published_match": (actual_hash == expected_hash) if exists and expected_hash else None,
        })
        if exists and expected_hash and actual_hash != expected_hash:
            flag("RED", "METADATA_CHECKSUM_MISMATCH", f"Checksum mismatch: {relative_path}")
    input_manifest = pd.DataFrame(input_rows)
    write_csv(input_manifest, out / "input_manifest.csv")

    published_image_paths = {
        path for path in published_checksums
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    }
    unlisted_images = sorted(set(relative_images) - published_image_paths)
    manifest_missing_images = sorted(published_image_paths - set(relative_images))
    if unlisted_images:
        flag("RED", "IMAGE_NOT_IN_SHA256SUMS", f"{len(unlisted_images)} image files are absent from SHA256SUMS.txt.")
    if manifest_missing_images:
        flag("RED", "SHA256SUMS_IMAGE_MISSING", f"{len(manifest_missing_images)} image entries in SHA256SUMS.txt lack files.")

    duplicate_hash_groups: list[tuple[str, list[str]]] = []
    image_hash_mismatches = 0
    if args.hash_images:
        hash_rows = []
        actual_hash_groups: dict[str, list[str]] = defaultdict(list)
        print(f"Hashing {len(images)} image files for exact-byte integrity...", flush=True)
        for number, (path, relative_path) in enumerate(zip(images, relative_images), start=1):
            actual_hash = sha256_file(path)
            expected_hash = published_checksums.get(relative_path)
            match = actual_hash == expected_hash if expected_hash else None
            if match is False:
                image_hash_mismatches += 1
            hash_rows.append({
                "relative_path": relative_path, "bytes": path.stat().st_size,
                "sha256": actual_hash, "published_sha256": expected_hash,
                "published_match": match,
            })
            actual_hash_groups[actual_hash].append(relative_path)
            if number % 2_000 == 0 or number == len(images):
                print(f"  hashed {number}/{len(images)}", flush=True)
        write_csv(pd.DataFrame(hash_rows), out / "image_sha256_manifest.csv")
        duplicate_hash_groups = [
            (digest, paths) for digest, paths in actual_hash_groups.items() if len(paths) > 1
        ]
        duplicate_rows = [
            {"duplicate_group": group_number, "sha256": digest, "relative_path": relative_path}
            for group_number, (digest, paths) in enumerate(duplicate_hash_groups, start=1)
            for relative_path in paths
        ]
        write_csv(
            pd.DataFrame(duplicate_rows), out / "exact_duplicate_images.csv",
            ["duplicate_group", "sha256", "relative_path"],
        )
        if image_hash_mismatches:
            flag("RED", "IMAGE_CHECKSUM_MISMATCH", f"{image_hash_mismatches} images fail published SHA-256 verification.")
        if duplicate_hash_groups:
            flag("YELLOW", "EXACT_DUPLICATE_IMAGE_BYTES", f"{len(duplicate_hash_groups)} exact duplicate image groups found.")
    else:
        write_csv(
            pd.DataFrame(), out / "image_sha256_manifest.csv",
            ["relative_path", "bytes", "sha256", "published_sha256", "published_match"],
        )
        write_csv(
            pd.DataFrame(), out / "exact_duplicate_images.csv",
            ["duplicate_group", "sha256", "relative_path"],
        )
        flag("YELLOW", "IMAGE_HASHING_DISABLED", "Image byte hashes were not computed; rerun with --hash-images for the required full check.")

    missingness = pd.DataFrame({
        "column": frame.columns,
        "missing_n": [int(frame[column].isna().sum()) for column in frame.columns],
        "missing_pct": [float(100 * frame[column].isna().mean()) for column in frame.columns],
        "unique_nonmissing": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        "dtype_read": [str(frame[column].dtype) for column in frame.columns],
    }).sort_values(["missing_pct", "column"], ascending=[False, True])
    write_csv(missingness, out / "missingness.csv")

    # Patient and bilateral structure.
    patient_structure = patient_structure_table(frame)
    write_csv(patient_structure, out / "patient_structure.csv")
    valid_patient_ids = set(patient_structure.loc[patient_structure.valid_unique_od_os_pair == 1, "patient_id"])
    paired_source = frame[frame.patient_id.isin(valid_patient_ids)].copy()
    od = paired_source.loc[paired_source.exam_eye == RIGHT_EYE].set_index("patient_id").sort_index()
    os_eye = paired_source.loc[paired_source.exam_eye == LEFT_EYE].set_index("patient_id").sort_index()
    if not od.index.equals(os_eye.index):
        raise RuntimeError("Internal bilateral construction error: OD and OS patient indexes differ")

    bilateral_counts = {
        "patients_total": int(len(patient_structure)),
        "patients_with_both_eye_codes_including_duplicates": int(patient_structure.has_both_eye_codes.sum()),
        "valid_unique_od_os_pairs": int(patient_structure.valid_unique_od_os_pair.sum()),
        "od_only_patients": int((patient_structure.eye_pattern == "OD-only").sum()),
        "os_only_patients": int((patient_structure.eye_pattern == "OS-only").sum()),
        "single_eye_patients_total": int(patient_structure.eye_pattern.isin(["OD-only", "OS-only"]).sum()),
        "patients_with_gt2_images": int((patient_structure.n_images > 2).sum()),
        "patients_with_duplicate_eye_records": int(patient_structure.has_duplicate_eye_record.sum()),
        "duplicate_patient_eye_combinations": int(len(duplicate_patient_eye)),
        "patients_on_multiple_cameras": int((patient_structure.n_cameras > 1).sum()),
    }

    patient_level_variables = [
        "patient_age", "patient_sex", "camera", "comorbidities",
        "diabetes_time_y", "insulin", "diabetes", "nationality",
    ]
    inconsistency_rows = []
    for variable in patient_level_variables:
        counts = frame.groupby("patient_id", dropna=False)[variable].nunique(dropna=False)
        inconsistent_ids = counts[counts > 1]
        inconsistency_rows.append({
            "variable": variable,
            "patients_with_inconsistent_values": int(len(inconsistent_ids)),
            "max_values_within_patient": int(counts.max()) if len(counts) else 0,
        })
    patient_metadata_inconsistency = pd.DataFrame(inconsistency_rows)
    write_csv(patient_metadata_inconsistency, out / "patient_metadata_inconsistency.csv")
    if int(patient_metadata_inconsistency.patients_with_inconsistent_values.sum()) > 0:
        flag("YELLOW", "PATIENT_METADATA_INCONSISTENCY", "Some nominal patient-level fields vary across image rows; see patient_metadata_inconsistency.csv.")

    # Labels and pairwise-complete co-occurrence.
    label_rows = []
    for label in BINARY_LABELS:
        numeric = numeric_views[label]
        valid = numeric.isin([0, 1])
        positive = numeric.eq(1) & valid
        negative = numeric.eq(0) & valid
        label_rows.append({
            "label": label, "evaluable_n": int(valid.sum()),
            "positive_n": int(positive.sum()), "negative_n": int(negative.sum()),
            "missing_n": int(frame[label].isna().sum()),
            "unexpected_n": int((frame[label].notna() & ~valid).sum()),
            "prevalence": safe_rate(int(positive.sum()), int(valid.sum())),
            "positive_patients": int(frame.loc[positive, "patient_id"].nunique()),
        })
    label_summary = pd.DataFrame(label_rows).sort_values("positive_n", ascending=False)
    write_csv(label_summary, out / "label_summary.csv")

    cooccurrence = pd.DataFrame(index=BINARY_LABELS, columns=BINARY_LABELS, dtype="Int64")
    jaccard = pd.DataFrame(index=BINARY_LABELS, columns=BINARY_LABELS, dtype=float)
    for label_a in BINARY_LABELS:
        a = numeric_views[label_a]
        for label_b in BINARY_LABELS:
            b = numeric_views[label_b]
            pair_valid = a.isin([0, 1]) & b.isin([0, 1])
            a_pos = a.eq(1) & pair_valid
            b_pos = b.eq(1) & pair_valid
            intersection = int((a_pos & b_pos).sum())
            union = int((a_pos | b_pos).sum())
            cooccurrence.loc[label_a, label_b] = intersection
            jaccard.loc[label_a, label_b] = safe_rate(intersection, union)
    cooccurrence.to_csv(out / "label_cooccurrence_counts.csv", index=True)
    jaccard.to_csv(out / "label_cooccurrence_jaccard.csv", index=True)

    binary_numeric = frame[BINARY_LABELS].apply(pd.to_numeric, errors="coerce")
    image_multilabel_valid = binary_numeric.isin([0, 1]).all(axis=1)
    image_label_cardinality = binary_numeric.eq(1).sum(axis=1).loc[image_multilabel_valid]
    image_cardinality_summary = (
        image_label_cardinality.value_counts().sort_index()
        .rename_axis("positive_label_count").rename("count").reset_index()
    )
    image_cardinality_summary.insert(0, "unit", "image")
    image_cardinality_summary["evaluable_units"] = int(image_multilabel_valid.sum())
    image_cardinality_summary["rate"] = (
        image_cardinality_summary["count"] / image_cardinality_summary["evaluable_units"]
    )
    write_csv(image_cardinality_summary, out / "multilabel_cardinality_summary.csv")

    # Quality and anatomy value summaries.
    quality_rows = []
    for quality_dimension in QUALITY_COLS:
        numeric = numeric_views[quality_dimension]
        valid = numeric.isin([1, 2])
        quality_rows.append({
            "quality_dimension": quality_dimension, "evaluable_n": int(valid.sum()),
            "normal_n": int((numeric.eq(1) & valid).sum()),
            "abnormal_n": int((numeric.eq(2) & valid).sum()),
            "missing_n": int(frame[quality_dimension].isna().sum()),
            "unexpected_n": int((frame[quality_dimension].notna() & ~valid).sum()),
            "abnormal_rate": safe_rate(int((numeric.eq(2) & valid).sum()), int(valid.sum())),
        })
    quality_summary = pd.DataFrame(quality_rows)
    write_csv(quality_summary, out / "quality_summary.csv")

    anatomy_rows = []
    for anatomy_dimension in ANATOMY_COLS:
        summary = code_summary(frame[anatomy_dimension], {1, 2})
        numeric = numeric_views[anatomy_dimension]
        anatomy_rows.append({
            "anatomy_dimension": anatomy_dimension, **summary,
            "normal_n": int(numeric.eq(1).sum()), "abnormal_n": int(numeric.eq(2).sum()),
        })
    anatomy_summary = pd.DataFrame(anatomy_rows)
    write_csv(anatomy_summary, out / "anatomy_summary.csv")

    overall_quality = (
        frame["quality"].value_counts(dropna=False).rename_axis("quality_value").rename("count").reset_index()
    )
    overall_quality["rate"] = overall_quality["count"] / len(frame)
    write_csv(overall_quality, out / "quality_overall_summary.csv")

    # Camera/device structure. All associations are descriptive.
    camera_summary = (
        frame.groupby("camera", dropna=False)
        .agg(images=("image_id", "size"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    camera_summary["image_fraction"] = camera_summary.images / len(frame)
    camera_summary["patient_fraction"] = camera_summary.patients / n_patients
    write_csv(camera_summary, out / "camera_summary.csv")

    camera_disease_rows = []
    for camera, group in frame.groupby("camera", dropna=False):
        for label in BINARY_LABELS:
            numeric = pd.to_numeric(group[label], errors="coerce")
            valid = numeric.isin([0, 1])
            positive_n = int((numeric.eq(1) & valid).sum())
            camera_disease_rows.append({
                "camera": camera, "label": label, "evaluable_images": int(valid.sum()),
                "positive_images": positive_n,
                "prevalence": safe_rate(positive_n, int(valid.sum())),
            })
    camera_disease = pd.DataFrame(camera_disease_rows)
    write_csv(camera_disease, out / "camera_disease_distribution.csv")

    camera_quality_rows = []
    for camera, group in frame.groupby("camera", dropna=False):
        for quality_dimension in QUALITY_COLS:
            numeric = pd.to_numeric(group[quality_dimension], errors="coerce")
            valid = numeric.isin([1, 2])
            abnormal_n = int((numeric.eq(2) & valid).sum())
            camera_quality_rows.append({
                "camera": camera, "quality_dimension": quality_dimension,
                "evaluable_images": int(valid.sum()),
                "normal_images": int((numeric.eq(1) & valid).sum()),
                "abnormal_images": abnormal_n,
                "abnormal_rate": safe_rate(abnormal_n, int(valid.sum())),
            })
    camera_quality = pd.DataFrame(camera_quality_rows)
    write_csv(camera_quality, out / "camera_quality_distribution.csv")

    quality_disease_rows = []
    for quality_dimension in QUALITY_COLS:
        quality_numeric = numeric_views[quality_dimension]
        for label in BINARY_LABELS:
            label_numeric = numeric_views[label]
            valid = quality_numeric.isin([1, 2]) & label_numeric.isin([0, 1])
            q_bad = quality_numeric.eq(2) & valid
            y_pos = label_numeric.eq(1) & valid
            quality_disease_rows.append({
                "quality_dimension": quality_dimension, "label": label,
                "evaluable_images": int(valid.sum()),
                "normal_label_negative": int((~q_bad & ~y_pos & valid).sum()),
                "normal_label_positive": int((~q_bad & y_pos & valid).sum()),
                "abnormal_label_negative": int((q_bad & ~y_pos).sum()),
                "abnormal_label_positive": int((q_bad & y_pos).sum()),
                "abnormal_rate_label_negative": safe_rate(int((q_bad & ~y_pos).sum()), int((~y_pos & valid).sum())),
                "abnormal_rate_label_positive": safe_rate(int((q_bad & y_pos).sum()), int((y_pos & valid).sum())),
            })
    quality_disease = pd.DataFrame(quality_disease_rows)
    write_csv(quality_disease, out / "quality_disease_associations.csv")

    camera_demographic_rows = []
    for camera, group in frame.groupby("camera", dropna=False):
        per_patient = group.groupby("patient_id", dropna=False).agg(
            age_median=("patient_age", "median"),
            age_nunique=("patient_age", lambda values: values.nunique(dropna=True)),
            sex_first=("patient_sex", "first"),
            sex_nunique=("patient_sex", lambda values: values.nunique(dropna=True)),
        )
        valid_age = per_patient.loc[per_patient.age_nunique <= 1, "age_median"].dropna()
        valid_sex = per_patient.loc[per_patient.sex_nunique <= 1, "sex_first"].dropna()
        camera_demographic_rows.append({
            "camera": camera, "patients": int(len(per_patient)),
            "age_evaluable_patients": int(len(valid_age)),
            "age_inconsistent_patients": int((per_patient.age_nunique > 1).sum()),
            "age_mean": float(valid_age.mean()) if len(valid_age) else None,
            "age_sd": float(valid_age.std(ddof=1)) if len(valid_age) > 1 else None,
            "age_median": float(valid_age.median()) if len(valid_age) else None,
            "age_min": float(valid_age.min()) if len(valid_age) else None,
            "age_max": float(valid_age.max()) if len(valid_age) else None,
            "sex_evaluable_patients": int(len(valid_sex)),
            "sex_inconsistent_patients": int((per_patient.sex_nunique > 1).sum()),
            "male_patients": int(valid_sex.eq(1).sum()),
            "female_patients": int(valid_sex.eq(2).sum()),
            "unexpected_sex_patients": int((~valid_sex.isin([1, 2])).sum()),
        })
    camera_demographics = pd.DataFrame(camera_demographic_rows)
    write_csv(camera_demographics, out / "camera_demographics.csv")

    multi_camera_patients = patient_structure.loc[
        patient_structure.n_cameras > 1, ["patient_id", "n_images", "cameras"]
    ]
    write_csv(multi_camera_patients, out / "patients_multiple_cameras.csv", ["patient_id", "n_images", "cameras"])
    camera_memberships = frame.groupby("patient_id")["camera"].agg(
        lambda values: "|".join(sorted(values.dropna().astype(str).unique()))
    )
    camera_patient_overlap = camera_memberships.value_counts().rename_axis("camera_membership").rename("patients").reset_index()
    write_csv(camera_patient_overlap, out / "camera_patient_overlap.csv")

    # Clean bilateral tables: exactly one OD and one OS record per patient.
    bilateral_label_rows = []
    for label in BINARY_LABELS:
        right = pd.to_numeric(od[label], errors="coerce")
        left = pd.to_numeric(os_eye[label], errors="coerce")
        valid = right.isin([0, 1]) & left.isin([0, 1])
        r = right[valid]
        l = left[valid]
        both_positive = int((r.eq(1) & l.eq(1)).sum())
        both_negative = int((r.eq(0) & l.eq(0)).sum())
        od_positive = int((r.eq(1) & l.eq(0)).sum())
        os_positive = int((r.eq(0) & l.eq(1)).sum())
        evaluable = int(valid.sum())
        bilateral_label_rows.append({
            "label": label, "evaluable_pairs": evaluable,
            "both_positive": both_positive, "both_negative": both_negative,
            "od_positive_os_negative": od_positive,
            "od_negative_os_positive": os_positive,
            "discordant_total": od_positive + os_positive,
            "discordance_rate": safe_rate(od_positive + os_positive, evaluable),
            "pairs_with_at_least_one_positive_eye": both_positive + od_positive + os_positive,
            "discordance_rate_among_pairs_with_positive_eye": safe_rate(
                od_positive + os_positive, both_positive + od_positive + os_positive
            ),
        })
    bilateral_label = pd.DataFrame(bilateral_label_rows)
    write_csv(bilateral_label, out / "bilateral_label_concordance.csv")

    od_binary = od[BINARY_LABELS].apply(pd.to_numeric, errors="coerce")
    os_binary = os_eye[BINARY_LABELS].apply(pd.to_numeric, errors="coerce")
    bilateral_multilabel_valid = (
        od_binary.isin([0, 1]).all(axis=1) & os_binary.isin([0, 1]).all(axis=1)
    )
    od_positive_vector = od_binary.eq(1)
    os_positive_vector = os_binary.eq(1)
    shared_positive_count = (od_positive_vector & os_positive_vector).sum(axis=1)
    union_positive_count = (od_positive_vector | os_positive_vector).sum(axis=1)
    vector_discordant = od_positive_vector.ne(os_positive_vector).any(axis=1)
    positive_union = bilateral_multilabel_valid & union_positive_count.gt(0)
    pair_jaccard = pd.Series(np.nan, index=od.index, dtype=float)
    pair_jaccard.loc[positive_union] = (
        shared_positive_count.loc[positive_union] / union_positive_count.loc[positive_union]
    )
    bilateral_multilabel_summary = pd.DataFrame([{
        "evaluable_pairs": int(bilateral_multilabel_valid.sum()),
        "pairs_no_positive_label_either_eye": int(
            (bilateral_multilabel_valid & union_positive_count.eq(0)).sum()
        ),
        "pairs_with_at_least_one_positive_label": int(positive_union.sum()),
        "pairs_with_identical_label_vector": int(
            (bilateral_multilabel_valid & ~vector_discordant).sum()
        ),
        "pairs_with_discordant_label_vector": int(
            (bilateral_multilabel_valid & vector_discordant).sum()
        ),
        "label_vector_discordance_rate_all_pairs": safe_rate(
            int((bilateral_multilabel_valid & vector_discordant).sum()),
            int(bilateral_multilabel_valid.sum()),
        ),
        "label_vector_discordance_rate_positive_union": safe_rate(
            int((positive_union & vector_discordant).sum()), int(positive_union.sum())
        ),
        "mean_od_os_label_jaccard_positive_union": float(pair_jaccard.loc[positive_union].mean()),
        "median_od_os_label_jaccard_positive_union": float(pair_jaccard.loc[positive_union].median()),
    }])
    write_csv(bilateral_multilabel_summary, out / "bilateral_multilabel_summary.csv")

    pair_union_cardinality_summary = (
        union_positive_count.loc[bilateral_multilabel_valid].value_counts().sort_index()
        .rename_axis("positive_label_count").rename("count").reset_index()
    )
    pair_union_cardinality_summary.insert(0, "unit", "bilateral_pair_union")
    pair_union_cardinality_summary["evaluable_units"] = int(bilateral_multilabel_valid.sum())
    pair_union_cardinality_summary["rate"] = (
        pair_union_cardinality_summary["count"] /
        pair_union_cardinality_summary["evaluable_units"]
    )
    write_csv(
        pd.concat([image_cardinality_summary, pair_union_cardinality_summary], ignore_index=True),
        out / "multilabel_cardinality_summary.csv",
    )

    bilateral_quality_rows = []
    quality_pair_discordance = pd.DataFrame(index=od.index)
    for quality_dimension in QUALITY_COLS:
        right = pd.to_numeric(od[quality_dimension], errors="coerce")
        left = pd.to_numeric(os_eye[quality_dimension], errors="coerce")
        valid = right.isin([1, 2]) & left.isin([1, 2])
        r = right[valid]
        l = left[valid]
        both_normal = int((r.eq(1) & l.eq(1)).sum())
        both_abnormal = int((r.eq(2) & l.eq(2)).sum())
        od_abnormal = int((r.eq(2) & l.eq(1)).sum())
        os_abnormal = int((r.eq(1) & l.eq(2)).sum())
        evaluable = int(valid.sum())
        bilateral_quality_rows.append({
            "quality_dimension": quality_dimension, "evaluable_pairs": evaluable,
            "both_normal": both_normal, "both_abnormal": both_abnormal,
            "od_abnormal_os_normal": od_abnormal,
            "od_normal_os_abnormal": os_abnormal,
            "discordant_total": od_abnormal + os_abnormal,
            "discordance_rate": safe_rate(od_abnormal + os_abnormal, evaluable),
        })
        quality_pair_discordance[quality_dimension] = (right.ne(left) & valid).astype("boolean")
    bilateral_quality = pd.DataFrame(bilateral_quality_rows)
    write_csv(bilateral_quality, out / "bilateral_quality_concordance.csv")
    all_quality_evaluable = pd.Series(True, index=od.index)
    for quality_dimension in QUALITY_COLS:
        all_quality_evaluable &= pd.to_numeric(od[quality_dimension], errors="coerce").isin([1, 2])
        all_quality_evaluable &= pd.to_numeric(os_eye[quality_dimension], errors="coerce").isin([1, 2])
    any_quality_discordance = quality_pair_discordance.fillna(False).any(axis=1)
    quality_pair_summary = pd.DataFrame([{
        "valid_unique_od_os_pairs": int(len(od)),
        "all_four_quality_dimensions_evaluable": int(all_quality_evaluable.sum()),
        "pairs_discordant_on_any_quality_dimension": int((any_quality_discordance & all_quality_evaluable).sum()),
        "any_quality_discordance_rate": safe_rate(
            int((any_quality_discordance & all_quality_evaluable).sum()), int(all_quality_evaluable.sum())
        ),
    }])
    write_csv(quality_pair_summary, out / "bilateral_quality_patient_summary.csv")

    bilateral_pair_structure = pd.DataFrame({
        "patient_id": od.index,
        "od_image_id": od["image_id"].values,
        "os_image_id": os_eye["image_id"].values,
        "od_camera": od["camera"].values,
        "os_camera": os_eye["camera"].values,
        "same_camera": od["camera"].eq(os_eye["camera"]).astype(int).values,
        "od_positive_label_count": od_positive_vector.sum(axis=1).values,
        "os_positive_label_count": os_positive_vector.sum(axis=1).values,
        "shared_positive_label_count": shared_positive_count.values,
        "union_positive_label_count": union_positive_count.values,
        "label_vector_discordant": vector_discordant.astype(int).values,
        "od_os_label_jaccard": pair_jaccard.values,
        "any_quality_dimension_discordant": any_quality_discordance.astype(int).values,
    })
    write_csv(bilateral_pair_structure, out / "bilateral_pair_structure.csv")

    valid_camera = od["camera"].notna() & os_eye["camera"].notna()
    same_camera = od.loc[valid_camera, "camera"].eq(os_eye.loc[valid_camera, "camera"])
    bilateral_camera = pd.DataFrame([{
        "evaluable_pairs": int(valid_camera.sum()), "same_camera": int(same_camera.sum()),
        "different_camera": int((~same_camera).sum()),
        "different_camera_rate": safe_rate(int((~same_camera).sum()), int(valid_camera.sum())),
    }])
    write_csv(bilateral_camera, out / "bilateral_camera_concordance.csv")
    camera_pair_table = (
        pd.DataFrame({
            "od_camera": od.loc[valid_camera, "camera"],
            "os_camera": os_eye.loc[valid_camera, "camera"],
        }).value_counts().rename("pairs").reset_index()
    )
    write_csv(camera_pair_table, out / "bilateral_camera_pairs.csv", ["od_camera", "os_camera", "pairs"])

    dr_grade_rows = []
    for grade in ORDINAL_LABELS:
        right = pd.to_numeric(od[grade], errors="coerce")
        left = pd.to_numeric(os_eye[grade], errors="coerce")
        valid = right.isin(range(5)) & left.isin(range(5))
        differences = (right[valid] - left[valid]).abs().astype(int)
        for difference, count in differences.value_counts().sort_index().items():
            dr_grade_rows.append({
                "grade_system": grade, "abs_grade_difference": int(difference),
                "count": int(count), "rate": safe_rate(int(count), int(len(differences))),
                "evaluable_pairs": int(len(differences)),
                "mean_abs_grade_difference": float(differences.mean()) if len(differences) else None,
                "median_abs_grade_difference": float(differences.median()) if len(differences) else None,
                "max_abs_grade_difference": int(differences.max()) if len(differences) else None,
            })
    dr_grade = pd.DataFrame(dr_grade_rows)
    write_csv(
        dr_grade, out / "bilateral_dr_grade_difference.csv",
        ["grade_system", "abs_grade_difference", "count", "rate", "evaluable_pairs",
         "mean_abs_grade_difference", "median_abs_grade_difference", "max_abs_grade_difference"],
    )

    # Naive image-level split leakage simulation. This is not a modeling split.
    leakage_rows = []
    simulation_frame = frame.loc[frame["patient_id"].notna(), ["patient_id"]].reset_index(drop=True)
    unique_patient_total = int(simulation_frame.patient_id.nunique())
    repeated_patient_total = int((simulation_frame.groupby("patient_id").size() > 1).sum())
    rng = np.random.default_rng(20260828)
    for repetition in range(100):
        indexes = rng.permutation(len(simulation_frame))
        n_train = int(round(0.70 * len(simulation_frame)))
        n_validation = int(round(0.15 * len(simulation_frame)))
        split = np.empty(len(simulation_frame), dtype=object)
        split[indexes[:n_train]] = "train"
        split[indexes[n_train:n_train + n_validation]] = "validation"
        split[indexes[n_train + n_validation:]] = "test"
        temporary = simulation_frame.assign(split=split)
        patient_split_count = temporary.groupby("patient_id")["split"].nunique()
        affected = int((patient_split_count > 1).sum())
        leakage_rows.append({
            "repetition": repetition, "patients_total": unique_patient_total,
            "patients_with_repeated_images": repeated_patient_total,
            "patients_in_multiple_splits": affected,
            "fraction_all_patients_in_multiple_splits": safe_rate(affected, unique_patient_total),
            "fraction_repeated_patients_in_multiple_splits": safe_rate(affected, repeated_patient_total),
        })
    leakage_simulation = pd.DataFrame(leakage_rows)
    write_csv(leakage_simulation, out / "image_level_split_leakage_simulation.csv")
    if leakage_simulation.patients_in_multiple_splits.mean() > 0:
        flag("RED", "IMAGE_LEVEL_SPLIT_LEAKAGE", "Naive image-level splitting places fellow-eye/repeated records in different partitions.")

    image_id_pattern = frame["image_id"].astype(str).str.fullmatch(r"img\d+").all()
    filename_leakage = pd.DataFrame([{
        "check": "identifier_pattern",
        "result": "all sequential img+digits" if image_id_pattern else "mixed pattern",
        "interpretation": (
            "No explicit patient, eye, or disease token is present in the filename pattern. "
            "Sequential order may still proxy acquisition workflow/time; image_id must not be a predictor."
        ),
    }])
    write_csv(filename_leakage, out / "filename_leakage_audit.csv")

    variable_leakage = build_variable_leakage_table(list(frame.columns))
    write_csv(variable_leakage, out / "r0_variable_availability_and_leakage.csv")

    schema_consistent = not missing_expected_columns and not legacy_found
    checksum_metadata_failures = int((input_manifest.published_match == False).sum())  # noqa: E712
    version_status = (
        "consistent_with_BRSET_v1.0.2"
        if root.name == EXPECTED_VERSION and schema_consistent
        and n_rows == EXPECTED_N_IMAGES and n_patients == EXPECTED_N_PATIENTS
        else "not_fully_verified"
    )

    summary = {
        "audit": "R0 BRSET Dataset Integrity & Scientific Feasibility Audit",
        "dataset_version_expected": EXPECTED_VERSION,
        "dataset_version_verification": version_status,
        "source_root": str(root), "labels_file": labels_path.name,
        "labels_sha256": label_sha256,
        "counts": {
            "metadata_rows": n_rows, "unique_image_ids": n_image_ids,
            "unique_patients": n_patients, "image_files": int(len(images)),
            "missing_image_files": int(len(missing_image_ids)),
            "orphan_image_files": int(len(orphan_file_stems)),
        },
        "schema": {
            "columns": list(frame.columns), "legacy_columns_found": legacy_found,
            "missing_expected_columns": missing_expected_columns,
            "unexpected_columns": unexpected_columns,
            "value_anomaly_rows": int(sum(row["count"] for row in value_anomalies)),
        },
        "integrity": {
            "duplicate_image_id_rows": int(duplicate_image_id_mask.sum()),
            "duplicate_patient_eye_combinations": int(len(duplicate_patient_eye)),
            "duplicate_patient_eye_rows": int(duplicate_patient_eye.row_count.sum()) if len(duplicate_patient_eye) else 0,
            "patients_with_duplicate_eye_records": bilateral_counts["patients_with_duplicate_eye_records"],
            "exact_duplicate_metadata_rows": int(full_duplicate_mask.sum()),
            "all_images_hashed": bool(args.hash_images),
            "image_checksum_mismatches": int(image_hash_mismatches) if args.hash_images else None,
            "metadata_checksum_mismatches": checksum_metadata_failures,
            "exact_duplicate_image_groups": int(len(duplicate_hash_groups)) if args.hash_images else None,
        },
        "bilateral": bilateral_counts,
        "bilateral_quality": quality_pair_summary.iloc[0].to_dict(),
        "bilateral_multilabel": bilateral_multilabel_summary.iloc[0].to_dict(),
        "bilateral_camera": bilateral_camera.iloc[0].to_dict(),
        "leakage_simulation": {
            "repetitions": int(len(leakage_simulation)),
            "mean_patients_in_multiple_splits": float(leakage_simulation.patients_in_multiple_splits.mean()),
            "min_patients_in_multiple_splits": int(leakage_simulation.patients_in_multiple_splits.min()),
            "max_patients_in_multiple_splits": int(leakage_simulation.patients_in_multiple_splits.max()),
            "mean_fraction_all_patients": float(leakage_simulation.fraction_all_patients_in_multiple_splits.mean()),
            "min_fraction_all_patients": float(leakage_simulation.fraction_all_patients_in_multiple_splits.min()),
            "max_fraction_all_patients": float(leakage_simulation.fraction_all_patients_in_multiple_splits.max()),
        },
        "findings": findings,
        "guardrails": {
            "raw_dataset_modified": False, "deep_learning_performed": False,
            "model_training_performed": False,
        },
    }
    with (out / "r0_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    report = [
        "# R0 — BRSET Dataset Integrity & Scientific Feasibility Audit", "",
        "This report contains dataset facts and descriptive associations only. It contains no model training or causal claims.", "",
        "## Dataset verification", "",
        f"- Expected/release version: `{EXPECTED_VERSION}`",
        f"- Verification status: `{version_status}`",
        f"- Local metadata filename: `{labels_path.name}`",
        f"- Metadata SHA-256: `{label_sha256}`",
        f"- Metadata rows / unique image IDs / unique patients: {n_rows:,} / {n_image_ids:,} / {n_patients:,}",
        f"- Image files: {len(images):,}; missing labeled files: {len(missing_image_ids):,}; orphan files: {len(orphan_file_stems):,}",
        f"- Full image SHA-256 verification performed: {'yes' if args.hash_images else 'no'}", "",
        "## Audit findings", "",
    ]
    if findings:
        report.extend(f"- **{item['level']} — {item['code']}**: {item['message']}" for item in findings)
    else:
        report.append("- No flags generated.")
    report.extend([
        "", "## Patient and bilateral structure", "",
        *[f"- {key}: {value}" for key, value in bilateral_counts.items()], "",
        "A valid pair requires exactly one OD (`exam_eye=1`) and one OS (`exam_eye=2`) record. Patients with repeated patient-eye records are retained in integrity tables but excluded from clean bilateral calculations.",
        "", "## Label prevalence", "", markdown_table(label_summary),
        "", "## Bilateral label concordance", "",
        "Observed OD/OS label discordance can reflect real biological asymmetry and is not classified as dataset error.",
        "", markdown_table(bilateral_label),
        "", "## Bilateral multilabel-vector structure", "",
        markdown_table(bilateral_multilabel_summary),
        "", "## Image-quality distribution", "", markdown_table(quality_summary),
        "", "## Bilateral image-quality concordance", "", markdown_table(bilateral_quality),
        "", markdown_table(quality_pair_summary),
        "", "## Camera/device structure", "", markdown_table(camera_summary),
        "", markdown_table(bilateral_camera), "",
        "Camera strata may also encode site, time period, population, operator, or workflow. These descriptive differences are not identified device effects.",
        "", "## Naive image-level split leakage", "",
        (
            f"Across 100 simulated 70/15/15 image-level partitions, a mean of "
            f"{leakage_simulation.patients_in_multiple_splits.mean():.1f} patients "
            f"({leakage_simulation.fraction_all_patients_in_multiple_splits.mean():.3%} of all patients) "
            f"appeared in more than one partition. The range was "
            f"{leakage_simulation.patients_in_multiple_splits.min()}-"
            f"{leakage_simulation.patients_in_multiple_splits.max()} patients."
        ),
        "", "Any future train/validation/test partition must be grouped by `patient_id`.",
        "", "## Interpretation guardrails", "",
        "- Dataset fact: v1.0.2 documents `exam_eye=1` as right/OD and `exam_eye=2` as left/OS.",
        "- Dataset fact: supplied quality fields are human assessments; they are not model uncertainty.",
        "- Descriptive association: camera, disease, quality, age, and sex tables do not establish causation.",
        "- Future hypothesis: prediction disagreement may be evaluated only after model protocol freeze; R0 does not test it.",
        "- Patient identifiers and sequential image identifiers are never candidate predictors.",
        "", "## Integrity statement", "",
        "- raw dataset modified: NO", "- deep learning performed: NO",
        "- model training performed: NO", "",
    ])
    (out / "R0_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    level_counts = Counter(item["level"] for item in findings)
    print(f"R0 audit complete: {out}")
    print(f"Findings: RED={level_counts['RED']} YELLOW={level_counts['YELLOW']} INFO={level_counts['INFO']}")
    print(f"Report: {out / 'R0_REPORT.md'}")
    print(f"Summary: {out / 'r0_summary.json'}")


if __name__ == "__main__":
    main()
