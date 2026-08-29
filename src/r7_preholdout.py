"""Pre-holdout validation and one-way release gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .r7_calibration import binary_nll
from .r7_config import project_path, sha256_file
from .r7_firewall import create_holdout_release_marker, verify_partition


def _source_files(config: dict[str, Any]) -> list[Path]:
    root = project_path(config, ".")
    paths = [
        root / "configs" / "r7_frozen_config.json",
        root / "requirements-r7.txt",
        root / "r7_run.py",
        root / "r7_artifacts" / "environment_freeze.txt",
    ]
    paths.extend(sorted((root / "src").glob("r7_*.py")))
    paths.extend(sorted((root / "tests").glob("test_r7_*.py")))
    paths.extend(sorted((root / "r7_implementation").glob("*.md")))
    return paths


def validate_pre_holdout(config: dict[str, Any]) -> dict[str, Any]:
    result_dir = project_path(config, "r7_results")
    report_path = project_path(config, config["firewall"]["pre_holdout_report"])
    machine_path = result_dir / "pre_holdout_validation.json"
    if report_path.exists() or machine_path.exists():
        raise RuntimeError("Refusing to overwrite pre-holdout validation")
    if project_path(config, config["firewall"]["holdout_release_marker"]).exists():
        raise RuntimeError("Holdout was already released")

    integrity = verify_partition(config)
    development_completion = project_path(
        config, config["firewall"]["development_completion_marker"]
    )
    if not development_completion.is_file():
        raise RuntimeError("Development completion marker missing")
    completion = json.loads(development_completion.read_text(encoding="utf-8"))
    if completion.get("holdout_accessed") is not False:
        raise RuntimeError("Development marker does not assert holdout isolation")

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"Pre-holdout validation failed: {name}: {detail}")

    record(
        "frozen split hash",
        integrity.manifest_sha256 == config["partition"]["manifest_sha256"],
        integrity.manifest_sha256,
    )
    record(
        "holdout embedding directory empty",
        not any(
            project_path(config, config["firewall"]["holdout_embedding_dir"]).iterdir()
        ),
        "No holdout embedding artifact exists",
    )
    record(
        "model registry",
        config["model"]["implementation"] == "torchvision.models.convnext_tiny"
        and config["model"]["weights_enum"]
        == "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
        and config["model"]["feature_dimension"] == 768,
        "Single frozen ConvNeXt-Tiny representation",
    )
    record(
        "preprocessing freeze",
        config["preprocessing"]["input_size"] == 224
        and config["preprocessing"]["horizontal_flip"] is False
        and config["preprocessing"]["augmentation"] == "none",
        "224-pixel letterbox; no flip/augmentation",
    )
    record(
        "uncertainty registry",
        config["reliability"]["epistemic_uncertainty_candidates"] == [],
        "No epistemic estimator introduced",
    )
    record(
        "endpoint hierarchy",
        config["endpoint_hierarchy"]
        == ["drusen", "increased_cup_disc", "diabetic_retinopathy"],
        "Frozen fixed sequence unchanged",
    )
    record(
        "calibration registry",
        config["calibration"]["candidates"] == ["temperature", "platt"],
        "Only temperature and Platt",
    )
    record(
        "probability clipping",
        float(config["probability"]["primary_clip"]) == 1e-6,
        "Primary epsilon 1e-6",
    )

    fold_path = project_path(
        config, "r7_artifacts/development/patient_fold_assignments.csv"
    )
    folds = pd.read_csv(fold_path, dtype={"patient_id": str})
    record(
        "patient-level fold manifest",
        len(folds) == 5771
        and not folds["patient_id"].duplicated().any()
        and set(folds["patient_id"]) == set(integrity.development_ids),
        "5,771 unique development patients",
    )

    artifact_dir = project_path(config, "r7_artifacts/development")
    sealed_paths: list[Path] = _source_files(config)
    sealed_paths.extend([development_completion, fold_path])
    for endpoint in config["endpoints"]:
        prediction_path = artifact_dir / f"{endpoint}_nested_oof_predictions.csv"
        selection_path = artifact_dir / f"{endpoint}_selection.json"
        final_oof_path = artifact_dir / f"{endpoint}_final_development_oof.csv"
        stack_path = artifact_dir / f"{endpoint}_final_stack.joblib"
        prediction = pd.read_csv(
            prediction_path, dtype={"patient_id": str, "image_id": str}
        )
        record(
            f"{endpoint} OOF coverage",
            len(prediction) == 11542
            and prediction["patient_id"].nunique() == 5771
            and not prediction["image_id"].duplicated().any(),
            "One nested-CV prediction per development eye",
        )
        record(
            f"{endpoint} holdout exclusion",
            not bool(set(prediction["patient_id"]) & set(integrity.holdout_ids)),
            "No holdout patient in development predictions",
        )
        expected_fold = prediction["patient_id"].map(
            folds.set_index("patient_id")["outer_fold"]
        )
        record(
            f"{endpoint} outer-fold provenance",
            expected_fold.notna().all()
            and np.array_equal(
                expected_fold.to_numpy(dtype=int),
                prediction["outer_fold"].to_numpy(dtype=int),
            ),
            "Prediction fold equals frozen patient assessment fold",
        )
        recomputed_nll = binary_nll(
            prediction["true_label"].to_numpy(dtype=int),
            prediction["calibrated_probability"].to_numpy(dtype=float),
            float(config["probability"]["primary_clip"]),
        )
        record(
            f"{endpoint} loss/clipping implementation",
            np.allclose(
                recomputed_nll,
                prediction["nll"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ),
            "Stored NLL exactly matches epsilon=1e-6 recomputation",
        )
        record(
            f"{endpoint} calibration provenance",
            set(prediction["selected_calibrator"]).issubset(
                set(config["calibration"]["candidates"])
            )
            and set(prediction["selected_C"]).issubset(
                set(map(float, config["disease_head"]["C_grid"]))
            ),
            "Only registered C/calibration choices present",
        )
        stack = joblib.load(stack_path)
        base_schema = stack["reliability_schemas"]["nll_r0"]["design_columns"]
        augmented_schema = stack["reliability_schemas"]["nll_r1"][
            "design_columns"
        ]
        added = [column for column in augmented_schema if column not in base_schema]
        record(
            f"{endpoint} R0/R1 schema",
            bool(added)
            and all(column.startswith("fellow_logit_spline_") for column in added)
            and not any("state" in column for column in augmented_schema),
            "R1 adds only fellow-logit spline; no ground-truth state",
        )
        record(
            f"{endpoint} stack config hash",
            stack["config_sha256"] == config["_config_sha256"],
            stack["config_sha256"],
        )
        sealed_paths.extend(
            [prediction_path, selection_path, final_oof_path, stack_path]
        )

    required_results = [
        result_dir / "R7_DEVELOPMENT_BASE_MODEL_REPORT.md",
        result_dir / "R7_DEVELOPMENT_RELIABILITY_REPORT.md",
        result_dir / "R7_DEVELOPMENT_MECHANISTIC_REPORT.md",
        result_dir / "development_base_model_metrics.json",
        result_dir / "development_reliability_results.json",
        result_dir / "development_mechanistic_results.json",
        result_dir / "development_reliability_gee.csv",
        result_dir / "development_mechanistic_ols_hc3.csv",
        result_dir / "development_quality_stratified_results.json",
    ]
    record(
        "development reports complete",
        all(path.is_file() for path in required_results),
        "Base, reliability, mechanistic, GEE, and quality results present",
    )
    sealed_paths.extend(required_results)
    development_embedding_files = sorted(
        project_path(config, "r7_artifacts/embeddings/development").glob("*")
    )
    sealed_paths.extend(development_embedding_files)

    relative_sealed = sorted(
        str(path.relative_to(project_path(config, "."))) for path in set(sealed_paths)
    )
    payload = {
        "status": "PASS",
        "frozen_config_sha256": config["_config_sha256"],
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "checks": checks,
        "sealed_relative_paths": relative_sealed,
        "scientific_specification_changed_after_development_results": False,
        "new_model_candidate_introduced": False,
        "new_uncertainty_estimator_introduced": False,
        "holdout_accessed": False,
    }
    machine_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# R7 Pre-Holdout Validation",
        "",
        "# PASS",
        "",
        "The frozen development implementation is complete and the holdout remains inaccessible to modeling code.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['check']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        )
    lines.extend(
        [
            "",
            "The R0/R1 schemas match the frozen registries; calibration, clipping, endpoint hierarchy, preprocessing, feature layer, patient folds, and model candidate set are unchanged. No ground-truth bilateral state enters a reliability predictor.",
            "",
            "Development findings did not cause a scientific specification change. Creating `HOLDOUT_RELEASED.json` now seals the listed implementation and development files.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def release_holdout(config: dict[str, Any]) -> Path:
    validation_path = project_path(config, "r7_results/pre_holdout_validation.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("Machine-readable pre-holdout validation is not PASS")
    sealed = list(validation["sealed_relative_paths"])
    sealed.extend(
        [
            "r7_results/pre_holdout_validation.json",
            config["firewall"]["pre_holdout_report"],
        ]
    )
    return create_holdout_release_marker(config, sealed)
