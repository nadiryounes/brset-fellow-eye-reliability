"""Frozen nested-CV development execution and final development refit."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .r7_calibration import (
    binary_nll,
    crossfit_calibrator,
    fit_calibrator,
    probability_logit,
)
from .r7_config import project_path, sha256_file
from .r7_cv import (
    fit_disease_head,
    make_inner_patient_folds,
    make_outer_patient_folds,
    modal_choice,
    row_fold_ids,
)
from .r7_data import attach_embeddings, load_stage_dataframe
from .r7_embeddings import load_embeddings
from .r7_firewall import DEVELOPMENT
from .r7_reliability import (
    ReliabilityRegressor,
    patient_mean_mse,
    select_ridge_alpha,
)


def patient_mean(frame: pd.DataFrame, values: np.ndarray) -> float:
    temporary = frame[["patient_id"]].copy()
    temporary["value"] = np.asarray(values, dtype=float)
    return float(temporary.groupby("patient_id")["value"].mean().mean())


def prepare_reliability_frame(
    metadata: pd.DataFrame,
    calibrated_probability: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    frame["_reliability_row_order"] = np.arange(len(frame))
    frame["calibrated_probability"] = np.asarray(
        calibrated_probability, dtype=float
    )
    frame["calibrated_logit"] = probability_logit(
        frame["calibrated_probability"].to_numpy(),
        float(config["probability"]["primary_clip"]),
    )
    fellow = frame[["patient_id", "exam_eye", "calibrated_logit"]].copy()
    fellow["exam_eye"] = fellow["exam_eye"].map({1: 2, 2: 1})
    fellow = fellow.rename(
        columns={"calibrated_logit": "fellow_calibrated_logit"}
    )
    frame = frame.merge(
        fellow,
        on=["patient_id", "exam_eye"],
        how="left",
        validate="one_to_one",
    )
    if frame["fellow_calibrated_logit"].isna().any():
        raise RuntimeError("Fellow-eye reliability mapping failed")
    return frame.sort_values("_reliability_row_order", kind="stable").drop(
        columns="_reliability_row_order"
    ).reset_index(drop=True)


def _select_calibrated_head(
    metadata: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    inner_fold_ids: np.ndarray,
    endpoint: str,
    outer_fold: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    candidates: list[dict[str, Any]] = []
    raw_by_c: dict[float, np.ndarray] = {}
    seed_base = int(config["cross_validation"]["training_seed_base"])
    epsilon = float(config["probability"]["primary_clip"])

    for c_index, C in enumerate(map(float, config["disease_head"]["C_grid"])):
        raw = np.full(len(metadata), np.nan, dtype=float)
        head_provenance: list[dict[str, Any]] = []
        for inner_fold in sorted(np.unique(inner_fold_ids)):
            train = inner_fold_ids != inner_fold
            assess = inner_fold_ids == inner_fold
            head = fit_disease_head(
                features[train],
                labels[train],
                C,
                config,
                seed=seed_base + outer_fold * 100 + c_index * 10 + int(inner_fold),
            )
            raw[assess] = head.raw_logit(features[assess])
            head_provenance.append(
                {
                    "inner_fold": int(inner_fold),
                    "fit_patients": int(metadata.loc[train, "patient_id"].nunique()),
                    "assessment_patients": int(
                        metadata.loc[assess, "patient_id"].nunique()
                    ),
                    "head": head.provenance(),
                }
            )
        if np.isnan(raw).any():
            raise RuntimeError("Inner disease OOF predictions incomplete")
        raw_by_c[C] = raw
        for method in config["calibration"]["candidates"]:
            calibrated, calibrator_provenance = crossfit_calibrator(
                method, raw, labels, inner_fold_ids, config
            )
            risk = patient_mean(metadata, binary_nll(labels, calibrated, epsilon))
            candidates.append(
                {
                    "endpoint": endpoint,
                    "outer_fold": int(outer_fold),
                    "C": C,
                    "calibration_method": method,
                    "patient_mean_calibrated_nll": risk,
                    "inner_head_provenance": head_provenance,
                    "calibrator_provenance": calibrator_provenance,
                    "crossfitted_calibrated_probability": calibrated,
                }
            )

    tolerance = float(config["calibration"]["tie_tolerance"])
    best_risk = min(item["patient_mean_calibrated_nll"] for item in candidates)
    eligible = [
        item
        for item in candidates
        if item["patient_mean_calibrated_nll"] <= best_risk + tolerance
    ]
    method_order = {"temperature": 0, "platt": 1}
    selected = sorted(
        eligible,
        key=lambda item: (float(item["C"]), method_order[item["calibration_method"]]),
    )[0]
    selected["candidate_summary"] = [
        {
            "C": item["C"],
            "calibration_method": item["calibration_method"],
            "patient_mean_calibrated_nll": item[
                "patient_mean_calibrated_nll"
            ],
        }
        for item in candidates
    ]
    return selected, raw_by_c


def _fit_outer_reliability(
    training_frame: pd.DataFrame,
    assessment_frame: pd.DataFrame,
    labels_train: np.ndarray,
    labels_assess: np.ndarray,
    inner_fold_ids: np.ndarray,
    calibrated_train: np.ndarray,
    calibrated_assess: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    epsilon = float(config["probability"]["primary_clip"])
    train = prepare_reliability_frame(training_frame, calibrated_train, config)
    assess = prepare_reliability_frame(assessment_frame, calibrated_assess, config)
    targets_train = {
        "nll": binary_nll(labels_train, calibrated_train, epsilon),
        "brier": (calibrated_train - labels_train) ** 2,
    }
    targets_assess = {
        "nll": binary_nll(labels_assess, calibrated_assess, epsilon),
        "brier": (calibrated_assess - labels_assess) ** 2,
    }
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for target_kind in ("nll", "brier"):
        for augmented, model_name in ((False, "r0"), (True, "r1")):
            alpha, selection = select_ridge_alpha(
                train,
                targets_train[target_kind],
                inner_fold_ids,
                augmented=augmented,
                target_kind=target_kind,
                config=config,
            )
            model = ReliabilityRegressor(
                config,
                alpha,
                augmented,
                target_kind=target_kind,
            ).fit(train, targets_train[target_kind])
            key = f"predicted_{target_kind}_{model_name}"
            predictions[key] = model.predict(assess)
            provenance[f"{target_kind}_{model_name}"] = {
                "selected_alpha": alpha,
                "selection": selection,
                "schema": model.schema(),
                "assessment_patient_mean_mse": patient_mean_mse(
                    assess, targets_assess[target_kind], predictions[key]
                ),
            }
    return predictions, provenance


def _json_safe_outer_selection(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in selected.items()
        if key != "crossfitted_calibrated_probability"
    }


def run_nested_endpoint(
    endpoint: str,
    metadata: pd.DataFrame,
    features: np.ndarray,
    outer_patients: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    labels_all = metadata[endpoint].to_numpy(dtype=int)
    outer_row_folds = row_fold_ids(metadata, outer_patients, "outer_fold")
    outputs: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    seed_base = int(config["cross_validation"]["training_seed_base"])
    epsilon = float(config["probability"]["primary_clip"])

    for outer_fold in sorted(np.unique(outer_row_folds)):
        train_mask = outer_row_folds != outer_fold
        assess_mask = outer_row_folds == outer_fold
        train_indices = np.flatnonzero(train_mask)
        assess_indices = np.flatnonzero(assess_mask)
        train_frame = metadata.iloc[train_indices].reset_index(drop=True)
        assess_frame = metadata.iloc[assess_indices].reset_index(drop=True)
        train_features = np.asarray(features[train_indices])
        assess_features = np.asarray(features[assess_indices])
        train_labels = labels_all[train_indices]
        assess_labels = labels_all[assess_indices]

        inner_patients = make_inner_patient_folds(train_frame, int(outer_fold), config)
        inner_fold_ids = row_fold_ids(train_frame, inner_patients, "inner_fold")
        selected, raw_by_c = _select_calibrated_head(
            train_frame,
            train_features,
            train_labels,
            inner_fold_ids,
            endpoint,
            int(outer_fold),
            config,
        )
        selected_C = float(selected["C"])
        selected_method = str(selected["calibration_method"])
        selected_inner_raw = raw_by_c[selected_C]
        calibrated_inner = np.asarray(
            selected["crossfitted_calibrated_probability"], dtype=float
        )

        outer_head = fit_disease_head(
            train_features,
            train_labels,
            selected_C,
            config,
            seed=seed_base + 10000 + int(outer_fold),
        )
        outer_raw = outer_head.raw_logit(assess_features)
        outer_calibrator = fit_calibrator(
            selected_method, selected_inner_raw, train_labels, config
        )
        outer_calibrated = outer_calibrator.predict(outer_raw)

        reliability_predictions, reliability_provenance = _fit_outer_reliability(
            train_frame,
            assess_frame,
            train_labels,
            assess_labels,
            inner_fold_ids,
            calibrated_inner,
            outer_calibrated,
            config,
        )
        result = assess_frame[
            [
                "patient_id",
                "image_id",
                "exam_eye",
                "laterality",
                "camera",
                "image_field",
                "focus",
            ]
        ].copy()
        result["endpoint"] = endpoint
        result["outer_fold"] = int(outer_fold)
        result["raw_logistic_score"] = outer_raw
        result["uncalibrated_probability"] = expit(outer_raw)
        result["calibrated_probability"] = outer_calibrated
        result["calibrated_logit"] = probability_logit(
            outer_calibrated, epsilon
        )
        result["true_label"] = assess_labels
        result["nll"] = binary_nll(assess_labels, outer_calibrated, epsilon)
        result["brier"] = (outer_calibrated - assess_labels) ** 2
        result["probability_confidence"] = np.maximum(
            outer_calibrated, 1.0 - outer_calibrated
        )
        result["classification_prediction"] = (
            outer_calibrated >= float(config["thresholds"]["classification"])
        ).astype(int)
        result["clipped_primary"] = (
            (outer_calibrated < epsilon) | (outer_calibrated > 1.0 - epsilon)
        )
        result["selected_C"] = selected_C
        result["selected_calibrator"] = selected_method
        for key, values in reliability_predictions.items():
            result[key] = values
        outputs.append(result)

        selection = _json_safe_outer_selection(selected)
        selection.update(
            {
                "outer_head": outer_head.provenance(),
                "outer_calibrator": outer_calibrator.to_dict(),
                "reliability": reliability_provenance,
                "outer_training_patients": int(train_frame["patient_id"].nunique()),
                "outer_assessment_patients": int(
                    assess_frame["patient_id"].nunique()
                ),
            }
        )
        selections.append(selection)

    combined = pd.concat(outputs, ignore_index=True)
    if len(combined) != len(metadata) or combined["image_id"].duplicated().any():
        raise RuntimeError(f"Nested OOF coverage invalid for {endpoint}")
    combined["_numeric_patient"] = pd.to_numeric(
        combined["patient_id"], errors="coerce"
    )
    combined = combined.sort_values(
        ["_numeric_patient", "patient_id", "exam_eye"], kind="stable"
    ).drop(columns="_numeric_patient")
    return combined.reset_index(drop=True), selections


def _final_choices(selections: list[dict[str, Any]]) -> dict[str, Any]:
    choices: dict[str, Any] = {
        "C": modal_choice([item["C"] for item in selections], tie="smaller_numeric"),
        "calibration_method": modal_choice(
            [item["calibration_method"] for item in selections], tie="temperature"
        ),
    }
    for target_kind in ("nll", "brier"):
        for model_name in ("r0", "r1"):
            values = [
                item["reliability"][f"{target_kind}_{model_name}"][
                    "selected_alpha"
                ]
                for item in selections
            ]
            choices[f"{target_kind}_{model_name}_alpha"] = modal_choice(
                values, tie="larger_numeric"
            )
    return choices


def fit_final_development_stack(
    endpoint: str,
    metadata: pd.DataFrame,
    features: np.ndarray,
    outer_patients: pd.DataFrame,
    choices: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = metadata[endpoint].to_numpy(dtype=int)
    folds = row_fold_ids(metadata, outer_patients, "outer_fold")
    raw_oof = np.full(len(metadata), np.nan, dtype=float)
    oof_head_provenance: list[dict[str, Any]] = []
    seed_base = int(config["cross_validation"]["training_seed_base"])
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        assess = folds == fold
        head = fit_disease_head(
            np.asarray(features[train]),
            labels[train],
            float(choices["C"]),
            config,
            seed=seed_base + 20000 + int(fold),
        )
        raw_oof[assess] = head.raw_logit(np.asarray(features[assess]))
        oof_head_provenance.append(
            {"fold": int(fold), "head": head.provenance()}
        )
    if np.isnan(raw_oof).any():
        raise RuntimeError("Final development OOF raw logits incomplete")

    calibrator = fit_calibrator(
        choices["calibration_method"], raw_oof, labels, config
    )
    calibrated_oof = calibrator.predict(raw_oof)
    reliability_frame = prepare_reliability_frame(
        metadata, calibrated_oof, config
    )
    epsilon = float(config["probability"]["primary_clip"])
    targets = {
        "nll": binary_nll(labels, calibrated_oof, epsilon),
        "brier": (calibrated_oof - labels) ** 2,
    }
    reliability_models: dict[str, ReliabilityRegressor] = {}
    schemas: dict[str, Any] = {}
    for target_kind in ("nll", "brier"):
        for augmented, model_name in ((False, "r0"), (True, "r1")):
            key = f"{target_kind}_{model_name}"
            model = ReliabilityRegressor(
                config,
                float(choices[f"{key}_alpha"]),
                augmented,
                target_kind=target_kind,
            ).fit(reliability_frame, targets[target_kind])
            reliability_models[key] = model
            schemas[key] = model.schema()

    full_head = fit_disease_head(
        np.asarray(features),
        labels,
        float(choices["C"]),
        config,
        seed=seed_base + 30000,
    )
    stack = {
        "endpoint": endpoint,
        "config_sha256": config["_config_sha256"],
        "choices": choices,
        "full_head": full_head,
        "calibrator": calibrator,
        "reliability_models": reliability_models,
        "reliability_schemas": schemas,
        "development_oof_head_provenance": oof_head_provenance,
        "feature_dimension": int(features.shape[1]),
        "development_patients": int(metadata["patient_id"].nunique()),
        "development_images": int(len(metadata)),
    }
    fitting_frame = metadata[
        [
            "patient_id",
            "image_id",
            "exam_eye",
            "camera",
            "image_field",
            "focus",
        ]
    ].copy()
    fitting_frame["endpoint"] = endpoint
    fitting_frame["outer_fold"] = folds
    fitting_frame["raw_logistic_score"] = raw_oof
    fitting_frame["calibrated_probability"] = calibrated_oof
    fitting_frame["calibrated_logit"] = reliability_frame["calibrated_logit"]
    fitting_frame["true_label"] = labels
    fitting_frame["nll"] = targets["nll"]
    fitting_frame["brier"] = targets["brier"]
    return stack, fitting_frame


def create_development_fold_manifest(
    metadata: pd.DataFrame,
    outer_patients: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    if path.exists():
        raise RuntimeError("Development fold manifest already exists")
    manifest = outer_patients.copy()
    for outer_fold in range(int(config["cross_validation"]["outer_folds"])):
        assessment = set(
            manifest.loc[manifest["outer_fold"] == outer_fold, "patient_id"]
        )
        training = metadata[~metadata["patient_id"].isin(assessment)]
        inner = make_inner_patient_folds(training, outer_fold, config)
        mapping = inner.set_index("patient_id")["inner_fold"]
        manifest[f"inner_fold_when_outer_{outer_fold}"] = manifest[
            "patient_id"
        ].map(mapping)
    manifest.to_csv(path, index=False, lineterminator="\n")


def run_development(config: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = project_path(config, "r7_artifacts/development")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    completion_path = project_path(
        config, config["firewall"]["development_completion_marker"]
    )
    if completion_path.exists():
        raise RuntimeError("Development stage is already complete")

    features, embedding_index = load_embeddings(config, DEVELOPMENT)
    metadata = load_stage_dataframe(
        config, DEVELOPMENT, purpose="development_modeling"
    )
    metadata = attach_embeddings(metadata, embedding_index, features)
    ordered_features = np.asarray(features[metadata["feature_row"].to_numpy(dtype=int)])
    if ordered_features.shape != (len(metadata), config["model"]["feature_dimension"]):
        raise RuntimeError("Development feature alignment failed")

    outer_patients = make_outer_patient_folds(metadata, config)
    fold_manifest = artifact_dir / "patient_fold_assignments.csv"
    create_development_fold_manifest(metadata, outer_patients, config, fold_manifest)

    started = time.monotonic()
    endpoint_outputs: dict[str, Any] = {}
    for endpoint in config["endpoints"]:
        nested_path = artifact_dir / f"{endpoint}_nested_oof_predictions.csv"
        selection_path = artifact_dir / f"{endpoint}_selection.json"
        fitting_path = artifact_dir / f"{endpoint}_final_development_oof.csv"
        stack_path = artifact_dir / f"{endpoint}_final_stack.joblib"
        if any(
            path.exists()
            for path in (nested_path, selection_path, fitting_path, stack_path)
        ):
            raise RuntimeError(f"Refusing to overwrite development {endpoint} artifacts")

        nested, selections = run_nested_endpoint(
            endpoint, metadata, ordered_features, outer_patients, config
        )
        choices = _final_choices(selections)
        stack, fitting_frame = fit_final_development_stack(
            endpoint,
            metadata,
            ordered_features,
            outer_patients,
            choices,
            config,
        )
        nested.to_csv(nested_path, index=False, lineterminator="\n")
        fitting_frame.to_csv(fitting_path, index=False, lineterminator="\n")
        joblib.dump(stack, stack_path, compress=3)
        selection_payload = {
            "endpoint": endpoint,
            "outer_selections": selections,
            "final_choices": choices,
            "nested_oof_predictions_sha256": sha256_file(nested_path),
            "final_development_oof_sha256": sha256_file(fitting_path),
            "final_stack_sha256": sha256_file(stack_path),
        }
        selection_path.write_text(
            json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        endpoint_outputs[endpoint] = {
            "nested_oof_predictions": str(nested_path.relative_to(project_path(config, "."))),
            "selection": str(selection_path.relative_to(project_path(config, "."))),
            "final_development_oof": str(fitting_path.relative_to(project_path(config, "."))),
            "final_stack": str(stack_path.relative_to(project_path(config, "."))),
            "final_choices": choices,
        }

    completion = {
        "status": "DEVELOPMENT_COMPLETE",
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "frozen_config_sha256": config["_config_sha256"],
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "development_embedding_manifest_sha256": sha256_file(
            project_path(
                config,
                "r7_artifacts/embeddings/development/extraction_manifest.json",
            )
        ),
        "fold_manifest_sha256": sha256_file(fold_manifest),
        "endpoint_outputs": endpoint_outputs,
        "holdout_accessed": False,
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion
