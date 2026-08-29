"""Write-once frozen final-holdout prediction execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .r7_calibration import binary_nll, probability_logit
from .r7_config import project_path, sha256_file
from .r7_data import attach_embeddings, load_stage_dataframe
from .r7_development import prepare_reliability_frame
from .r7_embeddings import load_embeddings
from .r7_firewall import FINAL_HOLDOUT, authorize_stage


def execute_holdout_predictions(config: dict[str, Any]) -> dict[str, Any]:
    authorize_stage(config, FINAL_HOLDOUT, purpose="prediction_generation")
    execution_marker = project_path(
        config, config["firewall"]["holdout_execution_marker"]
    )
    if execution_marker.exists():
        raise RuntimeError("Holdout predictions have already been generated")
    artifact_dir = project_path(config, "r7_artifacts/final_holdout")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        endpoint: artifact_dir / f"{endpoint}_holdout_predictions.csv"
        for endpoint in config["endpoints"]
    }
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError("Partial holdout prediction artifacts exist; review required")

    features, embedding_index = load_embeddings(config, FINAL_HOLDOUT)
    metadata = load_stage_dataframe(
        config, FINAL_HOLDOUT, purpose="prediction_generation"
    )
    metadata = attach_embeddings(metadata, embedding_index, features)
    ordered_features = np.asarray(features[metadata["feature_row"].to_numpy(dtype=int)])
    if ordered_features.shape != (3848, config["model"]["feature_dimension"]):
        raise RuntimeError("Holdout feature alignment failed")

    epsilon = float(config["probability"]["primary_clip"])
    outputs: dict[str, Any] = {}
    for endpoint in config["endpoints"]:
        stack_path = project_path(
            config, f"r7_artifacts/development/{endpoint}_final_stack.joblib"
        )
        stack = joblib.load(stack_path)
        if stack["config_sha256"] != config["_config_sha256"]:
            raise RuntimeError("Sealed stack/config hash mismatch")
        raw = stack["full_head"].raw_logit(ordered_features)
        probability = stack["calibrator"].predict(raw)
        reliability_frame = prepare_reliability_frame(metadata, probability, config)
        labels = metadata[endpoint].to_numpy(dtype=int)
        result = metadata[
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
        result["outer_fold"] = -1
        result["raw_logistic_score"] = raw
        result["uncalibrated_probability"] = expit(raw)
        result["calibrated_probability"] = probability
        result["calibrated_logit"] = reliability_frame["calibrated_logit"]
        result["true_label"] = labels
        result["nll"] = binary_nll(labels, probability, epsilon)
        result["brier"] = (probability - labels) ** 2
        result["probability_confidence"] = np.maximum(probability, 1.0 - probability)
        result["classification_prediction"] = (
            probability >= float(config["thresholds"]["classification"])
        ).astype(int)
        result["clipped_primary"] = (
            (probability < epsilon) | (probability > 1.0 - epsilon)
        )
        result["selected_C"] = float(stack["choices"]["C"])
        result["selected_calibrator"] = stack["choices"]["calibration_method"]
        for key, model in stack["reliability_models"].items():
            result[f"predicted_{key}"] = model.predict(reliability_frame)
        if result["patient_id"].nunique() != 1924 or len(result) != 3848:
            raise RuntimeError("Holdout prediction coverage changed")
        result.to_csv(output_paths[endpoint], index=False, lineterminator="\n")
        outputs[endpoint] = {
            "path": str(output_paths[endpoint].relative_to(project_path(config, "."))),
            "sha256": sha256_file(output_paths[endpoint]),
            "patients": 1924,
            "eyes": 3848,
            "selected_C": float(stack["choices"]["C"]),
            "selected_calibrator": stack["choices"]["calibration_method"],
        }

    marker = {
        "status": "HOLDOUT_EXECUTED",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_count": 1,
        "frozen_split_sha256": config["partition"]["manifest_sha256"],
        "frozen_config_sha256": config["_config_sha256"],
        "holdout_embedding_manifest_sha256": sha256_file(
            project_path(
                config,
                "r7_artifacts/embeddings/final_holdout/extraction_manifest.json",
            )
        ),
        "outputs": outputs,
        "holdout_used_for_model_selection": False,
        "scientific_protocol_modified_after_predictions": False,
        "original_outputs_preserved": True,
    }
    execution_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker
