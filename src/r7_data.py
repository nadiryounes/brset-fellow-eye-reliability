"""Authorized BRSET cohort and paired-eye data assembly."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .r7_firewall import stream_stage_metadata


MODEL_COLUMNS = (
    "image_id",
    "exam_eye",
    "camera",
    "image_field",
    "focus",
    "drusen",
    "increased_cup_disc",
    "diabetic_retinopathy",
)


def numeric_patient_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_stage_dataframe(
    config: dict[str, Any], stage: str, *, purpose: str
) -> pd.DataFrame:
    records = stream_stage_metadata(
        config, stage, columns=MODEL_COLUMNS, purpose=purpose
    )
    frame = pd.DataFrame.from_records(records)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["exam_eye"] = pd.to_numeric(frame["exam_eye"], errors="raise").astype(int)
    for endpoint in config["endpoints"]:
        frame[endpoint] = pd.to_numeric(frame[endpoint], errors="raise").astype(int)
        if not set(frame[endpoint].unique()).issubset({0, 1}):
            raise ValueError(f"Nonbinary endpoint values for {endpoint}")
    frame["image_field"] = pd.to_numeric(
        frame["image_field"], errors="raise"
    ).astype(int)
    frame["focus"] = pd.to_numeric(frame["focus"], errors="raise").astype(int)
    frame["laterality"] = frame["exam_eye"].map({1: "R", 2: "L"})
    if frame["laterality"].isna().any():
        raise ValueError("Unexpected exam_eye value")
    frame["partition"] = stage
    frame["image_path"] = frame["image_id"].map(
        lambda image_id: str(Path(config["source"]["image_dir"]) / f"{image_id}.jpg")
    )
    _assert_unambiguous_pairs(frame)
    order = pd.DataFrame(
        [numeric_patient_key(value) for value in frame["patient_id"]],
        columns=["_patient_kind", "_patient_value"],
        index=frame.index,
    )
    frame = pd.concat([frame, order], axis=1).sort_values(
        ["_patient_kind", "_patient_value", "exam_eye"], kind="stable"
    )
    return frame.drop(columns=["_patient_kind", "_patient_value"]).reset_index(
        drop=True
    )


def _assert_unambiguous_pairs(frame: pd.DataFrame) -> None:
    counts = frame.groupby("patient_id", sort=False).size()
    if not (counts == 2).all():
        raise ValueError("Every authorized patient must have exactly two rows")
    eye_sets = frame.groupby("patient_id", sort=False)["exam_eye"].agg(
        lambda values: frozenset(values)
    )
    if not all(value == frozenset({1, 2}) for value in eye_sets):
        raise ValueError("Every authorized patient must have one right and one left eye")
    if frame["image_id"].duplicated().any():
        raise ValueError("Image IDs must be unique in the authorized stage")


def attach_embeddings(
    metadata: pd.DataFrame, embedding_index: pd.DataFrame, features: Any
) -> pd.DataFrame:
    required = {"image_id", "patient_id", "exam_eye", "feature_row"}
    if not required.issubset(embedding_index.columns):
        raise ValueError("Embedding index schema mismatch")
    merged = metadata.merge(
        embedding_index[list(required)],
        on=["image_id", "patient_id", "exam_eye"],
        how="left",
        validate="one_to_one",
    )
    if merged["feature_row"].isna().any():
        raise ValueError("Missing embedding row for authorized image")
    rows = merged["feature_row"].astype(int).to_numpy()
    if rows.min() < 0 or rows.max() >= len(features):
        raise ValueError("Embedding row reference out of bounds")
    merged["feature_row"] = rows
    return merged


def add_pair_fields(frame: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    """Add fellow-eye values and right/left four-state truth for analysis."""

    _assert_unambiguous_pairs(frame)
    result = frame.copy()
    pair_columns = [
        "image_id",
        "calibrated_probability",
        "calibrated_logit",
        "image_field",
        "focus",
        "camera",
    ]
    fellow = result[["patient_id", "exam_eye", *pair_columns]].copy()
    fellow["exam_eye"] = fellow["exam_eye"].map({1: 2, 2: 1})
    fellow = fellow.rename(
        columns={name: f"fellow_{name}" for name in pair_columns}
    )
    result = result.merge(
        fellow, on=["patient_id", "exam_eye"], how="left", validate="one_to_one"
    )
    if result["fellow_image_id"].isna().any():
        raise ValueError("Fellow-eye mapping failed")

    right = result[result["exam_eye"] == 1].set_index("patient_id")
    left = result[result["exam_eye"] == 2].set_index("patient_id")
    for endpoint in endpoints:
        state = right[endpoint].astype(str) + left[endpoint].astype(str)
        result[f"{endpoint}_state"] = result["patient_id"].map(state)
    return result


def state_counts(frame: pd.DataFrame, endpoint: str) -> dict[str, int]:
    right = frame[frame["exam_eye"] == 1].set_index("patient_id")[endpoint]
    left = frame[frame["exam_eye"] == 2].set_index("patient_id")[endpoint]
    states = right.astype(str) + left.astype(str)
    counts = Counter(states)
    return {state: int(counts.get(state, 0)) for state in ("00", "11", "10", "01")}
