"""Frozen own-eye and fellow-eye loss-prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler


CATEGORICAL_COLUMNS = (
    "own_image_field_abnormal",
    "own_focus_abnormal",
    "own_focus_unevaluable",
    "laterality_right",
    "own_camera_nikon",
)


@dataclass
class ReliabilityRegressor:
    config: dict[str, Any]
    alpha: float
    augmented: bool
    target_kind: str = "nll"

    def __post_init__(self) -> None:
        spline = self.config["reliability"]["own_logit_spline"]
        kwargs = {
            "n_knots": int(spline["n_knots"]),
            "degree": int(spline["degree"]),
            "knots": spline["knots"],
            "extrapolation": spline["extrapolation"],
            "include_bias": bool(spline["include_bias"]),
        }
        self.own_spline = SplineTransformer(**kwargs)
        self.own_scaler = StandardScaler()
        self.fellow_spline = SplineTransformer(**kwargs) if self.augmented else None
        self.fellow_scaler = StandardScaler() if self.augmented else None
        self.regressor = Ridge(alpha=float(self.alpha), fit_intercept=True)
        self.design_columns_: list[str] = []

    @staticmethod
    def _categorical(frame: pd.DataFrame) -> np.ndarray:
        values = np.column_stack(
            [
                (frame["image_field"].to_numpy() == 2).astype(float),
                (frame["focus"].to_numpy() == 2).astype(float),
                (~frame["focus"].isin([1, 2]).to_numpy()).astype(float),
                (frame["exam_eye"].to_numpy() == 1).astype(float),
                (frame["camera"].to_numpy() == "NIKON NF5050").astype(float),
            ]
        )
        return values

    def _design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        own = frame[["calibrated_logit"]].to_numpy(dtype=float)
        if fit:
            own_basis = self.own_spline.fit_transform(own)
            own_basis = self.own_scaler.fit_transform(own_basis)
        else:
            own_basis = self.own_scaler.transform(self.own_spline.transform(own))
        blocks = [own_basis, self._categorical(frame)]
        columns = [f"own_logit_spline_{index}" for index in range(own_basis.shape[1])]
        columns.extend(CATEGORICAL_COLUMNS)

        if self.augmented:
            if "fellow_calibrated_logit" not in frame:
                raise ValueError("Augmented model requires fellow calibrated logit")
            fellow = frame[["fellow_calibrated_logit"]].to_numpy(dtype=float)
            assert self.fellow_spline is not None and self.fellow_scaler is not None
            if fit:
                fellow_basis = self.fellow_spline.fit_transform(fellow)
                fellow_basis = self.fellow_scaler.fit_transform(fellow_basis)
            else:
                fellow_basis = self.fellow_scaler.transform(
                    self.fellow_spline.transform(fellow)
                )
            blocks.append(fellow_basis)
            columns.extend(
                f"fellow_logit_spline_{index}"
                for index in range(fellow_basis.shape[1])
            )
        if fit:
            self.design_columns_ = columns
        elif columns != self.design_columns_:
            raise RuntimeError("Reliability design schema changed between fit/predict")
        return np.column_stack(blocks)

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "ReliabilityRegressor":
        design = self._design(frame, fit=True)
        self.regressor.fit(design, np.asarray(target, dtype=float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = self._design(frame, fit=False)
        prediction = self.regressor.predict(design)
        if self.target_kind == "nll":
            low, high = map(float, self.config["reliability"]["prediction_bounds"])
        elif self.target_kind == "brier":
            low, high = 0.0, 1.0
        else:
            raise ValueError(f"Unknown reliability target: {self.target_kind}")
        return np.clip(prediction, low, high)

    def schema(self) -> dict[str, Any]:
        return {
            "augmented": self.augmented,
            "alpha": float(self.alpha),
            "target_kind": self.target_kind,
            "design_columns": self.design_columns_,
            "target_bounds": self.config["reliability"]["prediction_bounds"],
        }


def patient_mean_mse(
    frame: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray
) -> float:
    values = frame[["patient_id"]].copy()
    values["squared_error"] = (
        np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    ) ** 2
    return float(values.groupby("patient_id")["squared_error"].mean().mean())


def select_ridge_alpha(
    frame: pd.DataFrame,
    target: np.ndarray,
    fold_ids: np.ndarray,
    *,
    augmented: bool,
    target_kind: str,
    config: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    target = np.asarray(target, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    results: list[dict[str, float]] = []
    for alpha in map(float, config["reliability"]["ridge_alpha_grid"]):
        predicted = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(np.unique(fold_ids)):
            train = fold_ids != fold
            assess = fold_ids == fold
            model = ReliabilityRegressor(
                config, alpha, augmented, target_kind=target_kind
            ).fit(
                frame.loc[train].reset_index(drop=True), target[train]
            )
            predicted[assess] = model.predict(
                frame.loc[assess].reset_index(drop=True)
            )
        risk = patient_mean_mse(frame, target, predicted)
        results.append({"alpha": alpha, "patient_mean_mse": risk})

    tolerance = float(config["reliability"]["ridge_tie_tolerance"])
    best_risk = min(item["patient_mean_mse"] for item in results)
    eligible = [
        item for item in results if item["patient_mean_mse"] <= best_risk + tolerance
    ]
    selected = max(item["alpha"] for item in eligible)
    return selected, results
