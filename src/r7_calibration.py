"""Frozen temperature and Platt calibration implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


def clipped_probability(probability: np.ndarray, epsilon: float) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)


def binary_nll(
    labels: np.ndarray, probability: np.ndarray, epsilon: float
) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    values = clipped_probability(probability, epsilon)
    return -(labels * np.log(values) + (1.0 - labels) * np.log1p(-values))


def probability_logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    values = clipped_probability(probability, epsilon)
    return np.log(values) - np.log1p(-values)


@dataclass(frozen=True)
class FittedCalibrator:
    method: str
    slope: float
    intercept: float
    optimizer_success: bool
    optimizer_message: str

    def predict(self, raw_logit: np.ndarray) -> np.ndarray:
        return expit(self.slope * np.asarray(raw_logit, dtype=float) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_calibrator(
    method: str,
    raw_logit: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
) -> FittedCalibrator:
    raw_logit = np.asarray(raw_logit, dtype=float)
    labels = np.asarray(labels, dtype=float)
    epsilon = float(config["probability"]["primary_clip"])
    calibration = config["calibration"]

    if method == "temperature":
        low, high = map(float, calibration["temperature_bounds"])

        def objective(log_temperature: float) -> float:
            slope = np.exp(-log_temperature)
            return float(binary_nll(labels, expit(slope * raw_logit), epsilon).mean())

        result = minimize_scalar(
            objective,
            bounds=(np.log(low), np.log(high)),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 1000},
        )
        return FittedCalibrator(
            method="temperature",
            slope=float(np.exp(-result.x)),
            intercept=0.0,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

    if method == "platt":
        slope_low, slope_high = map(float, calibration["platt_slope_bounds"])
        intercept_low, intercept_high = map(
            float, calibration["platt_intercept_bounds"]
        )

        def objective(parameters: np.ndarray) -> float:
            slope = np.exp(parameters[0])
            intercept = parameters[1]
            return float(
                binary_nll(
                    labels, expit(slope * raw_logit + intercept), epsilon
                ).mean()
            )

        result = minimize(
            objective,
            x0=np.array([0.0, 0.0]),
            method="L-BFGS-B",
            bounds=[
                (np.log(slope_low), np.log(slope_high)),
                (intercept_low, intercept_high),
            ],
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        return FittedCalibrator(
            method="platt",
            slope=float(np.exp(result.x[0])),
            intercept=float(result.x[1]),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

    raise ValueError(f"Unregistered calibration method: {method}")


def crossfit_calibrator(
    method: str,
    raw_logit: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    raw_logit = np.asarray(raw_logit, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fold_ids = np.asarray(fold_ids, dtype=int)
    output = np.full(len(labels), np.nan, dtype=float)
    provenance: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        assess = fold_ids == fold
        fitted = fit_calibrator(method, raw_logit[train], labels[train], config)
        output[assess] = fitted.predict(raw_logit[assess])
        provenance.append(
            {
                "assessment_fold": int(fold),
                "fit_rows": int(train.sum()),
                "assessment_rows": int(assess.sum()),
                "parameters": fitted.to_dict(),
            }
        )
    if np.isnan(output).any():
        raise RuntimeError("Cross-fitted calibration left missing predictions")
    return output, provenance
