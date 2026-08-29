"""Frozen disease and reliability evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score

from .r7_calibration import binary_nll, probability_logit


def calibration_intercept_slope(
    labels: np.ndarray, probability: np.ndarray, epsilon: float
) -> dict[str, Any]:
    logit = probability_logit(probability, epsilon)
    design = sm.add_constant(logit, has_constant="add")
    try:
        fitted = sm.GLM(labels, design, family=sm.families.Binomial()).fit()
        return {
            "intercept": float(fitted.params[0]),
            "slope": float(fitted.params[1]),
            "converged": bool(fitted.converged),
        }
    except Exception as error:  # numerical diagnostic, never changes model
        return {
            "intercept": None,
            "slope": None,
            "converged": False,
            "error": f"{type(error).__name__}: {error}",
        }


def equal_frequency_ece(
    labels: np.ndarray, probability: np.ndarray, bins: int
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.quantile(probability, np.linspace(0.0, 1.0, bins + 1))
    unique_edges = np.unique(edges)
    if len(unique_edges) != bins + 1:
        return {
            "ece": None,
            "estimable": False,
            "reason": "duplicate quantile edges",
            "requested_bins": int(bins),
        }
    membership = np.digitize(probability, edges[1:-1], right=True)
    ece = 0.0
    rows = []
    for bin_index in range(bins):
        mask = membership == bin_index
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean())
        observed_rate = float(labels[mask].mean())
        ece += count / len(labels) * abs(mean_probability - observed_rate)
        rows.append(
            {
                "bin": bin_index,
                "count": count,
                "lower_edge": float(edges[bin_index]),
                "upper_edge": float(edges[bin_index + 1]),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {"ece": float(ece), "estimable": True, "bins": rows}


def fixed_edge_ece(
    labels: np.ndarray,
    probability: np.ndarray,
    edges: np.ndarray,
    *,
    minimum_bin_count: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.asarray(edges, dtype=float)
    bins = len(edges) - 1
    membership = np.digitize(probability, edges[1:-1], right=True)
    rows = []
    ece = 0.0
    for bin_index in range(bins):
        mask = membership == bin_index
        count = int(mask.sum())
        if count == 0:
            mean_probability = None
            observed_rate = None
        else:
            mean_probability = float(probability[mask].mean())
            observed_rate = float(labels[mask].mean())
            ece += count / len(labels) * abs(mean_probability - observed_rate)
        rows.append(
            {
                "bin": bin_index,
                "count": count,
                "lower_edge": float(edges[bin_index]),
                "upper_edge": float(edges[bin_index + 1]),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    if any(row["count"] < minimum_bin_count for row in rows):
        return {
            "ece": None,
            "estimable": False,
            "reason": f"at least one fixed holdout bin has fewer than {minimum_bin_count} eyes",
            "bins": rows,
        }
    return {"ece": float(ece), "estimable": True, "bins": rows}


def base_model_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    labels = frame["true_label"].to_numpy(dtype=int)
    probability = frame["calibrated_probability"].to_numpy(dtype=float)
    threshold = float(config["thresholds"]["classification"])
    prediction = probability >= threshold
    positive = labels == 1
    negative = ~positive
    sensitivity = float((prediction[positive] == 1).mean()) if positive.any() else None
    specificity = float((prediction[negative] == 0).mean()) if negative.any() else None
    epsilon = float(config["probability"]["primary_clip"])
    nll_sensitivities = {
        f"epsilon_{epsilon_value:g}": float(
            binary_nll(labels, probability, float(epsilon_value)).mean()
        )
        for epsilon_value in config["probability"]["sensitivity_clips"]
    }
    return {
        "eyes": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "positive_eyes": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "nll": float(binary_nll(labels, probability, epsilon).mean()),
        "nll_clip_sensitivities": nll_sensitivities,
        "brier": float(np.mean((probability - labels) ** 2)),
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
        "calibration": calibration_intercept_slope(labels, probability, epsilon),
        "ece": equal_frequency_ece(
            labels, probability, int(config["thresholds"]["ece_bins"])
        ),
        "primary_clip_count": int(frame["clipped_primary"].sum()),
    }


def reliability_risks(
    frame: pd.DataFrame, target_kind: str
) -> dict[str, Any]:
    observed_column = "nll" if target_kind == "nll" else "brier"
    observed = frame[observed_column].to_numpy(dtype=float)
    r0 = frame[f"predicted_{target_kind}_r0"].to_numpy(dtype=float)
    r1 = frame[f"predicted_{target_kind}_r1"].to_numpy(dtype=float)
    contributions = frame[["patient_id"]].copy()
    contributions["risk_r0"] = (observed - r0) ** 2
    contributions["risk_r1"] = (observed - r1) ** 2
    patients = contributions.groupby("patient_id")[["risk_r0", "risk_r1"]].mean()
    risk_r0 = float(patients["risk_r0"].mean())
    risk_r1 = float(patients["risk_r1"].mean())
    delta = risk_r0 - risk_r1
    return {
        "target": target_kind,
        "patients": int(len(patients)),
        "risk_r0": risk_r0,
        "risk_r1": risk_r1,
        "delta_r": delta,
        "relative_mse_improvement": delta / risk_r0 if risk_r0 > 0 else None,
        "patient_contributions": (
            patients["risk_r0"] - patients["risk_r1"]
        ).to_numpy(dtype=float),
        "patient_ids": patients.index.astype(str).to_numpy(),
    }


def error_detection_metrics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    error = (
        frame["classification_prediction"].to_numpy(dtype=int)
        != frame["true_label"].to_numpy(dtype=int)
    ).astype(int)
    score = frame["predicted_nll_r1"].to_numpy(dtype=float)
    if error.min() == error.max():
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(error, score))
        auprc = float(average_precision_score(error, score))
    order = np.argsort(score, kind="stable")
    ordered_error = error[order]
    cumulative_risk = np.cumsum(ordered_error) / np.arange(1, len(error) + 1)
    selective = {}
    for coverage in map(float, config["thresholds"]["risk_coverage"]):
        retained = max(1, int(np.ceil(coverage * len(error))))
        selective[f"{coverage:.1f}"] = {
            "coverage": coverage,
            "retained_eyes": retained,
            "selective_risk": float(cumulative_risk[retained - 1]),
        }
    return {
        "classification_errors": int(error.sum()),
        "classification_error_rate": float(error.mean()),
        "error_detection_auroc": auroc,
        "error_detection_auprc": auprc,
        "aurc": float(cumulative_risk.mean()),
        "selective_risk": selective,
    }
