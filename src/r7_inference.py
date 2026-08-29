"""Patient-cluster bootstrap and frozen inferential helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm


def bca_mean_interval(
    contributions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha_low: float = 0.025,
    alpha_high: float = 0.975,
) -> dict[str, Any]:
    values = np.asarray(contributions, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "estimate": float(values.mean()) if len(values) else None,
            "lower": None,
            "upper": None,
            "method": "not_estimable",
            "n": int(len(values)),
        }
    estimate = float(values.mean())
    if np.ptp(values) == 0:
        return {
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "method": "percentile_point_mass_fallback",
            "replicates": int(replicates),
            "seed": int(seed),
            "n": int(len(values)),
            "significance_eligible": False,
        }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=float)
    for index in range(replicates):
        bootstrap[index] = values[rng.integers(0, len(values), len(values))].mean()

    fraction_less = np.clip(
        (np.count_nonzero(bootstrap < estimate) + 0.5 * np.count_nonzero(bootstrap == estimate))
        / replicates,
        1.0 / (2.0 * replicates),
        1.0 - 1.0 / (2.0 * replicates),
    )
    z0 = norm.ppf(fraction_less)
    total = values.sum()
    jackknife = (total - values) / (len(values) - 1)
    center = jackknife.mean()
    differences = center - jackknife
    denominator = 6.0 * np.sum(differences**2) ** 1.5
    acceleration = (
        float(np.sum(differences**3) / denominator) if denominator > 0 else 0.0
    )

    def adjusted(alpha: float) -> float:
        z = norm.ppf(alpha)
        value = norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z)))
        return float(np.clip(value, 0.0, 1.0))

    lower_probability = adjusted(alpha_low)
    upper_probability = adjusted(alpha_high)
    return {
        "estimate": estimate,
        "lower": float(np.quantile(bootstrap, lower_probability)),
        "upper": float(np.quantile(bootstrap, upper_probability)),
        "method": "BCa",
        "replicates": int(replicates),
        "seed": int(seed),
        "n": int(len(values)),
        "bias_correction": float(z0),
        "acceleration": acceleration,
        "adjusted_probabilities": [lower_probability, upper_probability],
        "significance_eligible": True,
    }


def one_sided_bca_lower(
    contributions: np.ndarray, *, replicates: int, seed: int, alpha: float
) -> dict[str, Any]:
    interval = bca_mean_interval(
        contributions,
        replicates=replicates,
        seed=seed,
        alpha_low=alpha,
        alpha_high=0.999999,
    )
    return {
        "estimate": interval["estimate"],
        "lower": interval["lower"],
        "alpha": alpha,
        "method": interval["method"],
        "significance_eligible": interval.get("significance_eligible", False),
        "replicates": interval.get("replicates"),
        "seed": seed,
    }


def patient_cluster_percentile_interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    patient_groups = [
        group.copy() for _, group in frame.groupby("patient_id", sort=True)
    ]
    estimate = float(statistic(frame))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(patient_groups), len(patient_groups))
        replicate = pd.concat(
            [patient_groups[index] for index in sampled], ignore_index=True
        )
        try:
            value = float(statistic(replicate))
        except ValueError:
            continue
        if np.isfinite(value):
            bootstrap.append(value)
    values = np.asarray(bootstrap, dtype=float)
    return {
        "estimate": estimate,
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "method": "patient-cluster percentile",
        "requested_replicates": int(replicates),
        "valid_replicates": int(len(values)),
        "seed": int(seed),
        "patients": int(len(patient_groups)),
    }
