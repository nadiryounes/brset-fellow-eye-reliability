"""Frozen patient-level folds and logistic disease heads."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedDiseaseHead:
    scaler: StandardScaler
    model: LogisticRegression
    convergence_warnings: list[str]

    def raw_logit(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features)
        return np.asarray(self.model.decision_function(transformed), dtype=float)

    def provenance(self) -> dict[str, Any]:
        return {
            "C": float(self.model.C),
            "solver": self.model.solver,
            "class_weight": self.model.class_weight,
            "max_iter": int(self.model.max_iter),
            "tol": float(self.model.tol),
            "n_iter": [int(value) for value in self.model.n_iter_],
            "convergence_warnings": self.convergence_warnings,
        }


def patient_state_table(metadata: pd.DataFrame) -> pd.DataFrame:
    right = metadata[metadata["exam_eye"] == 1][
        ["patient_id", "drusen"]
    ].rename(columns={"drusen": "drusen_R"})
    left = metadata[metadata["exam_eye"] == 2][
        ["patient_id", "drusen"]
    ].rename(columns={"drusen": "drusen_L"})
    patients = right.merge(left, on="patient_id", how="inner", validate="one_to_one")
    patients["drusen_state"] = (
        patients["drusen_R"].astype(str) + patients["drusen_L"].astype(str)
    )
    patients["_numeric"] = pd.to_numeric(patients["patient_id"], errors="coerce")
    patients["_string"] = patients["patient_id"].astype(str)
    patients = patients.sort_values(["_numeric", "_string"], kind="stable")
    return patients.drop(columns=["_numeric", "_string"]).reset_index(drop=True)


def make_outer_patient_folds(
    metadata: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    patients = patient_state_table(metadata)
    splitter = StratifiedKFold(
        n_splits=int(config["cross_validation"]["outer_folds"]),
        shuffle=True,
        random_state=int(config["cross_validation"]["outer_seed"]),
    )
    patients["outer_fold"] = -1
    for fold, (_, assessment) in enumerate(
        splitter.split(patients["patient_id"], patients["drusen_state"])
    ):
        patients.loc[assessment, "outer_fold"] = fold
    if (patients["outer_fold"] < 0).any():
        raise RuntimeError("Outer fold assignment incomplete")
    _assert_fold_states(patients, "outer_fold", config["cross_validation"]["outer_folds"])
    return patients[["patient_id", "drusen_state", "outer_fold"]].copy()


def make_inner_patient_folds(
    metadata: pd.DataFrame, outer_index: int, config: dict[str, Any]
) -> pd.DataFrame:
    patients = patient_state_table(metadata)
    seed = int(config["cross_validation"]["inner_seed_base"]) + int(outer_index)
    splitter = StratifiedKFold(
        n_splits=int(config["cross_validation"]["inner_folds"]),
        shuffle=True,
        random_state=seed,
    )
    patients["inner_fold"] = -1
    for fold, (_, assessment) in enumerate(
        splitter.split(patients["patient_id"], patients["drusen_state"])
    ):
        patients.loc[assessment, "inner_fold"] = fold
    if (patients["inner_fold"] < 0).any():
        raise RuntimeError("Inner fold assignment incomplete")
    _assert_fold_states(patients, "inner_fold", config["cross_validation"]["inner_folds"])
    return patients[["patient_id", "drusen_state", "inner_fold"]].copy()


def _assert_fold_states(
    patients: pd.DataFrame, fold_column: str, expected_folds: int
) -> None:
    if patients["patient_id"].duplicated().any():
        raise RuntimeError("Patient duplicated in fold table")
    if patients[fold_column].nunique() != int(expected_folds):
        raise RuntimeError(f"Unexpected number of folds for {fold_column}")
    for _, group in patients.groupby(fold_column):
        if set(group["drusen_state"]) != {"00", "11", "10", "01"}:
            raise RuntimeError(f"A {fold_column} lacks a drusen state")


def row_fold_ids(metadata: pd.DataFrame, folds: pd.DataFrame, column: str) -> np.ndarray:
    mapping = folds.set_index("patient_id")[column]
    values = metadata["patient_id"].map(mapping)
    if values.isna().any():
        raise RuntimeError(f"Missing row-level fold assignment: {column}")
    return values.astype(int).to_numpy()


def fit_disease_head(
    features: np.ndarray,
    labels: np.ndarray,
    C: float,
    config: dict[str, Any],
    *,
    seed: int,
) -> FittedDiseaseHead:
    head = config["disease_head"]
    scaler = StandardScaler()
    transformed = scaler.fit_transform(np.asarray(features))
    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver=head["solver"],
        class_weight=head["class_weight"],
        fit_intercept=bool(head["fit_intercept"]),
        max_iter=int(head["max_iter"]),
        tol=float(head["tol"]),
        random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(transformed, np.asarray(labels, dtype=int))
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return FittedDiseaseHead(scaler, model, convergence)


def modal_choice(values: list[Any], *, tie: str) -> Any:
    counts = pd.Series(values).value_counts()
    maximum = counts.max()
    candidates = list(counts[counts == maximum].index)
    if tie == "smaller_numeric":
        return min(float(value) for value in candidates)
    if tie == "larger_numeric":
        return max(float(value) for value in candidates)
    if tie == "temperature":
        return "temperature" if "temperature" in candidates else sorted(candidates)[0]
    raise ValueError(f"Unknown modal tie rule: {tie}")
