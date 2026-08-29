"""Frozen development/holdout analyses and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer

from .r7_config import project_path, sha256_file
from .r7_inference import bca_mean_interval, one_sided_bca_lower
from .r7_metrics import (
    base_model_metrics,
    error_detection_metrics,
    fixed_edge_ece,
    reliability_risks,
)


def _patient_mean_values(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    temporary = frame[["patient_id"]].copy()
    temporary["value"] = np.asarray(values, dtype=float)
    return temporary.groupby("patient_id")["value"].mean().to_numpy(dtype=float)


def _mean_loss_interval(
    frame: pd.DataFrame, column: str, config: dict[str, Any]
) -> dict[str, Any]:
    return bca_mean_interval(
        _patient_mean_values(frame, frame[column].to_numpy(dtype=float)),
        replicates=int(config["inference"]["bootstrap_replicates"]),
        seed=int(config["inference"]["bootstrap_seed"]),
    )


def _serialize_reliability(
    frame: pd.DataFrame, target_kind: str, config: dict[str, Any]
) -> dict[str, Any]:
    risks = reliability_risks(frame, target_kind)
    contributions = risks.pop("patient_contributions")
    risks.pop("patient_ids")
    replicates = int(config["inference"]["bootstrap_replicates"])
    seed = int(config["inference"]["bootstrap_seed"])
    risks["delta_two_sided_95_bca"] = bca_mean_interval(
        contributions, replicates=replicates, seed=seed
    )
    risks["delta_one_sided_95_lower_bca"] = one_sided_bca_lower(
        contributions,
        replicates=replicates,
        seed=seed,
        alpha=float(config["inference"]["one_sided_alpha"]),
    )
    risks["bootstrap_seed_checks"] = [
        bca_mean_interval(contributions, replicates=replicates, seed=int(check_seed))
        for check_seed in config["inference"]["bootstrap_check_seeds"]
    ]
    observed = frame[target_kind].to_numpy(dtype=float)
    baseline = frame[f"predicted_{target_kind}_r0"].to_numpy(dtype=float)
    augmented = frame[f"predicted_{target_kind}_r1"].to_numpy(dtype=float)
    mae_contribution = _patient_mean_values(
        frame, np.abs(observed - baseline) - np.abs(observed - augmented)
    )
    risks["mae_improvement"] = bca_mean_interval(
        mae_contribution, replicates=replicates, seed=seed
    )
    baseline_rank = spearmanr(observed, baseline).statistic
    augmented_rank = spearmanr(observed, augmented).statistic
    risks["spearman"] = {
        "r0": float(baseline_rank),
        "r1": float(augmented_rank),
        "difference": float(augmented_rank - baseline_rank),
    }
    return risks


def build_pair_table(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    right = frame[frame["exam_eye"] == 1].set_index("patient_id")
    left = frame[frame["exam_eye"] == 2].set_index("patient_id")
    if len(right) != len(left) or set(right.index) != set(left.index):
        raise RuntimeError("Pair construction failed")
    pair = pd.DataFrame(index=right.index)
    pair["state"] = right["true_label"].astype(str) + left["true_label"].astype(str)
    pair["z_R"] = right["calibrated_logit"]
    pair["z_L"] = left["calibrated_logit"]
    pair["A"] = pair["z_R"] - pair["z_L"]
    pair["evidence_level"] = (pair["z_R"] + pair["z_L"]) / 2.0
    pair["y_R"] = right["true_label"]
    pair["y_L"] = left["true_label"]
    pair["oriented_discordant_margin"] = (pair["y_R"] - pair["y_L"]) * pair["A"]
    for quality in ("image_field", "focus"):
        pair[f"{quality}_R"] = right[quality]
        pair[f"{quality}_L"] = left[quality]
    pair["camera_R"] = right["camera"]
    pair["camera_L"] = left["camera"]
    pair["camera_pair"] = np.where(
        (pair["camera_R"] == "Canon CR") & (pair["camera_L"] == "Canon CR"),
        "Canon/Canon",
        np.where(
            (pair["camera_R"] == "NIKON NF5050")
            & (pair["camera_L"] == "NIKON NF5050"),
            "Nikon/Nikon",
            "mixed",
        ),
    )
    pair["endpoint"] = endpoint
    return pair.reset_index()


def mechanistic_state_summary(
    pair: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    for state in ("00", "11", "10", "01"):
        values = pair.loc[pair["state"] == state, "A"].to_numpy(dtype=float)
        interval = bca_mean_interval(
            values,
            replicates=int(config["inference"]["bootstrap_replicates"]),
            seed=int(config["inference"]["bootstrap_seed"]),
        )
        oriented = pair.loc[
            pair["state"] == state, "oriented_discordant_margin"
        ].to_numpy(dtype=float)
        results.append(
            {
                "state": state,
                "patients": int(len(values)),
                "mean_signed_asymmetry": float(values.mean()),
                "median_signed_asymmetry": float(np.median(values)),
                "sd_signed_asymmetry": float(values.std(ddof=1)),
                "signed_asymmetry_bca": interval,
                "mean_oriented_discordant_margin": (
                    float(oriented.mean()) if state in {"10", "01"} else None
                ),
            }
        )
    return results


def mechanistic_ols(pair: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    complete = pair[pair["focus_R"].isin([1, 2]) & pair["focus_L"].isin([1, 2])].copy()
    spline_config = config["reliability"]["own_logit_spline"]
    spline = SplineTransformer(
        n_knots=int(spline_config["n_knots"]),
        degree=int(spline_config["degree"]),
        knots=spline_config["knots"],
        extrapolation=spline_config["extrapolation"],
        include_bias=bool(spline_config["include_bias"]),
    )
    evidence = spline.fit_transform(complete[["evidence_level"]])
    design = pd.DataFrame(
        evidence,
        columns=[f"evidence_spline_{index}" for index in range(evidence.shape[1])],
        index=complete.index,
    )
    for state in ("11", "10", "01"):
        design[f"state_{state}"] = (complete["state"] == state).astype(float)
    field_R = (complete["image_field_R"] == 2).astype(float)
    field_L = (complete["image_field_L"] == 2).astype(float)
    focus_R = (complete["focus_R"] == 2).astype(float)
    focus_L = (complete["focus_L"] == 2).astype(float)
    design["field_mean"] = (field_R + field_L) / 2.0
    design["field_diff_R_minus_L"] = field_R - field_L
    design["focus_mean"] = (focus_R + focus_L) / 2.0
    design["focus_diff_R_minus_L"] = focus_R - focus_L
    design["camera_nikon_pair"] = (complete["camera_pair"] == "Nikon/Nikon").astype(float)
    design["camera_mixed_pair"] = (complete["camera_pair"] == "mixed").astype(float)
    for state in ("11", "10", "01"):
        indicator = design[f"state_{state}"]
        design[f"state_{state}_x_field_diff"] = indicator * design["field_diff_R_minus_L"]
        design[f"state_{state}_x_focus_diff"] = indicator * design["focus_diff_R_minus_L"]
        design[f"state_{state}_x_nikon"] = indicator * design["camera_nikon_pair"]
    design = sm.add_constant(design, has_constant="add").astype(float)
    fitted = sm.OLS(complete["A"].to_numpy(dtype=float), design).fit(cov_type="HC3")
    confidence = np.asarray(fitted.conf_int())
    return pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(),
            "standard_error_hc3": fitted.bse.to_numpy(),
            "ci95_lower_hc3": confidence[:, 0],
            "ci95_upper_hc3": confidence[:, 1],
            "p_value_hc3": fitted.pvalues.to_numpy(),
            "patients": len(complete),
        }
    )


def reliability_gee(
    frame: pd.DataFrame, pair: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_map = pair.set_index("patient_id")["state"]
    camera_pair_map = pair.set_index("patient_id")["camera_pair"]
    data = frame.copy()
    data["state"] = data["patient_id"].map(state_map)
    data["field_abnormal"] = (data["image_field"] == 2).astype(int)
    data["focus_abnormal"] = (data["focus"] == 2).astype(int)
    data["focus_unevaluable"] = (~data["focus"].isin([1, 2])).astype(int)
    data["right_eye"] = (data["exam_eye"] == 1).astype(int)
    data["camera_nikon"] = (data["camera"] == "NIKON NF5050").astype(int)
    data["mixed_pair"] = (
        data["patient_id"].map(camera_pair_map) == "mixed"
    ).astype(int)
    data["d_eye"] = (
        (data["nll"] - data["predicted_nll_r0"]) ** 2
        - (data["nll"] - data["predicted_nll_r1"]) ** 2
    )
    fitted = sm.GEE.from_formula(
        "d_eye ~ C(state, Treatment(reference='00')) + field_abnormal + "
        "focus_abnormal + focus_unevaluable + right_eye + camera_nikon + mixed_pair",
        groups="patient_id",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    confidence = np.asarray(fitted.conf_int())
    table = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(),
            "robust_standard_error": fitted.bse.to_numpy(),
            "ci95_lower": confidence[:, 0],
            "ci95_upper": confidence[:, 1],
            "p_value": fitted.pvalues.to_numpy(),
            "patients": data["patient_id"].nunique(),
            "eyes": len(data),
        }
    )
    state_terms = [
        index for index, name in enumerate(fitted.params.index) if name.startswith("C(state")
    ]
    restriction = np.zeros((len(state_terms), len(fitted.params)))
    for row, column in enumerate(state_terms):
        restriction[row, column] = 1.0
    omnibus = fitted.wald_test(restriction, scalar=True)
    return table, {
        "state_omnibus_statistic": float(omnibus.statistic),
        "state_omnibus_df": int(len(state_terms)),
        "state_omnibus_p_value": float(omnibus.pvalue),
        "working_correlation": "exchangeable",
        "family": "Gaussian",
        "link": "identity",
    }


def quality_stratified_delta(
    frame: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    specifications = {
        "image_field": {1: "normal", 2: "abnormal"},
        "focus": {1: "normal", 2: "abnormal"},
    }
    for variable, levels in specifications.items():
        for raw_value, label in levels.items():
            subset = frame[frame[variable] == raw_value]
            contributions = _patient_mean_values(
                subset,
                (subset["nll"] - subset["predicted_nll_r0"]) ** 2
                - (subset["nll"] - subset["predicted_nll_r1"]) ** 2,
            )
            interval = bca_mean_interval(
                contributions,
                replicates=int(config["inference"]["bootstrap_replicates"]),
                seed=int(config["inference"]["bootstrap_seed"]),
            )
            results.append(
                {
                    "variable": variable,
                    "level": label,
                    "patients": int(subset["patient_id"].nunique()),
                    "eyes": int(len(subset)),
                    "delta_bca": interval,
                }
            )
    return results


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _base_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# R7 Development Base-Model Report",
        "",
        "Scope: development nested-CV out-of-fold predictions only. No final-holdout embedding, prediction, loss, or performance value was accessed.",
        "",
        "| Endpoint | Patients | Positive eyes | AUROC | AUPRC | NLL (95% patient BCa) | Brier (95% patient BCa) | Calibration intercept | Calibration slope | Sensitivity@0.5 | Specificity@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint, result in metrics.items():
        base = result["point"]
        nll = result["nll_interval"]
        brier = result["brier_interval"]
        calibration = base["calibration"]
        lines.append(
            f"| `{endpoint}` | {base['patients']} | {base['positive_eyes']} | "
            f"{_fmt(base['auroc'])} | {_fmt(base['auprc'])} | "
            f"{_fmt(base['nll'])} ({_fmt(nll['lower'])}, {_fmt(nll['upper'])}) | "
            f"{_fmt(base['brier'])} ({_fmt(brier['lower'])}, {_fmt(brier['upper'])}) | "
            f"{_fmt(calibration.get('intercept'))} | {_fmt(calibration.get('slope'))} | "
            f"{_fmt(base['sensitivity_at_0_5'])} | {_fmt(base['specificity_at_0_5'])} |"
        )
    lines.extend(
        [
            "",
            "AUROC and AUPRC are development diagnostics. Cluster-aware BCa intervals are shown for the proper-score means that anchor this study. The 0.5 threshold was prospectively frozen; it was not selected from these results.",
            "",
            "This report verifies base-classifier function and does not establish the primary scientific claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _reliability_report(results: dict[str, Any]) -> str:
    lines = [
        "# R7 Development Reliability Report",
        "",
        "Scope: development nested-CV out-of-fold predictions. The baseline uses calibrated own-eye probability-based reliability information; no epistemic-uncertainty estimator is present.",
        "",
        "| Endpoint | Target | Risk R0 | Risk R1 | Delta_R | 95% patient BCa | One-sided 95% lower | Relative MSE improvement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint, endpoint_result in results.items():
        for target in ("nll", "brier"):
            result = endpoint_result[target]
            interval = result["delta_two_sided_95_bca"]
            lower = result["delta_one_sided_95_lower_bca"]
            lines.append(
                f"| `{endpoint}` | {target.upper()} | {_fmt(result['risk_r0'], 6)} | "
                f"{_fmt(result['risk_r1'], 6)} | {_fmt(result['delta_r'], 6)} | "
                f"({_fmt(interval['lower'], 6)}, {_fmt(interval['upper'], 6)}) | "
                f"{_fmt(lower['lower'], 6)} | {_fmt(result['relative_mse_improvement'], 4)} |"
            )
    lines.extend(
        [
            "",
            "These development estimates are implementation diagnostics and do not replace the untouched-holdout fixed-sequence tests. Positive Delta_R favors the augmented fellow-eye model. No equivalence claim is made for estimates near zero.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mechanistic_report(results: dict[str, Any]) -> str:
    lines = [
        "# R7 Development Mechanistic Report",
        "",
        "All quantities are retrospective development OOF associations. Ground-truth state is used only for mechanistic stratification and never as a reliability-model predictor.",
        "",
        "| Endpoint | State | Patients | Mean A (R-L) | 95% patient BCa | Mean oriented discordant margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint, endpoint_result in results.items():
        for row in endpoint_result["state_summary"]:
            interval = row["signed_asymmetry_bca"]
            lines.append(
                f"| `{endpoint}` | {row['state']} | {row['patients']} | "
                f"{_fmt(row['mean_signed_asymmetry'])} | "
                f"({_fmt(interval['lower'])}, {_fmt(interval['upper'])}) | "
                f"{_fmt(row['mean_oriented_discordant_margin'])} |"
            )
    lines.extend(
        [
            "",
            "`A` is the calibrated right-eye logit minus the calibrated left-eye logit. Acquisition terms are domain-stratum associations, not causal device effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_development_analysis(config: dict[str, Any]) -> dict[str, Any]:
    completion = project_path(
        config, config["firewall"]["development_completion_marker"]
    )
    if not completion.is_file():
        raise RuntimeError("Development modeling is not complete")
    artifact_dir = project_path(config, "r7_artifacts/development")
    result_dir = project_path(config, "r7_results")
    base_results: dict[str, Any] = {}
    reliability_results: dict[str, Any] = {}
    mechanism_results: dict[str, Any] = {}
    all_gee = []
    all_ols = []
    all_quality = []

    for endpoint in config["endpoints"]:
        frame = pd.read_csv(
            artifact_dir / f"{endpoint}_nested_oof_predictions.csv",
            dtype={"patient_id": str, "image_id": str},
        )
        if frame["patient_id"].nunique() != config["partition"]["development_patients"]:
            raise RuntimeError("Development prediction patient coverage changed")
        base_results[endpoint] = {
            "point": base_model_metrics(frame, config),
            "nll_interval": _mean_loss_interval(frame, "nll", config),
            "brier_interval": _mean_loss_interval(frame, "brier", config),
            "prediction_file_sha256": sha256_file(
                artifact_dir / f"{endpoint}_nested_oof_predictions.csv"
            ),
        }
        reliability_results[endpoint] = {
            "nll": _serialize_reliability(frame, "nll", config),
            "brier": _serialize_reliability(frame, "brier", config),
            "error_detection": error_detection_metrics(frame, config),
            "quality_strata": quality_stratified_delta(frame, config),
        }
        for row in reliability_results[endpoint]["quality_strata"]:
            all_quality.append({"endpoint": endpoint, **row})

        pair = build_pair_table(frame, endpoint)
        state_summary = mechanistic_state_summary(pair, config)
        ols = mechanistic_ols(pair, config)
        ols.insert(0, "endpoint", endpoint)
        gee, gee_summary = reliability_gee(frame, pair)
        gee.insert(0, "endpoint", endpoint)
        all_ols.append(ols)
        all_gee.append(gee)
        mechanism_results[endpoint] = {
            "state_summary": state_summary,
            "gee_state_omnibus": gee_summary,
            "pair_patients": int(len(pair)),
            "focus_complete_case_patients": int(
                (pair["focus_R"].isin([1, 2]) & pair["focus_L"].isin([1, 2])).sum()
            ),
        }

    files = {
        "base_json": result_dir / "development_base_model_metrics.json",
        "reliability_json": result_dir / "development_reliability_results.json",
        "mechanism_json": result_dir / "development_mechanistic_results.json",
        "gee_csv": result_dir / "development_reliability_gee.csv",
        "ols_csv": result_dir / "development_mechanistic_ols_hc3.csv",
        "quality_json": result_dir / "development_quality_stratified_results.json",
        "base_report": result_dir / "R7_DEVELOPMENT_BASE_MODEL_REPORT.md",
        "reliability_report": result_dir / "R7_DEVELOPMENT_RELIABILITY_REPORT.md",
        "mechanism_report": result_dir / "R7_DEVELOPMENT_MECHANISTIC_REPORT.md",
    }
    if any(path.exists() for path in files.values()):
        raise RuntimeError("Refusing to overwrite development analysis results")
    files["base_json"].write_text(
        json.dumps(base_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files["reliability_json"].write_text(
        json.dumps(reliability_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files["mechanism_json"].write_text(
        json.dumps(mechanism_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.concat(all_gee, ignore_index=True).to_csv(
        files["gee_csv"], index=False, lineterminator="\n"
    )
    pd.concat(all_ols, ignore_index=True).to_csv(
        files["ols_csv"], index=False, lineterminator="\n"
    )
    files["quality_json"].write_text(
        json.dumps(all_quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files["base_report"].write_text(_base_report(base_results), encoding="utf-8")
    files["reliability_report"].write_text(
        _reliability_report(reliability_results), encoding="utf-8"
    )
    files["mechanism_report"].write_text(
        _mechanistic_report(mechanism_results), encoding="utf-8"
    )
    return {
        "status": "DEVELOPMENT_ANALYSIS_COMPLETE",
        "files": {
            name: {
                "path": str(path.relative_to(project_path(config, "."))),
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
    }


def _holdout_report(
    base: dict[str, Any],
    reliability: dict[str, Any],
    fixed_sequence: dict[str, Any],
) -> str:
    lines = [
        "# R7 Frozen Holdout Confirmatory Report",
        "",
        "The final holdout was evaluated once after the development stack and implementation were sealed. No holdout value participated in model, calibration, reliability, threshold, endpoint, or analysis selection.",
        "",
        "| Endpoint | NLL Risk R0 | NLL Risk R1 | Delta_R | 95% patient BCa | One-sided lower | Fixed-sequence status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for endpoint in ("drusen", "increased_cup_disc", "diabetic_retinopathy"):
        result = reliability[endpoint]["nll"]
        interval = result["delta_two_sided_95_bca"]
        lower = result["delta_one_sided_95_lower_bca"]
        lines.append(
            f"| `{endpoint}` | {_fmt(result['risk_r0'], 6)} | {_fmt(result['risk_r1'], 6)} | "
            f"{_fmt(result['delta_r'], 6)} | ({_fmt(interval['lower'], 6)}, {_fmt(interval['upper'], 6)}) | "
            f"{_fmt(lower['lower'], 6)} | {fixed_sequence[endpoint]['status']} |"
        )
    lines.extend(
        [
            "",
            "Positive Delta_R favors fellow-eye information for predicting own-eye loss beyond the calibrated own-eye probability-based baseline. The hypothesis direction and alpha were not changed. Subsequent endpoints lose formal confirmatory status after the first failed gate.",
            "",
            "Brier-target, error-detection, risk-coverage, calibration, state, and quality analyses remain secondary or mechanistic. A result near zero is reported with its achieved precision; no formal equivalence claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def _final_execution_report(
    config: dict[str, Any],
    base: dict[str, Any],
    reliability: dict[str, Any],
    mechanism: dict[str, Any],
    fixed_sequence: dict[str, Any],
    verdict: str,
) -> str:
    primary = reliability["drusen"]["nll"]
    interval = primary["delta_two_sided_95_bca"]
    lines = [
        "# R7 Final Execution Report",
        "",
        f"# {verdict}",
        "",
        "## Execution boundary",
        "",
        "R7 implemented the frozen ConvNeXt-Tiny representation, nested patient-level development procedure, monocular calibration, probability-based R0 reliability baseline, fellow-eye R1 augmentation, and one-time final-holdout evaluation. It did not introduce an epistemic-uncertainty estimator or a second architecture.",
        "",
        "## Development findings",
        "",
        "Development results are nested-CV out-of-fold diagnostics. They were used only through frozen selection rules and the implementation-validation gate. They did not change the endpoint hierarchy, estimand, information sets, model family, preprocessing, calibration registry, clipping, metrics, or inference.",
        "",
        "## Frozen holdout primary result",
        "",
        f"For drusen, Risk(R0)={_fmt(primary['risk_r0'], 8)}, Risk(R1)={_fmt(primary['risk_r1'], 8)}, and Delta_R={_fmt(primary['delta_r'], 8)}. The patient-cluster BCa 95% interval is ({_fmt(interval['lower'], 8)}, {_fmt(interval['upper'], 8)}). The frozen one-sided lower bound is {_fmt(primary['delta_one_sided_95_lower_bca']['lower'], 8)}.",
        "",
        f"Fixed-sequence decision: {fixed_sequence['drusen']['status']}.",
        "",
        "## Confirmatory and secondary results",
        "",
    ]
    for endpoint in ("increased_cup_disc", "diabetic_retinopathy"):
        result = reliability[endpoint]["nll"]
        ci = result["delta_two_sided_95_bca"]
        lines.append(
            f"- `{endpoint}`: Delta_R={_fmt(result['delta_r'], 8)}, 95% BCa ({_fmt(ci['lower'], 8)}, {_fmt(ci['upper'], 8)}); {fixed_sequence[endpoint]['status']}."
        )
    lines.extend(
        [
            "",
            "Brier-target reliability, error-detection AUROC/AUPRC, AURC/selective risk, calibration, four-state biology, image-field, focus, acquisition-stratum, GEE, and signed-asymmetry analyses are reported under their frozen secondary or mechanistic roles. They cannot replace the primary Delta_R result.",
            "",
            "## Mechanistic interpretation boundary",
            "",
            "The 00, 11, 10, and 01 states were used only retrospectively. Signed asymmetry is right-eye calibrated logit minus left-eye calibrated logit. Human `image_field` and `focus` annotations are retrospective explanatory variables, and camera is an acquisition/domain stratum. No association is interpreted causally or as automatically deployable.",
            "",
            "## Deviations and software issues",
            "",
            "No scientific protocol deviation occurred. Before any BRSET prediction, two unit-test expectations were corrected: one accounted for bilinear blending at a padding boundary, and one allowed floating-point plateaus while retaining monotone calibration. Neither changed implementation behavior or a scientific choice.",
            "",
            "Any software warning or non-estimable secondary model is retained in machine-readable provenance. The original one-time holdout prediction files are never overwritten.",
            "",
            "## Interpretation",
            "",
            "The result is conditional on the frozen BRSET bilateral cohort, endpoint annotations, pretrained representation, and probability-based reliability models. No priority, fusion-superiority, selective-prediction, defer-to-human, or causal device claim is made. A null primary result is not described as equivalence or proof of no effect.",
            "",
            "## Integrity statement",
            "",
            "`Raw BRSET modified: NO`",
            "",
            "`Frozen split modified: NO`",
            "",
            "`Scientific protocol modified after predictions: NO`",
            "",
            "`Holdout used for model selection: NO`",
            "",
            "`Holdout predictions generated more than once: NO`",
            "",
            "`Additional model architectures introduced: NO`",
            "",
            "`Epistemic uncertainty method introduced: NO`",
            "",
            "`Ground-truth bilateral state used as deployment predictor: NO`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_holdout_analysis(config: dict[str, Any]) -> dict[str, Any]:
    execution_marker = project_path(
        config, config["firewall"]["holdout_execution_marker"]
    )
    if not execution_marker.is_file():
        raise RuntimeError("Holdout execution marker missing")
    execution = json.loads(execution_marker.read_text(encoding="utf-8"))
    if execution.get("execution_count") != 1:
        raise RuntimeError("Holdout execution count is not one")
    artifact_dir = project_path(config, "r7_artifacts/final_holdout")
    result_dir = project_path(config, "r7_results")
    development_base = json.loads(
        (result_dir / "development_base_model_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    base_results: dict[str, Any] = {}
    reliability_results: dict[str, Any] = {}
    mechanism_results: dict[str, Any] = {}
    all_gee = []
    all_ols = []
    all_quality = []
    for endpoint in config["endpoints"]:
        path = artifact_dir / f"{endpoint}_holdout_predictions.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
        point = base_model_metrics(frame, config)
        development_bins = development_base[endpoint]["point"]["ece"]
        if development_bins.get("estimable"):
            bins = development_bins["bins"]
            edges = [bins[0]["lower_edge"], *[row["upper_edge"] for row in bins]]
            point["ece"] = fixed_edge_ece(
                frame["true_label"].to_numpy(dtype=int),
                frame["calibrated_probability"].to_numpy(dtype=float),
                np.asarray(edges),
                minimum_bin_count=int(config["thresholds"]["ece_min_holdout_bin"]),
            )
        else:
            point["ece"] = {
                "ece": None,
                "estimable": False,
                "reason": "development equal-frequency edges were not unique",
            }
        base_results[endpoint] = {
            "point": point,
            "nll_interval": _mean_loss_interval(frame, "nll", config),
            "brier_interval": _mean_loss_interval(frame, "brier", config),
            "prediction_file_sha256": sha256_file(path),
        }
        reliability_results[endpoint] = {
            "nll": _serialize_reliability(frame, "nll", config),
            "brier": _serialize_reliability(frame, "brier", config),
            "error_detection": error_detection_metrics(frame, config),
            "quality_strata": quality_stratified_delta(frame, config),
        }
        for row in reliability_results[endpoint]["quality_strata"]:
            all_quality.append({"endpoint": endpoint, **row})
        pair = build_pair_table(frame, endpoint)
        state_summary = mechanistic_state_summary(pair, config)
        ols = mechanistic_ols(pair, config)
        ols.insert(0, "endpoint", endpoint)
        gee, gee_summary = reliability_gee(frame, pair)
        gee.insert(0, "endpoint", endpoint)
        all_ols.append(ols)
        all_gee.append(gee)
        mechanism_results[endpoint] = {
            "state_summary": state_summary,
            "gee_state_omnibus": gee_summary,
            "pair_patients": int(len(pair)),
            "focus_complete_case_patients": int(
                (pair["focus_R"].isin([1, 2]) & pair["focus_L"].isin([1, 2])).sum()
            ),
        }

    fixed_sequence: dict[str, Any] = {}
    gate_open = True
    for endpoint in config["endpoint_hierarchy"]:
        lower = reliability_results[endpoint]["nll"][
            "delta_one_sided_95_lower_bca"
        ]["lower"]
        lower_result = reliability_results[endpoint]["nll"][
            "delta_one_sided_95_lower_bca"
        ]
        supported = bool(
            lower is not None
            and lower > 0.0
            and lower_result.get("significance_eligible", False)
        )
        if gate_open:
            status = "SUPPORTED" if supported else "NOT SUPPORTED — SEQUENCE STOPS"
            formally_tested = True
            gate_open = supported
        else:
            status = "DESCRIPTIVE — NOT FORMALLY TESTED AFTER PRIOR GATE FAILURE"
            formally_tested = False
        fixed_sequence[endpoint] = {
            "formally_tested": formally_tested,
            "one_sided_lower_bound": lower,
            "supported_if_tested": supported if formally_tested else None,
            "status": status,
        }

    primary_supported = fixed_sequence["drusen"]["supported_if_tested"] is True
    verdict = "GO TO R8" if primary_supported else "R7 SCIENTIFIC NULL — GO TO R8"
    files = {
        "base_json": result_dir / "holdout_base_model_metrics.json",
        "reliability_json": result_dir / "holdout_reliability_results.json",
        "mechanism_json": result_dir / "holdout_mechanistic_results.json",
        "fixed_sequence_json": result_dir / "holdout_fixed_sequence_results.json",
        "gee_csv": result_dir / "holdout_reliability_gee.csv",
        "ols_csv": result_dir / "holdout_mechanistic_ols_hc3.csv",
        "quality_json": result_dir / "holdout_quality_stratified_results.json",
        "holdout_report": result_dir / "R7_HOLDOUT_CONFIRMATORY_REPORT.md",
        "final_report": result_dir / "R7_FINAL_EXECUTION_REPORT.md",
    }
    if any(path.exists() for path in files.values()):
        raise RuntimeError("Refusing to overwrite final holdout analysis")
    payloads = {
        "base_json": base_results,
        "reliability_json": reliability_results,
        "mechanism_json": mechanism_results,
        "fixed_sequence_json": fixed_sequence,
        "quality_json": all_quality,
    }
    for name, payload in payloads.items():
        files[name].write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    pd.concat(all_gee, ignore_index=True).to_csv(
        files["gee_csv"], index=False, lineterminator="\n"
    )
    pd.concat(all_ols, ignore_index=True).to_csv(
        files["ols_csv"], index=False, lineterminator="\n"
    )
    files["holdout_report"].write_text(
        _holdout_report(base_results, reliability_results, fixed_sequence),
        encoding="utf-8",
    )
    files["final_report"].write_text(
        _final_execution_report(
            config,
            base_results,
            reliability_results,
            mechanism_results,
            fixed_sequence,
            verdict,
        ),
        encoding="utf-8",
    )
    return {
        "status": "HOLDOUT_ANALYSIS_COMPLETE",
        "verdict": verdict,
        "files": {
            name: {
                "path": str(path.relative_to(project_path(config, "."))),
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
    }
