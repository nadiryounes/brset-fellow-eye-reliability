#!/usr/bin/env python3
"""R8 analysis-only derivations from sealed R7 prediction artifacts.

This script never writes outside r8_analysis and never fits or changes a model.
It verifies the R7 SHA-256 ledger before reading the sealed holdout predictions.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "r8_analysis"
FIG = OUT / "figures"
ENDPOINTS = ("drusen", "increased_cup_disc", "diabetic_retinopathy")
LABELS = {
    "drusen": "Drusen",
    "increased_cup_disc": "Increased cup/disc",
    "diabetic_retinopathy": "Diabetic retinopathy",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r7_ledger() -> list[dict[str, Any]]:
    ledger = ROOT / "r7_artifacts/R7_SHA256SUMS.txt"
    rows: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256_file(ROOT / relative)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    failures = [row for row in rows if not row["match"]]
    if failures:
        raise RuntimeError(f"R8 INTEGRITY FAILURE: {failures}")
    return rows


def load_predictions(endpoint: str) -> pd.DataFrame:
    path = ROOT / f"r7_artifacts/final_holdout/{endpoint}_holdout_predictions.csv"
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
    if len(frame) != 3848 or frame["patient_id"].nunique() != 1924:
        raise RuntimeError(f"R8 INTEGRITY FAILURE: coverage changed for {endpoint}")
    if frame.duplicated(["patient_id", "exam_eye"]).any():
        raise RuntimeError(f"R8 INTEGRITY FAILURE: duplicate eye for {endpoint}")
    if set(frame["exam_eye"].unique()) != {1, 2}:
        raise RuntimeError(f"R8 INTEGRITY FAILURE: laterality coding changed for {endpoint}")
    return frame


def patient_risk_frame(frame: pd.DataFrame, target: str = "nll") -> pd.DataFrame:
    data = frame[["patient_id"]].copy()
    observed = frame[target].to_numpy(dtype=float)
    data["observed_loss"] = observed
    data["risk_r0"] = (observed - frame[f"predicted_{target}_r0"].to_numpy(dtype=float)) ** 2
    data["risk_r1"] = (observed - frame[f"predicted_{target}_r1"].to_numpy(dtype=float)) ** 2
    result = data.groupby("patient_id", sort=True).mean()
    result["contribution"] = result["risk_r0"] - result["risk_r1"]
    return result


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=float)
    # Chunking preserves the RNG stream while avoiding one large index matrix.
    cursor = 0
    chunk = 250
    while cursor < replicates:
        size = min(chunk, replicates - cursor)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        output[cursor : cursor + size] = values[indices].mean(axis=1)
        cursor += size
    return output


def bca_mean(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    alpha_low: float = 0.025,
    alpha_high: float = 0.975,
) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "estimate": float(values.mean()) if len(values) else None,
            "lower": None,
            "upper": None,
            "method": "not_estimable",
            "patients": int(len(values)),
            "valid_replicates": 0,
            "failures": BOOTSTRAP_REPLICATES,
        }
    estimate = float(values.mean())
    if np.ptp(values) == 0:
        return {
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "method": "point_mass_percentile_fallback",
            "patients": int(len(values)),
            "valid_replicates": BOOTSTRAP_REPLICATES,
            "failures": 0,
        }
    bootstrap = bootstrap_mean(values, BOOTSTRAP_REPLICATES, seed)
    fraction_less = np.clip(
        (
            np.count_nonzero(bootstrap < estimate)
            + 0.5 * np.count_nonzero(bootstrap == estimate)
        )
        / BOOTSTRAP_REPLICATES,
        1.0 / (2.0 * BOOTSTRAP_REPLICATES),
        1.0 - 1.0 / (2.0 * BOOTSTRAP_REPLICATES),
    )
    z0 = norm.ppf(fraction_less)
    jackknife = (values.sum() - values) / (len(values) - 1)
    differences = jackknife.mean() - jackknife
    denominator = 6.0 * np.sum(differences**2) ** 1.5
    acceleration = float(np.sum(differences**3) / denominator) if denominator else 0.0

    def adjusted(alpha: float) -> float:
        z = norm.ppf(alpha)
        probability = norm.cdf(
            z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))
        )
        return float(np.clip(probability, 0.0, 1.0))

    return {
        "estimate": estimate,
        "lower": float(np.quantile(bootstrap, adjusted(alpha_low))),
        "upper": float(np.quantile(bootstrap, adjusted(alpha_high))),
        "method": "BCa",
        "patients": int(len(values)),
        "replicates": BOOTSTRAP_REPLICATES,
        "valid_replicates": BOOTSTRAP_REPLICATES,
        "failures": 0,
        "seed": int(seed),
        "bias_correction": float(z0),
        "acceleration": acceleration,
        "bootstrap_mean": float(bootstrap.mean()),
        "bootstrap_sd": float(bootstrap.std(ddof=1)),
        "bootstrap_min": float(bootstrap.min()),
        "bootstrap_max": float(bootstrap.max()),
    }


def make_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "patient_id",
        "true_label",
        "calibrated_logit",
        "calibrated_probability",
        "nll",
        "image_field",
        "focus",
        "camera",
    ]
    right = frame.loc[frame["exam_eye"] == 1, columns].set_index("patient_id")
    left = frame.loc[frame["exam_eye"] == 2, columns].set_index("patient_id")
    pair = right.join(left, lsuffix="_R", rsuffix="_L", how="inner", validate="one_to_one")
    pair["state"] = (
        pair["true_label_R"].astype(int).astype(str)
        + pair["true_label_L"].astype(int).astype(str)
    )
    pair["A"] = pair["calibrated_logit_R"] - pair["calibrated_logit_L"]
    pair["pair_evidence"] = (
        pair["calibrated_logit_R"] + pair["calibrated_logit_L"]
    ) / 2.0
    pair["oriented_margin"] = (
        pair["true_label_R"] - pair["true_label_L"]
    ) * pair["A"]
    pair["field_pattern"] = pair.apply(
        lambda row: quality_pair_pattern(row["image_field_R"], row["image_field_L"]),
        axis=1,
    )
    pair["focus_pattern"] = pair.apply(
        lambda row: quality_pair_pattern(row["focus_R"], row["focus_L"]), axis=1
    )
    pair["acquisition_pair"] = np.where(
        pair["camera_R"] == pair["camera_L"],
        np.where(pair["camera_R"].eq("Canon CR"), "Canon/Canon", "Nikon/Nikon"),
        "mixed",
    )
    return pair


def quality_pair_pattern(right: Any, left: Any) -> str:
    if right not in (1, 2) or left not in (1, 2):
        return "unevaluable"
    labels = {1: "satisfactory", 2: "abnormal"}
    return f"R_{labels[int(right)]}__L_{labels[int(left)]}"


def add_fellow_quality(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    result = frame.copy()
    fellow = frame[["patient_id", "exam_eye", variable]].copy()
    fellow["exam_eye"] = fellow["exam_eye"].map({1: 2, 2: 1})
    fellow = fellow.rename(columns={variable: f"fellow_{variable}"})
    result = result.merge(fellow, on=["patient_id", "exam_eye"], validate="one_to_one")
    own = result[variable]
    other = result[f"fellow_{variable}"]
    conditions = [
        own.eq(1) & other.eq(1),
        own.eq(2) & other.eq(1),
        own.eq(1) & other.eq(2),
        own.eq(2) & other.eq(2),
    ]
    labels = [
        "both_satisfactory",
        "own_abnormal__fellow_satisfactory",
        "own_satisfactory__fellow_abnormal",
        "both_abnormal",
    ]
    result[f"{variable}_directional_stratum"] = np.select(
        conditions, labels, default="unevaluable"
    )
    return result


def risk_summary(frame: pd.DataFrame, target: str = "nll") -> dict[str, Any]:
    patient = patient_risk_frame(frame, target)
    interval = bca_mean(patient["contribution"].to_numpy())
    risk0 = float(patient["risk_r0"].mean())
    risk1 = float(patient["risk_r1"].mean())
    return {
        "patients": int(len(patient)),
        "eyes": int(len(frame)),
        "mean_observed_loss": float(patient["observed_loss"].mean()),
        "risk_r0": risk0,
        "risk_r1": risk1,
        "delta_r": risk0 - risk1,
        "relative_delta": (risk0 - risk1) / risk0 if risk0 else None,
        "ci95_lower": interval["lower"],
        "ci95_upper": interval["upper"],
        "interval_method": interval["method"],
        "positive_contribution_fraction": float(
            (patient["contribution"] > 0).mean()
        ),
    }


def write_csv(rows: list[dict[str, Any]], name: str) -> Path:
    path = OUT / name
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Generated only from sealed R7 results.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:20px;font-weight:700}.axis{font-size:11px}.label{font-size:13px}.small{font-size:10px}.grid{stroke:#d5d8dc;stroke-width:1}.zero{stroke:#566573;stroke-width:1.5}.ci{stroke:#1f618d;stroke-width:3}.point{fill:#1f618d}.box{fill:#d6eaf8;stroke:#1f618d;stroke-width:1.5}.warn{fill:#922b21}</style>',
    ]


def svg_text(x: float, y: float, text_value: str, css: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}">{html.escape(text_value)}</text>'


def make_flow_figure() -> None:
    width, height = 940, 300
    lines = svg_start(width, height, "Frozen BRSET bilateral patient flow")
    lines.append(svg_text(470, 30, "Frozen BRSET bilateral patient flow", "title", "middle"))
    boxes = [
        (40, 90, 220, 90, "Clean bilateral cohort", "7,695 patients / 15,390 eyes"),
        (360, 60, 220, 90, "Development", "5,771 patients / 11,542 eyes"),
        (680, 60, 220, 90, "Untouched final holdout", "1,924 patients / 3,848 eyes"),
        (680, 200, 220, 60, "One-time evaluation", "3 endpoints; execution_count = 1"),
    ]
    for x, y, w, h, head, sub in boxes:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="box"/>')
        lines.append(svg_text(x + w / 2, y + 35, head, "label", "middle"))
        lines.append(svg_text(x + w / 2, y + 60, sub, "small", "middle"))
    for x1, y1, x2, y2 in ((260, 120, 360, 105), (260, 145, 680, 105), (790, 150, 790, 200)):
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1f618d" stroke-width="2" marker-end="url(#arrow)"/>')
    lines.insert(6, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#1f618d"/></marker></defs>')
    lines.append(svg_text(40, 285, "Patient-level splitting; both eyes retained together; no holdout role in selection", "small"))
    lines.append("</svg>")
    (FIG / "R8_FIG1_STUDY_FLOW.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_state_figure(state_table: pd.DataFrame, asymmetry: pd.DataFrame) -> None:
    width, height = 760, 460
    lines = svg_start(width, height, "Drusen four-state bilateral structure")
    lines.append(svg_text(380, 30, "Drusen four-state bilateral structure", "title", "middle"))
    positions = {"00": (110, 90), "01": (400, 90), "10": (110, 255), "11": (400, 255)}
    for state, (x, y) in positions.items():
        s = state_table[(state_table.endpoint == "drusen") & (state_table.state == state)].iloc[0]
        a = asymmetry[(asymmetry.endpoint == "drusen") & (asymmetry.state == state)].iloc[0]
        lines.append(f'<rect x="{x}" y="{y}" width="250" height="125" rx="8" class="box"/>')
        lines.append(svg_text(x + 15, y + 28, f"State {state}", "label"))
        lines.append(svg_text(x + 15, y + 54, f"Patients: {int(s.patients)}", "small"))
        lines.append(svg_text(x + 15, y + 76, f"Mean A (R-L): {a.mean_A:.3f}", "small"))
        lines.append(svg_text(x + 15, y + 98, f"Delta_R: {s.delta_r:.4f}", "small"))
    lines.append(svg_text(35, 125, "R true label = 0", "small", "middle"))
    lines.append(svg_text(35, 290, "R true label = 1", "small", "middle"))
    lines.append(svg_text(235, 435, "L true label = 0", "small", "middle"))
    lines.append(svg_text(525, 435, "L true label = 1", "small", "middle"))
    lines.append(svg_text(720, 415, "A = right minus left calibrated logit", "small", "end"))
    lines.append("</svg>")
    (FIG / "R8_FIG2_FOUR_STATE_STRUCTURE.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_contribution_figure(patient: pd.DataFrame) -> None:
    values = patient["contribution"].to_numpy(dtype=float)
    low, high = np.quantile(values, [0.01, 0.99])
    clipped = values[(values >= low) & (values <= high)]
    counts, edges = np.histogram(clipped, bins=36)
    width, height = 900, 480
    left, right, top, bottom = 80, 860, 65, 390
    lines = svg_start(width, height, "Drusen patient-level R0 minus R1 squared-error contribution")
    lines.append(svg_text(450, 30, "Drusen patient-level reliability advantage distribution", "title", "middle"))
    xscale = lambda x: left + (x - low) / (high - low) * (right - left)
    yscale = lambda y: bottom - y / max(counts.max(), 1) * (bottom - top)
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="zero"/>')
    if low < 0 < high:
        zx = xscale(0)
        lines.append(f'<line x1="{zx}" y1="{top}" x2="{zx}" y2="{bottom}" class="zero"/>')
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        x = xscale(lo)
        w = max(1.0, xscale(hi) - x)
        y = yscale(count)
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bottom-y:.1f}" fill="#5dade2" stroke="white"/>')
    for value in np.linspace(low, high, 5):
        x = xscale(value)
        lines.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom+5}" class="zero"/>')
        lines.append(svg_text(x, bottom + 22, f"{value:.3f}", "axis", "middle"))
    lines.append(svg_text(470, 445, "Patient contribution dₚ (positive favors R1); central 98% shown", "label", "middle"))
    lines.append(svg_text(85, 455, f"Outside plotted range: {(values < low).sum()} lower, {(values > high).sum()} upper", "small"))
    lines.append("</svg>")
    (FIG / "R8_FIG3_DRUSEN_RELIABILITY_DISTRIBUTION.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_endpoint_forest(primary: pd.DataFrame) -> None:
    width, height = 850, 360
    left, right = 260, 780
    xmin = min(-0.007, primary.ci95_lower.min() * 1.1)
    xmax = max(0.012, primary.ci95_upper.max() * 1.1)
    xscale = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    lines = svg_start(width, height, "Incremental reliability across endpoints")
    lines.append(svg_text(425, 30, "Held-out incremental fellow-eye reliability", "title", "middle"))
    for value in np.linspace(xmin, xmax, 6):
        x = xscale(value)
        lines.append(f'<line x1="{x}" y1="55" x2="{x}" y2="285" class="grid"/>')
        lines.append(svg_text(x, 310, f"{value:.3f}", "axis", "middle"))
    lines.append(f'<line x1="{xscale(0)}" y1="55" x2="{xscale(0)}" y2="285" class="zero"/>')
    for idx, endpoint in enumerate(ENDPOINTS):
        row = primary[primary.endpoint == endpoint].iloc[0]
        y = 95 + idx * 80
        lines.append(svg_text(245, y + 5, LABELS[endpoint], "label", "end"))
        lines.append(f'<line x1="{xscale(row.ci95_lower)}" y1="{y}" x2="{xscale(row.ci95_upper)}" y2="{y}" class="ci"/>')
        lines.append(f'<circle cx="{xscale(row.delta_r)}" cy="{y}" r="6" class="point"/>')
        lines.append(svg_text(790, y + 4, f"{row.delta_r:.4f}", "small"))
    lines.append(svg_text(520, 345, "Delta_R = patient-mean MSE(R0) − MSE(R1)", "label", "middle"))
    lines.append("</svg>")
    (FIG / "R8_FIG4_ENDPOINT_DELTA_FOREST.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_asymmetry_figure(asymmetry: pd.DataFrame) -> None:
    width, height = 980, 620
    panel_lefts = [80, 390, 700]
    panel_width = 240
    ymin, ymax = -4.2, 4.2
    yscale = lambda y: 525 - (y - ymin) / (ymax - ymin) * 420
    lines = svg_start(width, height, "Signed predictive asymmetry by bilateral truth state")
    lines.append(svg_text(490, 30, "Signed predictive asymmetry by four-state truth", "title", "middle"))
    for panel, endpoint in enumerate(ENDPOINTS):
        x0 = panel_lefts[panel]
        lines.append(svg_text(x0 + panel_width / 2, 65, LABELS[endpoint], "label", "middle"))
        lines.append(f'<line x1="{x0}" y1="{yscale(0)}" x2="{x0+panel_width}" y2="{yscale(0)}" class="zero"/>')
        for index, state in enumerate(("00", "11", "10", "01")):
            row = asymmetry[(asymmetry.endpoint == endpoint) & (asymmetry.state == state)].iloc[0]
            x = x0 + 30 + index * 58
            q10, q25, med, q75, q90 = [row[key] for key in ("q10_A", "q25_A", "median_A", "q75_A", "q90_A")]
            lines.append(f'<line x1="{x}" y1="{yscale(q10)}" x2="{x}" y2="{yscale(q90)}" class="ci"/>')
            ytop, ybottom = yscale(q75), yscale(q25)
            lines.append(f'<rect x="{x-12}" y="{ytop}" width="24" height="{max(1,ybottom-ytop)}" class="box"/>')
            lines.append(f'<line x1="{x-12}" y1="{yscale(med)}" x2="{x+12}" y2="{yscale(med)}" class="zero"/>')
            lines.append(svg_text(x, 550, state, "axis", "middle"))
        if panel == 0:
            for val in (-4, -2, 0, 2, 4):
                lines.append(svg_text(x0 - 12, yscale(val) + 4, str(val), "axis", "end"))
    lines.append(svg_text(490, 600, "A = right-eye calibrated logit − left-eye calibrated logit; whiskers show 10th–90th percentiles", "small", "middle"))
    lines.append("</svg>")
    (FIG / "R8_FIG5_ASYMMETRY_BY_STATE.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_quality_figure(quality: pd.DataFrame) -> None:
    data = quality[quality.endpoint.eq("drusen") & quality.interval_method.eq("BCa")].copy()
    width, height = 920, 610
    left, right = 370, 850
    xmin = min(-0.01, data.ci95_lower.min() * 1.1)
    xmax = max(0.02, data.ci95_upper.max() * 1.1)
    xscale = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    lines = svg_start(width, height, "Drusen quality-stratified incremental reliability")
    lines.append(svg_text(460, 30, "Drusen reliability by directional quality pattern", "title", "middle"))
    lines.append(f'<line x1="{xscale(0)}" y1="60" x2="{xscale(0)}" y2="535" class="zero"/>')
    labels = {
        "both_satisfactory": "both satisfactory",
        "own_abnormal__fellow_satisfactory": "own abnormal / fellow satisfactory",
        "own_satisfactory__fellow_abnormal": "own satisfactory / fellow abnormal",
        "both_abnormal": "both abnormal",
    }
    cursor = 95
    for variable in ("image_field", "focus"):
        lines.append(svg_text(25, cursor, variable, "label"))
        cursor += 25
        for stratum in labels:
            subset = data[(data.quality_dimension == variable) & (data.stratum == stratum)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            lines.append(svg_text(350, cursor + 4, f"{labels[stratum]} (n={int(row.patients)})", "small", "end"))
            lines.append(f'<line x1="{xscale(row.ci95_lower)}" y1="{cursor}" x2="{xscale(row.ci95_upper)}" y2="{cursor}" class="ci"/>')
            lines.append(f'<circle cx="{xscale(row.delta_r)}" cy="{cursor}" r="5" class="point"/>')
            cursor += 45
        cursor += 20
    for value in np.linspace(xmin, xmax, 6):
        lines.append(svg_text(xscale(value), 570, f"{value:.3f}", "axis", "middle"))
    lines.append(svg_text(610, 600, "Descriptive strata; positive Delta_R favors R1", "small", "middle"))
    lines.append("</svg>")
    (FIG / "R8_FIG6_QUALITY_STRATA_FOREST.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    ledger_rows = verify_r7_ledger()
    reported = json.loads(
        (ROOT / "r7_results/holdout_reliability_results.json").read_text(encoding="utf-8")
    )
    base = json.loads(
        (ROOT / "r7_results/holdout_base_model_metrics.json").read_text(encoding="utf-8")
    )
    fixed = json.loads(
        (ROOT / "r7_results/holdout_fixed_sequence_results.json").read_text(encoding="utf-8")
    )
    mechanism = json.loads(
        (ROOT / "r7_results/holdout_mechanistic_results.json").read_text(encoding="utf-8")
    )
    predictions = {endpoint: load_predictions(endpoint) for endpoint in ENDPOINTS}
    pairs = {endpoint: make_pair_frame(predictions[endpoint]) for endpoint in ENDPOINTS}

    reproduction_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    brier_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    state_distribution_rows: list[dict[str, Any]] = []
    state_risk_rows: list[dict[str, Any]] = []
    asymmetry_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for endpoint in ENDPOINTS:
        frame = predictions[endpoint]
        pair = pairs[endpoint]
        nll_patient = patient_risk_frame(frame, "nll")
        brier_patient = patient_risk_frame(frame, "brier")
        primary_interval = bca_mean(nll_patient["contribution"].to_numpy())
        one_sided = bca_mean(
            nll_patient["contribution"].to_numpy(), alpha_low=0.05, alpha_high=0.999999
        )
        risk0 = float(nll_patient["risk_r0"].mean())
        risk1 = float(nll_patient["risk_r1"].mean())
        delta = float(nll_patient["contribution"].mean())
        reference = reported[endpoint]["nll"]
        maximum_difference = max(
            abs(risk0 - reference["risk_r0"]),
            abs(risk1 - reference["risk_r1"]),
            abs(delta - reference["delta_r"]),
            abs(primary_interval["lower"] - reference["delta_two_sided_95_bca"]["lower"]),
            abs(primary_interval["upper"] - reference["delta_two_sided_95_bca"]["upper"]),
            abs(one_sided["lower"] - reference["delta_one_sided_95_lower_bca"]["lower"]),
        )
        reproduced = maximum_difference < 1e-14
        if not reproduced:
            raise RuntimeError(f"R8 INTEGRITY FAILURE: primary result mismatch for {endpoint}")
        reproduction_rows.append(
            {
                "endpoint": endpoint,
                "patients": len(nll_patient),
                "risk_r0": risk0,
                "risk_r1": risk1,
                "delta_r": delta,
                "ci95_lower": primary_interval["lower"],
                "ci95_upper": primary_interval["upper"],
                "one_sided_lower": one_sided["lower"],
                "maximum_absolute_difference_from_r7": maximum_difference,
                "reproduced_within_1e-14": reproduced,
            }
        )
        primary_rows.append(
            {
                "classification": "PRIMARY" if endpoint == "drusen" else "CONFIRMATORY",
                "endpoint": endpoint,
                "patients": len(nll_patient),
                "risk_r0": risk0,
                "risk_r1": risk1,
                "delta_r": delta,
                "relative_delta": delta / risk0,
                "relative_delta_percent": 100.0 * delta / risk0,
                "ci95_lower": primary_interval["lower"],
                "ci95_upper": primary_interval["upper"],
                "one_sided_lower": one_sided["lower"],
                "fixed_sequence_status": fixed[endpoint]["status"],
            }
        )
        brier_interval = bca_mean(brier_patient["contribution"].to_numpy())
        b0 = float(brier_patient["risk_r0"].mean())
        b1 = float(brier_patient["risk_r1"].mean())
        brier_rows.append(
            {
                "classification": "SECONDARY",
                "endpoint": endpoint,
                "patients": len(brier_patient),
                "risk_r0": b0,
                "risk_r1": b1,
                "delta_r": b0 - b1,
                "relative_delta": (b0 - b1) / b0,
                "relative_delta_percent": 100.0 * (b0 - b1) / b0,
                "ci95_lower": brier_interval["lower"],
                "ci95_upper": brier_interval["upper"],
            }
        )

        state_counts = pair["state"].value_counts()
        cohort_rows.append(
            {
                "endpoint": endpoint,
                "patients": int(frame["patient_id"].nunique()),
                "eyes": len(frame),
                "positive_eyes": int(frame["true_label"].sum()),
                "prevalence": float(frame["true_label"].mean()),
                "classification_errors_at_0_5": int(
                    (frame["classification_prediction"] != frame["true_label"]).sum()
                ),
                **{f"state_{state}": int(state_counts.get(state, 0)) for state in ("00", "11", "10", "01")},
            }
        )
        for state in ("00", "11", "10", "01"):
            state_patients = pair.index[pair["state"].eq(state)]
            subset = frame[frame["patient_id"].isin(state_patients)].copy()
            summary = risk_summary(subset, "nll")
            state_weight = len(state_patients) / len(pair)
            state_distribution_rows.append(
                {
                    "endpoint": endpoint,
                    "state": state,
                    "patients": int(len(state_patients)),
                    "percent_patients": 100.0 * len(state_patients) / len(pair),
                }
            )
            state_risk_rows.append(
                {
                    "classification": "SECONDARY_MECHANISTIC",
                    "endpoint": endpoint,
                    "state": state,
                    "population_weight": state_weight,
                    "weighted_delta_contribution": state_weight * summary["delta_r"],
                    **summary,
                }
            )
            values = pair.loc[pair["state"].eq(state), "A"].to_numpy(dtype=float)
            a_interval = bca_mean(values)
            oriented = pair.loc[pair["state"].eq(state), "oriented_margin"].to_numpy(dtype=float)
            asymmetry_rows.append(
                {
                    "classification": "SECONDARY_MECHANISTIC",
                    "endpoint": endpoint,
                    "state": state,
                    "patients": len(values),
                    "mean_A": float(values.mean()),
                    "sd_A": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "q10_A": float(np.quantile(values, 0.10)),
                    "q25_A": float(np.quantile(values, 0.25)),
                    "median_A": float(np.median(values)),
                    "q75_A": float(np.quantile(values, 0.75)),
                    "q90_A": float(np.quantile(values, 0.90)),
                    "mean_A_ci95_lower": a_interval["lower"],
                    "mean_A_ci95_upper": a_interval["upper"],
                    "mean_oriented_margin": float(oriented.mean()) if state in ("10", "01") else None,
                    "median_oriented_margin": float(np.median(oriented)) if state in ("10", "01") else None,
                    "fraction_oriented_margin_positive": float((oriented > 0).mean()) if state in ("10", "01") else None,
                }
            )

        for variable in ("image_field", "focus"):
            quality_frame = add_fellow_quality(frame, variable)
            stratum_col = f"{variable}_directional_stratum"
            for stratum in (
                "both_satisfactory",
                "own_abnormal__fellow_satisfactory",
                "own_satisfactory__fellow_abnormal",
                "both_abnormal",
                "unevaluable",
            ):
                subset = quality_frame[quality_frame[stratum_col].eq(stratum)]
                if subset.empty:
                    continue
                summary = risk_summary(subset, "nll")
                quality_rows.append(
                    {
                        "classification": "SECONDARY_MECHANISTIC_DESCRIPTIVE",
                        "endpoint": endpoint,
                        "quality_dimension": variable,
                        "stratum": stratum,
                        **summary,
                    }
                )

        acquisition_map = pair["acquisition_pair"]
        acquisition_frame = frame.merge(
            acquisition_map.rename("acquisition_pair"),
            left_on="patient_id",
            right_index=True,
            validate="many_to_one",
        )
        for category in ("Canon/Canon", "Nikon/Nikon", "mixed"):
            subset = acquisition_frame[acquisition_frame["acquisition_pair"].eq(category)]
            summary = risk_summary(subset, "nll")
            acquisition_rows.append(
                {
                    "classification": "SECONDARY_SENSITIVITY",
                    "endpoint": endpoint,
                    "acquisition_pair": category,
                    "precision_flag": "very_sparse_descriptive" if summary["patients"] < 10 else "estimable",
                    **summary,
                }
            )

        point = base[endpoint]["point"]
        base_rows.append(
            {
                "classification": "SECONDARY_CONTEXT",
                "endpoint": endpoint,
                "patients": point["patients"],
                "eyes": point["eyes"],
                "positive_eyes": point["positive_eyes"],
                "prevalence": point["prevalence"],
                "auroc": point["auroc"],
                "auprc": point["auprc"],
                "nll": point["nll"],
                "nll_ci95_lower": base[endpoint]["nll_interval"]["lower"],
                "nll_ci95_upper": base[endpoint]["nll_interval"]["upper"],
                "brier": point["brier"],
                "brier_ci95_lower": base[endpoint]["brier_interval"]["lower"],
                "brier_ci95_upper": base[endpoint]["brier_interval"]["upper"],
                "calibration_intercept": point["calibration"]["intercept"],
                "calibration_slope": point["calibration"]["slope"],
                "ece": point["ece"]["ece"] if point["ece"].get("estimable") else None,
                "primary_clip_count": point["primary_clip_count"],
            }
        )
        rel = reported[endpoint]
        err = rel["error_detection"]
        secondary_rows.append(
            {
                "classification": "SECONDARY",
                "endpoint": endpoint,
                "mae_delta": rel["nll"]["mae_improvement"]["estimate"],
                "mae_ci95_lower": rel["nll"]["mae_improvement"]["lower"],
                "mae_ci95_upper": rel["nll"]["mae_improvement"]["upper"],
                "spearman_r0": rel["nll"]["spearman"]["r0"],
                "spearman_r1": rel["nll"]["spearman"]["r1"],
                "spearman_difference_r1_minus_r0": rel["nll"]["spearman"]["difference"],
                "classification_errors": err["classification_errors"],
                "error_detection_auprc_r1": err["error_detection_auprc"],
                "error_detection_auroc_r1": err["error_detection_auroc"],
                "aurc_r1": err["aurc"],
                "selective_risk_90pct": err["selective_risk"]["0.9"]["selective_risk"],
                "selective_risk_80pct": err["selective_risk"]["0.8"]["selective_risk"],
                "selective_risk_70pct": err["selective_risk"]["0.7"]["selective_risk"],
            }
        )

        interval_variants = [rel["nll"]["delta_two_sided_95_bca"], *rel["nll"]["bootstrap_seed_checks"]]
        for interval in interval_variants:
            bootstrap_rows.append(
                {
                    "endpoint": endpoint,
                    "seed": interval["seed"],
                    "patients": interval["n"],
                    "requested_replicates": interval["replicates"],
                    "valid_replicates": interval["replicates"],
                    "failures": 0,
                    "estimate": interval["estimate"],
                    "ci95_lower": interval["lower"],
                    "ci95_upper": interval["upper"],
                    "acceleration": interval["acceleration"],
                    "bias_correction": interval["bias_correction"],
                }
            )

        values = nll_patient["contribution"].to_numpy(dtype=float)
        loo = (values.sum() - values) / (len(values) - 1)
        order_abs = np.argsort(np.abs(values))[::-1]
        net_sum = values.sum()
        abs_sum = np.abs(values).sum()
        observed_order = np.argsort(nll_patient["observed_loss"].to_numpy())[::-1]
        top_one_percent_n = max(1, math.ceil(0.01 * len(values)))
        influence_rows.append(
            {
                "classification": "EXPLORATORY_DIAGNOSTIC",
                "endpoint": endpoint,
                "patients": len(values),
                "delta_r": float(values.mean()),
                "median_patient_contribution": float(np.median(values)),
                "q05_patient_contribution": float(np.quantile(values, 0.05)),
                "q95_patient_contribution": float(np.quantile(values, 0.95)),
                "minimum_patient_contribution": float(values.min()),
                "maximum_patient_contribution": float(values.max()),
                "positive_fraction": float((values > 0).mean()),
                "maximum_absolute_leave_one_out_shift": float(np.max(np.abs(loo - values.mean()))),
                "delta_without_most_absolute_influential_patient": float(loo[order_abs[0]]),
                "top_1pct_absolute_contribution_share": float(np.abs(values[order_abs[:top_one_percent_n]]).sum() / abs_sum),
                "top_1pct_highest_loss_net_share": float(values[observed_order[:top_one_percent_n]].sum() / net_sum) if net_sum else None,
                "single_largest_positive_net_share": float(values.max() / net_sum) if net_sum else None,
                "single_largest_negative_net_share": float(values.min() / net_sum) if net_sum else None,
            }
        )

        observed = frame["nll"].to_numpy(dtype=float)
        for model in ("r0", "r1"):
            predicted = frame[f"predicted_nll_{model}"].to_numpy(dtype=float)
            residual = observed - predicted
            residual_rows.append(
                {
                    "classification": "EXPLORATORY_DIAGNOSTIC",
                    "endpoint": endpoint,
                    "model": model.upper(),
                    "eyes": len(frame),
                    "mean_residual": float(residual.mean()),
                    "sd_residual": float(residual.std(ddof=1)),
                    "mae": float(np.abs(residual).mean()),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "q01_residual": float(np.quantile(residual, 0.01)),
                    "q50_residual": float(np.quantile(residual, 0.50)),
                    "q95_residual": float(np.quantile(residual, 0.95)),
                    "q99_residual": float(np.quantile(residual, 0.99)),
                    "maximum_absolute_residual": float(np.max(np.abs(residual))),
                    "correlation_abs_residual_with_observed_nll": float(np.corrcoef(np.abs(residual), observed)[0, 1]),
                }
            )

        quantiles = nll_patient["observed_loss"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
        bands = [
            ("0-50%", -np.inf, quantiles[0.5]),
            ("50-90%", quantiles[0.5], quantiles[0.9]),
            ("90-95%", quantiles[0.9], quantiles[0.95]),
            ("95-99%", quantiles[0.95], quantiles[0.99]),
            ("99-100%", quantiles[0.99], np.inf),
        ]
        for label, low, high in bands:
            mask = nll_patient["observed_loss"].gt(low) & nll_patient["observed_loss"].le(high)
            subset = nll_patient.loc[mask]
            tail_rows.append(
                {
                    "classification": "EXPLORATORY_DIAGNOSTIC",
                    "endpoint": endpoint,
                    "patient_mean_nll_band": label,
                    "patients": len(subset),
                    "nll_lower_exclusive": None if not np.isfinite(low) else low,
                    "nll_upper_inclusive": None if not np.isfinite(high) else high,
                    "mean_observed_nll": float(subset["observed_loss"].mean()),
                    "risk_r0": float(subset["risk_r0"].mean()),
                    "risk_r1": float(subset["risk_r1"].mean()),
                    "delta_r": float(subset["contribution"].mean()),
                    "net_contribution_share": float(subset["contribution"].sum() / net_sum) if net_sum else None,
                }
            )

        patient_with_state = nll_patient.join(pair[["state"]])
        selections = {
            "highest_patient_mean_nll": patient_with_state["observed_loss"].nlargest(10).index,
            "largest_r0_to_r1_improvement": patient_with_state["contribution"].nlargest(10).index,
            "largest_r1_deterioration": patient_with_state["contribution"].nsmallest(10).index,
        }
        for rule, patient_ids in selections.items():
            for rank, patient_id in enumerate(patient_ids, start=1):
                row = patient_with_state.loc[patient_id]
                case_rows.append(
                    {
                        "classification": "EXPLORATORY_CASE_AUDIT",
                        "endpoint": endpoint,
                        "selection_rule": rule,
                        "rank": rank,
                        "patient_key_sha256": hashlib.sha256(f"R8_CASE_AUDIT|{patient_id}".encode()).hexdigest(),
                        "state": row["state"],
                        "patient_mean_nll": row["observed_loss"],
                        "risk_r0": row["risk_r0"],
                        "risk_r1": row["risk_r1"],
                        "contribution": row["contribution"],
                    }
                )

    # Quality distribution is endpoint-invariant because the same paired cohort is used.
    quality_distribution_rows: list[dict[str, Any]] = []
    drusen_frame = predictions["drusen"]
    drusen_pair = pairs["drusen"]
    for variable, pair_column in (("image_field", "field_pattern"), ("focus", "focus_pattern")):
        for pattern, count in drusen_pair[pair_column].value_counts(dropna=False).items():
            quality_distribution_rows.append(
                {
                    "quality_dimension": variable,
                    "pair_pattern": pattern,
                    "patients": int(count),
                    "percent_patients": 100.0 * count / len(drusen_pair),
                    "abnormal_eyes": int((drusen_frame[variable] == 2).sum()),
                    "unevaluable_eyes": int((~drusen_frame[variable].isin([1, 2])).sum()),
                }
            )

    # Frozen mechanistic H2/H3 family for the primary endpoint, with Holm adjustment.
    gee = pd.read_csv(ROOT / "r7_results/holdout_reliability_gee.csv")
    drusen_gee = gee[gee["endpoint"].eq("drusen")]
    hypothesis_raw = [
        ("H2 four-state heterogeneity", mechanism["drusen"]["gee_state_omnibus"]["state_omnibus_p_value"]),
        ("H3a image_field modification", float(drusen_gee.loc[drusen_gee.term.eq("field_abnormal"), "p_value"].iloc[0])),
        ("H3b focus modification", float(drusen_gee.loc[drusen_gee.term.eq("focus_abnormal"), "p_value"].iloc[0])),
    ]
    order = np.argsort([p for _, p in hypothesis_raw])
    adjusted = np.empty(3)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (3 - rank) * hypothesis_raw[index][1])
        running = max(running, candidate)
        adjusted[index] = running
    mechanism_hypothesis_rows = [
        {
            "classification": "SECONDARY_MECHANISTIC_FROZEN_FAMILY",
            "hypothesis": name,
            "raw_p_value": p_value,
            "holm_adjusted_p_value": adjusted[index],
            "supported_at_two_sided_0_05": bool(adjusted[index] < 0.05),
        }
        for index, (name, p_value) in enumerate(hypothesis_raw)
    ]

    gee_audit_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        frame = predictions[endpoint]
        invalid_focus = int((~frame["focus"].isin([1, 2])).sum())
        term = gee[(gee.endpoint == endpoint) & (gee.term == "focus_unevaluable")].iloc[0]
        gee_audit_rows.append(
            {
                "endpoint": endpoint,
                "term": "focus_unevaluable",
                "unevaluable_focus_eyes": invalid_focus,
                "design_column_constant_zero": invalid_focus == 0,
                "coefficient_estimable": False if invalid_focus == 0 else bool(np.isfinite(term.robust_standard_error)),
                "reported_robust_standard_error": term.robust_standard_error,
                "primary_bootstrap_affected": False,
                "confirmatory_conclusion_affected": False,
            }
        )

    # Write machine-readable tables.
    write_csv(reproduction_rows, "R8_REPRODUCED_PRIMARY_RESULTS.csv")
    primary_path = write_csv(primary_rows, "R8_PRIMARY_RELIABILITY_TABLE.csv")
    write_csv(brier_rows, "R8_BRIER_SENSITIVITY.csv")
    write_csv(cohort_rows, "R8_COHORT_TABLE.csv")
    write_csv(state_distribution_rows, "R8_BILATERAL_STATE_DISTRIBUTION.csv")
    state_path = write_csv(state_risk_rows, "R8_STATE_RELIABILITY_TABLE.csv")
    asymmetry_path = write_csv(asymmetry_rows, "R8_DIRECTIONAL_ASYMMETRY_TABLE.csv")
    quality_path = write_csv(quality_rows, "R8_QUALITY_STRATIFIED_RELIABILITY.csv")
    write_csv(quality_distribution_rows, "R8_IMAGE_QUALITY_TABLE.csv")
    write_csv(acquisition_rows, "R8_ACQUISITION_STRATA_SENSITIVITY.csv")
    write_csv(base_rows, "R8_BASE_MODEL_PERFORMANCE.csv")
    write_csv(secondary_rows, "R8_SECONDARY_METRICS.csv")
    write_csv(bootstrap_rows, "R8_BOOTSTRAP_DIAGNOSTICS.csv")
    write_csv(influence_rows, "R8_INFLUENCE_DIAGNOSTICS.csv")
    write_csv(residual_rows, "R8_RESIDUAL_DIAGNOSTICS.csv")
    write_csv(tail_rows, "R8_LOSS_TAIL_DIAGNOSTICS.csv")
    write_csv(case_rows, "R8_OBJECTIVE_CASE_AUDIT.csv")
    write_csv(mechanism_hypothesis_rows, "R8_MECHANISTIC_HYPOTHESIS_TESTS.csv")
    write_csv(gee_audit_rows, "R8_GEE_SENSITIVITY_AUDIT.csv")
    write_csv(ledger_rows, "R8_R7_HASH_VERIFICATION.csv")

    primary_df = pd.read_csv(primary_path)
    state_df = pd.read_csv(state_path, dtype={"state": str})
    asymmetry_df = pd.read_csv(asymmetry_path, dtype={"state": str})
    quality_df = pd.read_csv(quality_path)
    make_flow_figure()
    make_state_figure(state_df, asymmetry_df)
    make_contribution_figure(patient_risk_frame(predictions["drusen"], "nll"))
    make_endpoint_forest(primary_df)
    make_asymmetry_figure(asymmetry_df)
    make_quality_figure(quality_df)

    input_hashes = {
        f"r7_artifacts/final_holdout/{endpoint}_holdout_predictions.csv": sha256_file(
            ROOT / f"r7_artifacts/final_holdout/{endpoint}_holdout_predictions.csv"
        )
        for endpoint in ENDPOINTS
    }
    input_hashes["r7_artifacts/R7_SHA256SUMS.txt"] = sha256_file(
        ROOT / "r7_artifacts/R7_SHA256SUMS.txt"
    )
    input_hashes["r7_artifacts/R7_RUN_MANIFEST.json"] = sha256_file(
        ROOT / "r7_artifacts/R7_RUN_MANIFEST.json"
    )
    output_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name not in {"R8_ANALYSIS_MANIFEST.json", "R8_SHA256SUMS.txt"}
    }
    summary = {
        "status": "R8_NUMERICAL_ANALYSIS_COMPLETE",
        "r7_ledger_entries_verified": len(ledger_rows),
        "r7_hash_failures": 0,
        "primary_results_reproduced": all(row["reproduced_within_1e-14"] for row in reproduction_rows),
        "maximum_primary_reproduction_difference": max(row["maximum_absolute_difference_from_r7"] for row in reproduction_rows),
        "patients": 1924,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "input_hashes": input_hashes,
        "output_hashes_before_manifest": output_hashes,
        "prohibited_actions": {
            "model_retraining": False,
            "embedding_regeneration": False,
            "holdout_prediction_regeneration": False,
            "new_architecture": False,
            "estimand_change": False,
        },
    }
    (OUT / "R8_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [
        path
        for path in sorted(OUT.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "R8_SHA256SUMS.txt"
    ]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT)}\n" for path in checksum_paths
    )
    (OUT / "R8_SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
