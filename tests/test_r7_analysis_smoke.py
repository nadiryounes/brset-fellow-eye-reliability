from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.r7_analysis import (
    build_pair_table,
    mechanistic_ols,
    mechanistic_state_summary,
    reliability_gee,
)
from src.r7_calibration import binary_nll, probability_logit
from src.r7_config import load_config


class AnalysisSmokeTest(unittest.TestCase):
    def test_mechanistic_and_gee_models_fit_synthetic_pairs(self) -> None:
        config = load_config()
        rng = np.random.default_rng(99)
        states = ["00", "11", "10", "01"] * 15
        rows = []
        for patient_index, state in enumerate(states):
            for eye, label in ((1, int(state[0])), (2, int(state[1]))):
                raw = (-1.0 + 2.0 * label) + rng.normal(scale=0.8)
                probability = 1.0 / (1.0 + np.exp(-raw))
                nll = binary_nll(np.array([label]), np.array([probability]), 1e-6)[0]
                rows.append(
                    {
                        "patient_id": str(patient_index + 1),
                        "image_id": f"x{patient_index}_{eye}",
                        "exam_eye": eye,
                        "camera": (
                            "Canon CR"
                            if (patient_index + eye) % 3
                            else "NIKON NF5050"
                        ),
                        "image_field": 2 if (patient_index + eye) % 7 == 0 else 1,
                        "focus": (
                            0
                            if patient_index == 0 and eye == 1
                            else 2 if (patient_index + eye) % 9 == 0 else 1
                        ),
                        "true_label": label,
                        "calibrated_probability": probability,
                        "calibrated_logit": probability_logit(
                            np.array([probability]), 1e-6
                        )[0],
                        "nll": nll,
                        "brier": (probability - label) ** 2,
                        "predicted_nll_r0": max(0.0, nll + rng.normal(scale=0.2)),
                        "predicted_nll_r1": max(0.0, nll + rng.normal(scale=0.15)),
                    }
                )
        frame = pd.DataFrame(rows)
        pair = build_pair_table(frame, "drusen")
        summary = mechanistic_state_summary(pair, config)
        ols = mechanistic_ols(pair, config)
        gee, omnibus = reliability_gee(frame, pair)
        self.assertEqual({item["state"] for item in summary}, {"00", "11", "10", "01"})
        self.assertIn("term", ols.columns)
        self.assertIn("term", gee.columns)
        self.assertIn("state_omnibus_p_value", omnibus)


if __name__ == "__main__":
    unittest.main()
