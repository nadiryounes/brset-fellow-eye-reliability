from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.r7_calibration import binary_nll, fit_calibrator
from src.r7_config import load_config
from src.r7_reliability import ReliabilityRegressor


class CalibrationAndReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_registered_calibrators_are_monotone(self) -> None:
        logits = np.linspace(-3, 3, 60)
        labels = (logits + np.sin(logits) > 0).astype(int)
        for method in ("temperature", "platt"):
            fitted = fit_calibrator(method, logits, labels, self.config)
            probability = fitted.predict(logits)
            self.assertTrue(np.all(np.diff(probability) >= 0))
            self.assertGreater(probability[-1], probability[0])
            self.assertGreater(fitted.slope, 0)

    def test_nll_clipping_is_finite(self) -> None:
        values = binary_nll(
            np.array([1, 0]), np.array([0.0, 1.0]), 1e-6
        )
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values.max(), 13.815510557964274)

    def test_reliability_schema_adds_only_fellow_spline(self) -> None:
        count = 40
        frame = pd.DataFrame(
            {
                "patient_id": np.repeat(np.arange(count // 2), 2).astype(str),
                "exam_eye": np.tile([1, 2], count // 2),
                "camera": np.tile(["Canon CR", "NIKON NF5050"], count // 2),
                "image_field": np.tile([1, 2], count // 2),
                "focus": np.tile([1, 2], count // 2),
                "calibrated_logit": np.linspace(-2, 2, count),
                "fellow_calibrated_logit": np.linspace(2, -2, count),
            }
        )
        target = np.linspace(0.01, 2.0, count)
        baseline = ReliabilityRegressor(self.config, 0.1, False).fit(frame, target)
        augmented = ReliabilityRegressor(self.config, 0.1, True).fit(frame, target)
        base_columns = baseline.schema()["design_columns"]
        augmented_columns = augmented.schema()["design_columns"]
        added = [column for column in augmented_columns if column not in base_columns]
        self.assertTrue(added)
        self.assertTrue(all(column.startswith("fellow_logit_spline_") for column in added))
        self.assertFalse(any("state" in column for column in augmented_columns))


if __name__ == "__main__":
    unittest.main()
