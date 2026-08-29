from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.r7_config import load_config
from src.r7_cv import make_outer_patient_folds
from src.r7_development import run_nested_endpoint


class SyntheticDevelopmentSmokeTest(unittest.TestCase):
    def test_nested_pipeline_is_complete_and_oof(self) -> None:
        config = load_config()
        states = ["00", "11", "10", "01"] * 10
        records = []
        rng = np.random.default_rng(1234)
        features = []
        for patient_index, state in enumerate(states):
            for eye_index, label in ((1, int(state[0])), (2, int(state[1]))):
                records.append(
                    {
                        "patient_id": str(patient_index + 1),
                        "image_id": f"synthetic_{patient_index + 1}_{eye_index}",
                        "exam_eye": eye_index,
                        "laterality": "R" if eye_index == 1 else "L",
                        "camera": "Canon CR" if patient_index % 2 == 0 else "NIKON NF5050",
                        "image_field": 1 if patient_index % 5 else 2,
                        "focus": 1 if patient_index % 7 else 2,
                        "drusen": label,
                        "increased_cup_disc": label,
                        "diabetic_retinopathy": label,
                    }
                )
                vector = rng.normal(size=12)
                vector[0] += 0.75 * label
                features.append(vector)
        metadata = pd.DataFrame(records)
        matrix = np.asarray(features, dtype=np.float64)
        outer = make_outer_patient_folds(metadata, config)
        nested, selections = run_nested_endpoint(
            "drusen", metadata, matrix, outer, config
        )
        self.assertEqual(len(nested), len(metadata))
        self.assertFalse(nested["image_id"].duplicated().any())
        self.assertEqual(nested["outer_fold"].nunique(), 5)
        self.assertEqual(len(selections), 5)
        self.assertFalse(any("state" in column for column in nested.columns))
        self.assertTrue(nested["calibrated_probability"].between(0, 1).all())
        self.assertTrue(
            {
                "predicted_nll_r0",
                "predicted_nll_r1",
                "predicted_brier_r0",
                "predicted_brier_r1",
            }.issubset(nested.columns)
        )


if __name__ == "__main__":
    unittest.main()
