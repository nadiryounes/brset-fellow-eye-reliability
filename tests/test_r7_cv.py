from __future__ import annotations

import unittest
from pathlib import Path

from src.r7_config import load_config
from src.r7_cv import make_inner_patient_folds, make_outer_patient_folds
from src.r7_data import load_stage_dataframe


class FrozenFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        project_root = Path(cls.config["_project_root"])
        manifest = project_root / cls.config["partition"]["manifest_path"]
        metadata = project_root / cls.config["source"]["metadata_path"]
        if not manifest.is_file() or not metadata.is_file():
            raise unittest.SkipTest(
                "credentialed BRSET metadata and the private patient partition "
                "are intentionally absent from the public repository"
            )
        cls.metadata = load_stage_dataframe(
            cls.config, "development", purpose="development_modeling"
        )

    def test_outer_folds_are_patient_level_and_complete(self) -> None:
        folds = make_outer_patient_folds(self.metadata, self.config)
        self.assertEqual(len(folds), 5771)
        self.assertFalse(folds["patient_id"].duplicated().any())
        self.assertEqual(folds["outer_fold"].nunique(), 5)
        for _, group in folds.groupby("outer_fold"):
            self.assertEqual(set(group["drusen_state"]), {"00", "11", "10", "01"})

    def test_inner_folds_exclude_outer_assessment(self) -> None:
        outer = make_outer_patient_folds(self.metadata, self.config)
        outer_assessment = set(outer.loc[outer.outer_fold == 0, "patient_id"])
        training_metadata = self.metadata[
            ~self.metadata.patient_id.isin(outer_assessment)
        ]
        inner = make_inner_patient_folds(training_metadata, 0, self.config)
        self.assertFalse(set(inner["patient_id"]) & outer_assessment)
        self.assertEqual(inner["inner_fold"].nunique(), 3)

    def test_folds_are_deterministic(self) -> None:
        first = make_outer_patient_folds(self.metadata, self.config)
        second = make_outer_patient_folds(self.metadata, self.config)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()
