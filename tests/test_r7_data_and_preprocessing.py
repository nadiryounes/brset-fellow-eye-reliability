from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.r7_config import load_config
from src.r7_data import load_stage_dataframe, state_counts
from src.r7_preprocessing import letterbox_rgb, preprocess_image


class DataAndPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_development_pairs_and_laterality(self) -> None:
        project_root = Path(self.config["_project_root"])
        manifest = project_root / self.config["partition"]["manifest_path"]
        metadata = project_root / self.config["source"]["metadata_path"]
        if not manifest.is_file() or not metadata.is_file():
            self.skipTest(
                "credentialed BRSET metadata and the private patient partition "
                "are intentionally absent from the public repository"
            )
        frame = load_stage_dataframe(
            self.config, "development", purpose="development_modeling"
        )
        self.assertEqual(len(frame), 11542)
        per_patient = frame.groupby("patient_id")["exam_eye"].agg(set)
        self.assertTrue(all(value == {1, 2} for value in per_patient))
        self.assertTrue((frame.loc[frame.exam_eye == 1, "laterality"] == "R").all())
        self.assertTrue((frame.loc[frame.exam_eye == 2, "laterality"] == "L").all())
        self.assertEqual(
            state_counts(frame, "drusen"),
            {"00": 4501, "11": 738, "10": 272, "01": 260},
        )

    def test_letterbox_preserves_horizontal_orientation(self) -> None:
        array = np.zeros((2, 4, 3), dtype=np.uint8)
        array[:, :2, 0] = 255
        array[:, 2:, 2] = 255
        image = Image.fromarray(array, mode="RGB")
        result = np.asarray(letterbox_rgb(image, size=8))
        center = result[2:6]
        self.assertGreater(center[:, :4, 0].mean(), center[:, :4, 2].mean())
        self.assertGreater(center[:, 4:, 2].mean(), center[:, 4:, 0].mean())
        self.assertEqual(result[0].max(), 0)
        self.assertEqual(result[-1].max(), 0)

    def test_preprocessing_shape_and_determinism(self) -> None:
        array = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"
            Image.fromarray(array, mode="RGB").save(path)
            first = preprocess_image(path, self.config)
            second = preprocess_image(path, self.config)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(np.array_equal(first.numpy(), second.numpy()))


if __name__ == "__main__":
    unittest.main()
