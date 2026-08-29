from __future__ import annotations

import unittest

import numpy as np

from src.r7_inference import bca_mean_interval


class InferenceTests(unittest.TestCase):
    def test_bca_is_deterministic_and_contains_mean_for_symmetric_values(self) -> None:
        values = np.arange(1.0, 11.0)
        first = bca_mean_interval(values, replicates=1000, seed=20261001)
        second = bca_mean_interval(values, replicates=1000, seed=20261001)
        self.assertEqual(first, second)
        self.assertLess(first["lower"], values.mean())
        self.assertGreater(first["upper"], values.mean())


if __name__ == "__main__":
    unittest.main()
