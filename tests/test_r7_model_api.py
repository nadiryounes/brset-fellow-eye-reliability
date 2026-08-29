from __future__ import annotations

import unittest

import torch

from src.r7_config import load_config
from src.r7_embeddings import build_frozen_encoder


class FrozenModelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.encoder, cls.provenance = build_frozen_encoder(cls.config)

    def test_feature_layer_and_dimension(self) -> None:
        with torch.inference_mode():
            feature = self.encoder(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(feature.shape), (2, 768))
        self.assertEqual(self.provenance["feature_dimension"], 768)
        self.assertEqual(
            self.provenance["weight_enum"],
            "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        )

    def test_encoder_is_frozen_eval_and_deterministic(self) -> None:
        self.assertFalse(self.encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.encoder.parameters()))
        value = torch.linspace(0, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        with torch.inference_mode():
            first = self.encoder(value)
            second = self.encoder(value)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
