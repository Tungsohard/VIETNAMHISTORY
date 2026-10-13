import unittest

import numpy as np
import torch

from colorization.inference import _lab_prediction_to_rgb, _normalized_luminance


class InferenceTests(unittest.TestCase):
    def test_luminance_preparation_uses_model_resolution(self):
        image = np.full((40, 80, 3), 128, dtype=np.uint8)
        luminance, tensor = _normalized_luminance(image, 256)
        self.assertEqual(luminance.shape, (40, 80))
        self.assertEqual(tuple(tensor.shape), (1, 256, 256))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_neutral_prediction_preserves_original_size(self):
        image = np.full((37, 83, 3), [40, 120, 210], dtype=np.uint8)
        luminance, _ = _normalized_luminance(image, 256)
        result = _lab_prediction_to_rgb(luminance, torch.zeros(2, 256, 256))
        self.assertEqual(result.shape, image.shape)
        channel_spread = result.max(axis=2).astype(int) - result.min(axis=2).astype(int)
        self.assertLessEqual(int(channel_spread.max()), 1)


if __name__ == "__main__":
    unittest.main()
