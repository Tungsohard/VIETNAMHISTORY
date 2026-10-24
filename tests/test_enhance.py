import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from colorization.enhance import modernize_ab, refine_ab, restore_luminance


class EnhanceTests(unittest.TestCase):
    def test_refine_ab_stops_colour_at_luminance_edge(self):
        # Dark person (left) next to bright sky (right); the low-res prediction blurs across the edge.
        luminance = np.full((240, 320), 80.0, np.float32)
        luminance[:, :160] = 35.0
        ab_low = np.zeros((32, 32, 2), np.float32)
        ab_low[:, :16] = (20.0, 15.0)
        ab_low[:, 16:] = (-5.0, -30.0)
        ab_low = cv2.GaussianBlur(ab_low, (0, 0), 2.0)
        refined = refine_ab(luminance, ab_low)
        naive = cv2.resize(ab_low, (320, 240), interpolation=cv2.INTER_CUBIC)
        # Six pixels either side of the edge: refined colour must be much closer to its own region.
        for column, expected in ((153, (20.0, 15.0)), (166, (-5.0, -30.0))):
            refined_error = np.abs(refined[:, column] - expected).mean()
            naive_error = np.abs(naive[:, column] - expected).mean()
            self.assertLess(refined_error, 0.5 * naive_error)

    def test_modernize_removes_uniform_cast(self):
        luminance = np.full((64, 64), 50.0, np.float32)
        ab = np.zeros((64, 64, 2), np.float32)
        ab[:, :32] = (0.0, 25.0)          # genuinely colourful half
        ab += (3.0, 8.0)                  # sepia cast on everything
        result = modernize_ab(luminance, ab, vividness=0.6)
        neutral = result[:, 40:].reshape(-1, 2).mean(axis=0)
        # The cast (|8.5|) is cut to the 3-unit tolerance on the grey half...
        self.assertLess(np.hypot(*neutral), 3.5)
        # ...while the colourful half is not desaturated by the correction.
        colourful = np.hypot(*result[:, :16].reshape(-1, 2).mean(axis=0))
        self.assertGreaterEqual(colourful, np.hypot(3.0, 33.0))

    def test_modernize_never_pushes_grey_past_neutral(self):
        # Warm cast on most of the image, but white hair / deep shadow is already grey.
        luminance = np.full((64, 64), 50.0, np.float32)
        ab = np.tile(np.array([3.0, 9.0], np.float32), (64, 64, 1))
        ab[:, :20] = (0.0, 25.0)
        ab[:, 56:] = 0.0
        result = modernize_ab(luminance, ab, vividness=1.0)
        grey = result[:, 56:].reshape(-1, 2)
        self.assertLess(np.abs(grey).max(), 0.5)   # not tinted blue

    def test_uncovered_areas_become_separate_objects(self):
        from colorization.segmentation import _fill_uncovered
        labels = np.full((40, 40), -1, np.int32)
        labels[:, 18:22] = 0                      # one object splits the uncovered area in two
        filled = _fill_uncovered(labels.copy(), min_fraction=0.0)
        self.assertGreaterEqual(filled.min(), 0)
        self.assertNotEqual(filled[0, 0], filled[0, 39])   # left and right stay distinct
        self.assertEqual(filled[0, 20], 0)

    def test_refine_ab_never_borrows_across_object_border(self):
        # Two objects with the same brightness: luminance alone cannot separate them.
        luminance = np.full((120, 160), 60.0, np.float32)
        objects = np.zeros((120, 160), np.int32)
        objects[:, 80:] = 1
        ab_low = np.zeros((16, 16, 2), np.float32)
        ab_low[:, :8] = (25.0, 20.0)
        ab_low[:, 8:] = (-5.0, -30.0)
        ab_low = cv2.GaussianBlur(ab_low, (0, 0), 1.5)
        without = refine_ab(luminance, ab_low)
        with_objects = refine_ab(luminance, ab_low, objects=objects)
        target = np.array([25.0, 20.0])
        self.assertLess(np.abs(with_objects[:, 74] - target).mean(),
                        0.6 * np.abs(without[:, 74] - target).mean())

    def test_restore_luminance_reduces_noise_and_keeps_range(self):
        rng = np.random.default_rng(0)
        clean = np.tile(np.linspace(20, 80, 256, dtype=np.float32), (256, 1))
        noisy = np.clip(clean + rng.normal(0, 4, clean.shape), 0, 100).astype(np.float32)
        restored = restore_luminance(noisy)
        self.assertEqual(restored.dtype, np.float32)
        self.assertGreaterEqual(restored.min(), 0.0)
        self.assertLessEqual(restored.max(), 100.0)
        high_pass = lambda x: np.abs(x - cv2.GaussianBlur(x, (0, 0), 3)).mean()
        self.assertLess(high_pass(restored), high_pass(noisy))


if __name__ == "__main__":
    unittest.main()


class VicowCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare"))

    def test_correction_is_mild_and_keeps_layout(self):
        from correct_vicow_colors import correct_colors
        rng = np.random.default_rng(0)
        base = np.full((90, 160, 3), (110, 135, 90), np.uint8)       # green-ish frame
        base[:, 80:] = (200, 190, 170)                               # warm wall on the right
        base = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        out = correct_colors(base)
        self.assertEqual(out.shape, base.shape)
        lab_in = cv2.cvtColor(base.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        # Mild: colours move by under ~2 Lab units on average and ~10 at the 99th percentile.
        change = np.abs(lab_out[..., 1:] - lab_in[..., 1:])
        self.assertLess(change.mean(), 2.5)
        self.assertLess(np.percentile(change, 99), 10.0)
        self.assertLess(abs(lab_out[:, 40, 1] - lab_in[:, 40, 1]).mean(), 6.0)
        self.assertGreater(lab_out[:, 120, 2].mean(), lab_out[:, 40, 2].mean() - 15.0)

    def test_excluded_source_does_not_change_selection_score(self):
        from fine_tune_vietnam import add_source_balanced_score
        metrics = {"val_ktvic_selection_score": 0.8, "val_viet_cultural_vqa_selection_score": 0.9,
                   "val_vicow_val_selection_score": 2.0}
        with_all = dict(metrics); add_source_balanced_score(with_all)
        without = dict(metrics); add_source_balanced_score(without, ("vicow_val",))
        self.assertAlmostEqual(without["val_source_balanced_selection_score"], 0.85)
        self.assertGreater(with_all["val_source_balanced_selection_score"], 1.2)

