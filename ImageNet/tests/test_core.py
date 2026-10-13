import unittest

import cv2
import numpy as np
import torch

from colorization.color import lab_to_rgb, rgb_to_tensors
from colorization.ddcolor_adapter import neutral_lab_l_to_srgb
from fine_tune_vietnam import add_source_balanced_score, chroma_aware_loss, chroma_selection_score


class CoreTests(unittest.TestCase):
    def test_lab_round_trip(self):
        image = np.full((64, 64, 3), [70, 130, 190], dtype=np.uint8)
        sample = rgb_to_tensors(image, 64)
        restored = lab_to_rgb(sample["L"], sample["ab"])
        self.assertLess(float(np.abs(restored - image / 255.0).max()), 0.006)

    def test_ddcolor_neutral_input(self):
        values = neutral_lab_l_to_srgb(torch.tensor([[[[-1.0, 0.0, 1.0]]]]))
        self.assertEqual(tuple(values.shape), (1, 3, 1, 3))
        self.assertTrue(torch.all(values >= 0) and torch.all(values <= 1))
        self.assertTrue(torch.all(values[..., 1:] >= values[..., :-1]))

    def test_chroma_aware_loss_prefers_correct_color(self):
        target = torch.tensor([[[[0.12]], [[-0.08]]]])
        exact = chroma_aware_loss(target, target, {})
        desaturated = chroma_aware_loss(target * 0.5, target, {})
        neutral = chroma_aware_loss(torch.zeros_like(target), target, {})
        self.assertLess(exact.item(), desaturated.item())
        self.assertLess(desaturated.item(), neutral.item())

    def test_v5_edge_loss_penalizes_color_error_at_luminance_boundary(self):
        luminance = torch.zeros(1, 1, 8, 8)
        luminance[:, :, :, 4:] = 1.0
        target = torch.zeros(1, 2, 8, 8)
        prediction = target.clone()
        prediction[:, :, :, 3:5] = 0.25
        common = {"loss_version": "hue_chroma_edge_v5", "edge_gamma": 0.7}
        without_edge = chroma_aware_loss(
            prediction, target, {**common, "edge_chroma_loss_weight": 0.0}, luminance)
        with_edge = chroma_aware_loss(
            prediction, target, {**common, "edge_chroma_loss_weight": 0.25}, luminance)
        exact = chroma_aware_loss(target, target, common, luminance)
        self.assertGreater(with_edge.item(), without_edge.item())
        self.assertAlmostEqual(exact.item(), 0.0, places=7)

    def test_v5_calibration_penalizes_under_and_over_chroma(self):
        luminance = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[0.0, 0.20], [0.0, 0.20]],
                                [[0.0, 0.00], [0.0, 0.00]]]])
        exact = chroma_aware_loss(target, target, {"loss_version": "chroma_calibration_v5"}, luminance)
        flattened = torch.full_like(target, 0.08)
        wrong = chroma_aware_loss(flattened, target, {"loss_version": "chroma_calibration_v5"}, luminance)
        self.assertLess(exact.item(), wrong.item())

    def test_selection_score_rejects_low_chroma_shortcut(self):
        config = {"selection_chroma_weight": 0.03, "selection_chroma_min_ratio": 0.8,
                  "selection_chroma_max_ratio": 1.2, "selection_chroma_band_weight": 0.2}
        self.assertLess(chroma_selection_score(0.1, 1.0, config),
                        chroma_selection_score(0.1, 0.6, config))

    def test_v6_selection_score_penalizes_bad_image_distribution(self):
        config = {"selection_under_fraction_weight": 0.5,
                  "selection_over_fraction_weight": 0.5}
        good = chroma_selection_score(0.1, 1.0, config, 0.2, 0.1)
        bad = chroma_selection_score(0.1, 1.0, config, 0.5, 0.3)
        self.assertLess(good, bad)

    def test_source_balanced_score_weights_sources_equally(self):
        metrics = {
            "val_selection_score": 0.2,
            "val_historical_video_selection_score": 0.08,
            "val_vietnam_video_selection_score": 0.14,
        }
        add_source_balanced_score(metrics)
        self.assertAlmostEqual(metrics["val_source_balanced_selection_score"], 0.11)

        # Recomputing must not accidentally include the previously derived score.
        add_source_balanced_score(metrics)
        self.assertAlmostEqual(metrics["val_source_balanced_selection_score"], 0.11)


if __name__ == "__main__":
    unittest.main()
