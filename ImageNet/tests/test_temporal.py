import unittest

import numpy as np

from colorization.temporal import TemporalABStabilizer, warp_error


class TemporalTests(unittest.TestCase):
    def test_first_frame_is_unchanged(self):
        stabilizer = TemporalABStabilizer(0.6)
        luminance = np.full((32, 32), 50.0, dtype=np.float32)
        chroma = np.full((32, 32, 2), 0.1, dtype=np.float32)
        np.testing.assert_allclose(stabilizer.apply(luminance, chroma), chroma)

    def test_static_scene_reduces_chroma_jump(self):
        stabilizer = TemporalABStabilizer(0.6)
        luminance = np.full((32, 32), 50.0, dtype=np.float32)
        previous = np.zeros((32, 32, 2), dtype=np.float32)
        current = np.full((32, 32, 2), 0.1, dtype=np.float32)
        stabilizer.apply(luminance, previous)
        stable = stabilizer.apply(luminance, current)
        self.assertLess(float(stable.mean()), float(current.mean()))
        self.assertLess(warp_error(luminance, luminance, previous, stable),
                        warp_error(luminance, luminance, previous, current))

    def test_scene_cut_does_not_carry_prior_color(self):
        stabilizer = TemporalABStabilizer(0.8, scene_cut_threshold=20.0)
        stabilizer.apply(np.zeros((32, 32), np.float32),
                         np.full((32, 32, 2), -0.1, np.float32))
        current = np.full((32, 32, 2), 0.15, np.float32)
        stable = stabilizer.apply(np.full((32, 32), 100.0, np.float32), current)
        np.testing.assert_allclose(stable, current)


if __name__ == "__main__":
    unittest.main()
