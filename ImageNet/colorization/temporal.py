"""Optical-flow temporal stabilization for frame-wise color predictions."""
import cv2
import numpy as np


class TemporalABStabilizer:
    """Blend aligned prior chroma into stable regions while resetting on scene cuts."""

    def __init__(self, strength=0.55, scene_cut_threshold=32.0, confidence_threshold=28.0):
        if not 0.0 <= strength <= 0.9:
            raise ValueError("Temporal strength must be between 0 and 0.9")
        self.strength = float(strength)
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.previous_gray = None
        self.previous_ab = None

    def reset(self):
        self.previous_gray = None
        self.previous_ab = None

    def apply(self, luminance, current_ab):
        gray = np.clip(luminance * 2.55, 0, 255).astype(np.uint8)
        current_ab = np.asarray(current_ab, dtype=np.float32)
        if self.strength == 0 or self.previous_gray is None:
            stable = current_ab
        elif np.mean(np.abs(gray.astype(np.float32) - self.previous_gray.astype(np.float32))) >= self.scene_cut_threshold:
            stable = current_ab
        else:
            # Backward flow maps each current pixel to its location in the prior frame.
            flow = cv2.calcOpticalFlowFarneback(
                gray, self.previous_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            height, width = gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                         np.arange(height, dtype=np.float32))
            map_x, map_y = grid_x + flow[..., 0], grid_y + flow[..., 1]
            warped_ab = cv2.remap(self.previous_ab, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT101)
            warped_gray = cv2.remap(self.previous_gray, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT101)
            alignment_error = np.abs(gray.astype(np.float32) - warped_gray.astype(np.float32))
            confidence = np.clip(1.0 - alignment_error / self.confidence_threshold, 0.0, 1.0)
            alpha = (self.strength * confidence)[..., None]
            stable = current_ab * (1.0 - alpha) + warped_ab * alpha

        self.previous_gray = gray
        self.previous_ab = stable.copy()
        return stable


def warp_error(previous_luminance, current_luminance, previous_ab, current_ab,
               confidence_threshold=28.0):
    """Return flow-aligned chroma MAE for non-cut consecutive frames."""
    previous_gray = np.clip(previous_luminance * 2.55, 0, 255).astype(np.uint8)
    current_gray = np.clip(current_luminance * 2.55, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    map_x, map_y = grid_x + flow[..., 0], grid_y + flow[..., 1]
    warped_ab = cv2.remap(np.asarray(previous_ab, np.float32), map_x, map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warped_gray = cv2.remap(previous_gray, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
    confidence = np.clip(1.0 - np.abs(current_gray.astype(np.float32) -
                                      warped_gray.astype(np.float32)) / confidence_threshold, 0, 1)
    weights = confidence[..., None]
    return float((np.abs(np.asarray(current_ab) - warped_ab) * weights).sum() /
                 max(weights.sum() * 2, 1e-8))
