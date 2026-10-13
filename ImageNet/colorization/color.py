import cv2
import numpy as np
import torch


def rgb_to_tensors(rgb, image_size, augment=False):
    interpolation = cv2.INTER_AREA if image_size <= min(rgb.shape[:2]) else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=interpolation)
    if augment and np.random.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    rgb = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return {
        "L": torch.from_numpy(lab[..., :1] / 50.0 - 1.0).permute(2, 0, 1),
        "ab": torch.from_numpy(lab[..., 1:] / 128.0).permute(2, 0, 1),
        "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
    }


def lab_to_rgb(l_tensor, ab_tensor):
    l = l_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    ab = ab_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    lab = np.concatenate(((l + 1.0) * 50.0, ab * 128.0), axis=-1).astype(np.float32)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
