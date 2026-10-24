import cv2
import numpy as np
import torch


def degrade_luminance(luminance, rng=np.random):
    """Simulate archival damage on L (0..100) only; the ab target stays clean.

    Old photographs and film scans are faded, blurred, noisy and JPEG-compressed.
    Training on these inputs teaches the model to colour degraded scans instead
    of only clean digital photographs.
    """
    L = luminance.astype(np.float32)
    height, width = L.shape
    if rng.random() < 0.5:
        mean = float(L.mean())
        L = mean + (L - mean) * rng.uniform(0.6, 1.0) + rng.uniform(-6.0, 6.0)
    if rng.random() < 0.5:
        L = cv2.GaussianBlur(L, (0, 0), sigmaX=rng.uniform(0.5, 2.0))
    elif rng.random() < 0.4:
        factor = rng.uniform(1.5, 3.0)
        small = cv2.resize(L, (max(8, int(width / factor)), max(8, int(height / factor))),
                           interpolation=cv2.INTER_AREA)
        L = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.5:
        L = L + rng.normal(0.0, rng.uniform(1.0, 5.0), L.shape).astype(np.float32)
    if rng.random() < 0.4:
        encoded = np.clip(L * 2.55, 0, 255).astype(np.uint8)
        ok, buffer = cv2.imencode(".jpg", encoded, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 85))])
        if ok:
            L = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 2.55
    return np.clip(L, 0.0, 100.0)


def rgb_to_tensors(rgb, image_size, augment=False, degrade=0.0):
    interpolation = cv2.INTER_AREA if image_size <= min(rgb.shape[:2]) else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=interpolation)
    if augment and np.random.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    rgb = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if degrade and np.random.random() < degrade:
        lab[..., 0] = degrade_luminance(lab[..., 0])
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
