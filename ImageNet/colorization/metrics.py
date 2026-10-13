import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity


def image_metrics(prediction, target):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(error ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    pred_lab = cv2.cvtColor(prediction.astype(np.float32), cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2LAB)
    pred_ab, target_ab = pred_lab[..., 1:], target_lab[..., 1:]
    return {
        "psnr": psnr,
        "ssim": float(structural_similarity(prediction, target, channel_axis=2, data_range=1.0)),
        "delta_e2000": float(deltaE_ciede2000(pred_lab, target_lab).mean()),
        "ab_mae": float(np.abs(pred_ab - target_ab).mean()),
        "pred_chroma": float(np.linalg.norm(pred_ab, axis=2).mean()),
        "target_chroma": float(np.linalg.norm(target_ab, axis=2).mean()),
    }

