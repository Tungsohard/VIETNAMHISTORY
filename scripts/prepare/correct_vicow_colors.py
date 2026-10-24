"""Gently colour-correct the ViCoW training films before they are used to train V15.

The footage is scanned from old colour film: *Những người viết huyền thoại* is washed out
(mean chroma ~7 against ~17 for modern photographs) and *Vào Nam ra Bắc* leans green. Much
of that green is real jungle, so the correction is deliberately mild:

* cast: estimated from the 15% least-coloured mid-tone pixels (walls, sky, clothing,
  skin shadows), capped at ``--max-cast`` Lab units and removed only from near-neutral
  pixels, never pushed past grey;
* vibrance: films below ``--target-chroma`` get a bounded gain that lifts weak colours
  more than already saturated ones;
* sharpness: the 1280x720 frames are upscaled from low-resolution video, so the luminance
  is soft and the colour smears across edges. L gets a mild unsharp mask and the chroma is
  re-aligned to the sharpened L with an edge-aware guided filter, which is exactly the
  colour/edge relationship the colouriser is asked to learn.

Only the training films are corrected and written to ``data/sources/vicow_corrected``.
The validation and test films stay untouched in ``data/sources/vicow``: they are the
ground truth the model is measured against.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILMS = ("NhungNguoiVietLenHuyenThoai", "VaoNamRaBac")


def sharpen_and_align(lab, amount=0.7, sigma=1.4, radius=4, eps=3e-4):
    """Unsharp-mask L, then snap ab to the sharpened luminance edges (in place on a copy)."""
    from colorization.enhance import guided_filter
    lab = lab.copy()
    luminance = lab[..., 0]
    scale = max(luminance.shape) / 1280.0
    blur = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma * scale)
    luminance = np.clip(luminance + amount * (luminance - blur), 0.0, 100.0)
    guide = (luminance / 100.0).astype(np.float32)
    r = max(2, round(radius * scale))
    for c in (1, 2):
        lab[..., c] = guided_filter(guide, lab[..., c].astype(np.float32), r, eps)
    lab[..., 0] = luminance
    return lab


def correct_colors(rgb, max_cast=6.0, cast_strength=0.8, target_chroma=14.0, max_gain=1.5, sharpen=True):
    """Return a gently cast-corrected, vibrance-lifted, sharper copy of an RGB uint8 image."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    if sharpen:
        lab = sharpen_and_align(lab)
    luminance, ab = lab[..., 0], lab[..., 1:].copy()
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    mid = (luminance > 20.0) & (luminance < 90.0)
    if mid.sum() > 200:
        neutral = mid & (chroma <= np.percentile(chroma[mid], 15))
        cast = np.median(ab[neutral], axis=0)
        magnitude = float(np.hypot(*cast))
        if magnitude > 1e-6:
            cast = cast * min(1.0, max_cast / magnitude) * cast_strength
            direction = cast / max(float(np.hypot(*cast)), 1e-6)
            weight = np.clip((25.0 - chroma) / 15.0, 0.0, 1.0)
            along = ab[..., 0] * direction[0] + ab[..., 1] * direction[1]
            removal = np.minimum(weight * float(np.hypot(*cast)), np.maximum(along, 0.0))
            ab -= removal[..., None] * direction
            chroma = np.hypot(ab[..., 0], ab[..., 1])
        mean_chroma = float(chroma[mid].mean())
        gain = float(np.clip(target_chroma / max(mean_chroma, 1e-3), 1.0, max_gain))
        local = 1.0 + (gain - 1.0) * (1.0 - np.clip(chroma / 40.0, 0.0, 1.0))
        taper = np.clip((luminance - 4.0) / 14.0, 0.0, 1.0) * np.clip((99.0 - luminance) / 9.0, 0.0, 1.0)
        ab *= (1.0 + (local - 1.0) * taper)[..., None]
    lab[..., 1:] = ab
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    return np.rint(out * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "sources" / "vicow")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sources" / "vicow_corrected")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    total = 0
    for film in TRAIN_FILMS:
        folder = args.source / "Color_image" / film
        out_folder = args.output / "Color_image" / film
        out_folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.jpg")):
            with Image.open(path) as image:
                corrected = correct_colors(np.asarray(image.convert("RGB")))
            Image.fromarray(corrected).save(out_folder / path.name, quality=args.quality)
            total += 1
    print(f"corrected {total} frames of {len(TRAIN_FILMS)} training films into {args.output}")


if __name__ == "__main__":
    main()
