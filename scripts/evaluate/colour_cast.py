"""Measure where models go yellow or lose blue/green on the modern test photos.

Over pixels of one colour class (skin, grey, foliage, sky), in Lab against the true colours:
predicted b* and a* errors and, for skin, chroma ratio. Skin is a rough YCrCb mask on the true image.
Mean over images with a bootstrap 95% interval.

Run: python scripts/evaluate/colour_cast.py historical=<checkpoint> ddcolor=@selection
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import ROOT, load_set  # noqa: E402
from statistical_comparison import grey_input, interval, load_engine  # noqa: E402

MIN_FRACTION = 0.01  # a class must cover at least 1% of the image to be scored


def image_cast(engine, rgb, mode=None):
    result = np.asarray(engine.colorize_pil(Image.fromarray(grey_input(rgb)), 1.0, mode=mode))
    truth = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    pred = cv2.cvtColor(result.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L, a, b = truth[..., 0], truth[..., 1], truth[..., 2]
    pa, pb = pred[..., 1], pred[..., 2]
    chroma = np.hypot(a, b)
    out = {"overall_b_error": float((pb - b).mean())}
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = ((ycrcb[..., 1] >= 133) & (ycrcb[..., 1] <= 173) & (ycrcb[..., 2] >= 77) & (ycrcb[..., 2] <= 127)
            & (L > 25) & (L < 90) & (chroma > 8))
    masks = {"skin": skin, "neutral": chroma < 5, "green": (a < -10) & (b > 5), "sky": (b < -8) & (L > 45)}
    for name, mask in masks.items():
        if mask.mean() < MIN_FRACTION:
            continue
        if name == "skin":
            out["skin_a_error"] = float((pa - a)[mask].mean())
            out["skin_b_error"] = float((pb - b)[mask].mean())
            out["skin_ab_mae"] = float((np.abs(pa - a) + np.abs(pb - b))[mask].mean() / 2)
            out["skin_chroma_ratio"] = float(np.hypot(pa[mask], pb[mask]).mean() / max(chroma[mask].mean(), 1e-3))
        elif name == "neutral":
            out["neutral_b"] = float(pb[mask].mean())
            out["neutral_chroma"] = float(np.hypot(pa[mask], pb[mask]).mean())
        elif name == "green":
            out["green_a_error"] = float((pa - a)[mask].mean())
            out["green_b_error"] = float((pb - b)[mask].mean())
        else:
            out["sky_b_error"] = float((pb - b)[mask].mean())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="NAME=checkpoint, or NAME=@selection for the DDColor baseline")
    parser.add_argument("--mode", choices=("natural", "modern", "vivid"), default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--resamples", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reports" / "colour_cast")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    images = load_set("modern-test", args.stride, args.max_side)
    rng = np.random.default_rng(0)
    report = {"images": len(images), "mode": args.mode or "raw", "models": {}}
    for item in args.models:
        name, _, spec = item.partition("=")
        engine = load_engine(spec, args.device)
        rows = [image_cast(engine, rgb, args.mode) for rgb in images]
        del engine
        summary = {}
        for key in sorted({k for r in rows for k in r}):
            values = np.array([r[key] for r in rows if key in r])
            summary[key] = {"mean": float(values.mean()), "ci95": interval(values, 1, args.resamples, rng),
                            "images": int(len(values))}
        report["models"][name] = summary
        print(f"{name}: " + ", ".join(f"{k}={v['mean']:+.2f}" for k, v in summary.items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
