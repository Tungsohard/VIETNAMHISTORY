"""Compare VietColorNet checkpoints on modern photographs and on the held-out ViCoW film.

Every checkpoint colours the same greyscale inputs and is scored against the original colours:

* ``modern-val`` / ``modern-test``: Viet-Cultural-VQA and KTVIC images of the V9+ protocols;
* ``vicow-test``: the frames of *Hà Nội 12 ngày đêm*, which no V9+ model has trained on.

Reported per checkpoint and set: Delta E 2000, ab MAE, edge ab MAE (colour error within 2 px
of luminance edges, i.e. where colour bleeds), median chroma ratio, and the mean predicted vs
true ab over green, near-neutral, warm and blue ground-truth pixels (the yellow-brown bias).
ViCoW's colours are those of old film, so on ``vicow-test`` a *lower* chroma ratio is not
worse: read it together with the modern sets.

Usage:
    python scripts/evaluate/compare_checkpoints.py --set modern-test --set vicow-test \
        V12=checkpoints/runs/vietcolornet_v12_384/epoch_06.pth V15=checkpoints/runs/vietcolornet_v15_vicow/epoch_03.pth
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.inference import ColorizationEngine, _normalized_luminance

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources"
GROUPS = {
    "green": lambda gt, c: gt[..., 0] < -6,
    "neutral": lambda gt, c: c < 6,
    "warm": lambda gt, c: (gt[..., 1] > 12) & (gt[..., 0] > 0),
    "blue": lambda gt, c: gt[..., 1] < -10,
}


def load_set(name, stride, max_side):
    """Return a list of RGB uint8 images for an evaluation set."""
    if name.startswith("modern"):
        split = name.split("-")[1]
        rows = list(csv.DictReader(open(ROOT / "data/protocols/vietnam_modern_v9" / f"{split}.csv",
                                        encoding="utf-8-sig")))[::stride]
        roots = {"viet_cultural_vqa": SOURCES / "viet_cultural_vqa", "ktvic": SOURCES / "ktvic"}
        paths = [roots[r["source"]] / r["colorPath"] for r in rows]
    elif name == "vicow-test":
        rows = list(csv.DictReader(open(ROOT / "data/protocols/vietnam_protocol_v1/test.csv", encoding="utf-8-sig")))
        paths = [SOURCES / "vicow" / r["colorPath"] for r in rows][::stride]
    else:
        raise ValueError(f"Unknown set: {name}")
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((max_side, max_side))
        images.append(np.asarray(image))
    return images


@torch.inference_mode()
def evaluate(engine, images):
    de, ab_mae, edge_mae, ratios = [], [], [], []
    groups = {g: [] for g in GROUPS}
    for rgb in images:
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        luminance, tensor = _normalized_luminance(gray, engine.image_size)
        with torch.autocast("cuda", enabled=engine.device.type == "cuda"):
            ab = engine.model(tensor[None].to(engine.device)).float().cpu().numpy()[0].transpose(1, 2, 0) * 128.0
        ab = cv2.resize(ab, luminance.shape[::-1], interpolation=cv2.INTER_CUBIC)
        truth = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        predicted = np.concatenate([truth[..., :1], ab], axis=-1)
        gt = truth[..., 1:]
        chroma = np.hypot(gt[..., 0], gt[..., 1])
        error = np.abs(ab - gt).mean(axis=-1)
        edges = cv2.dilate(cv2.Canny(np.clip(truth[..., 0] * 2.55, 0, 255).astype(np.uint8), 80, 160),
                           np.ones((5, 5), np.uint8)) > 0
        de.append(float(deltaE_ciede2000(truth, predicted).mean()))
        ab_mae.append(float(error.mean()))
        edge_mae.append(float(error[edges].mean()) if edges.any() else float(error.mean()))
        ratios.append(float(np.hypot(ab[..., 0], ab[..., 1]).mean() / max(chroma.mean(), 1e-3)))
        for name, rule in GROUPS.items():
            mask = rule(gt, chroma)
            if mask.sum() > 200:
                groups[name].append(np.r_[ab[mask].mean(axis=0), gt[mask].mean(axis=0)])
    result = {"images": len(images), "delta_e2000": float(np.mean(de)), "ab_mae": float(np.mean(ab_mae)),
              "edge_ab_mae": float(np.mean(edge_mae)), "chroma_ratio_median": float(np.median(ratios))}
    for name, values in groups.items():
        if values:
            mean = np.mean(values, axis=0)
            result[f"{name}_pred_ab"] = [float(mean[0]), float(mean[1])]
            result[f"{name}_true_ab"] = [float(mean[2]), float(mean[3])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="NAME=path/to/checkpoint.pth")
    parser.add_argument("--set", dest="sets", action="append", default=None,
                        choices=("modern-val", "modern-test", "vicow-test"))
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth image of a set")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--output", type=Path, help="Write the results as JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sets = args.sets or ["modern-test", "vicow-test"]
    images = {name: load_set(name, args.stride, args.max_side) for name in sets}
    results = {}
    for item in args.checkpoints:
        name, _, path = item.partition("=")
        engine = ColorizationEngine(ROOT / path, device=args.device)
        results[name] = {s: evaluate(engine, images[s]) for s in sets}
        del engine
        torch.cuda.empty_cache()
    for s in sets:
        print(f"\n== {s} ({results[next(iter(results))][s]['images']} images)")
        print(f"{'model':14s} {'dE2000':>7} {'abMAE':>6} {'edgeAB':>7} {'chroma':>7} | green a,b pred -> true      | neutral b pred (true)")
        for name, by_set in results.items():
            r = by_set[s]
            green = r.get("green_pred_ab"), r.get("green_true_ab")
            neutral = r.get("neutral_pred_ab"), r.get("neutral_true_ab")
            g = f"({green[0][0]:+5.1f},{green[0][1]:+5.1f}) -> ({green[1][0]:+5.1f},{green[1][1]:+5.1f})" if green[0] else "n/a"
            n = f"{neutral[0][1]:+5.1f} ({neutral[1][1]:+5.1f})" if neutral[0] else "n/a"
            print(f"{name:14s} {r['delta_e2000']:7.3f} {r['ab_mae']:6.3f} {r['edge_ab_mae']:7.3f} "
                  f"{r['chroma_ratio_median']:7.3f} | {g:28s} | {n}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
