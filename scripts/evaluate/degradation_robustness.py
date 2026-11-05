"""Robustness of colourisation to degraded (archival-quality) greyscale input.

Real archival photographs are blurred, noisy, faded, scratched and heavily compressed, while the
held-out test images are clean. Every test image is turned into a greyscale input and degraded
at four levels (clean, mild, medium, severe) by a deterministic recipe, identical for all models.
Models colour the degraded input and are scored against the ORIGINAL colours.

To isolate colour errors from the luminance damage (which every model passes through unchanged),
the predicted L* is replaced by the true L* before scoring: only chroma errors count.

Pre-registered criterion (written before any result was seen)
-------------------------------------------------------------
Primary outcome: the *increase* of colour error caused by degradation, i.e. (metric at level)
minus (metric on the clean input) per image, in colour-only dE2000. At the medium and severe
levels, on each of the two test sets, VietColorNet "historical" is called MORE ROBUST than DDColor
if its mean increase is smaller and the 95% block-bootstrap interval of the paired difference
(historical minus ddcolor) excludes 0. That is 4 tests (2 levels x 2 sets), corrected with Holm.
Everything else (the "vivid" model, the mild level, other metrics) is secondary and reported
without a claim. Results are reported whichever way they fall.

Usage:
    python scripts/evaluate/degradation_robustness.py historical=checkpoints/models/vietcolornet_historical.pth \
        vivid=checkpoints/models/vietcolornet_vivid.pth ddcolor=@selection --stride 2
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import ROOT, load_set  # noqa: E402
from statistical_comparison import BLOCK, block_permutation_p, grey_input, holm, interval, load_engine  # noqa: E402

LEVELS = {  # blur sigma, noise sigma, contrast, lift, scratches, jpeg quality
    "clean": None,
    "mild": dict(blur=1.0, noise=4.0, contrast=0.90, lift=6.0, scratches=0, jpeg=70),
    "medium": dict(blur=2.0, noise=9.0, contrast=0.75, lift=12.0, scratches=3, jpeg=40),
    "severe": dict(blur=3.0, noise=16.0, contrast=0.60, lift=20.0, scratches=8, jpeg=20),
}
METRICS = ("delta_e_colour", "ab_mae", "edge_ab_mae")
PRIMARY_LEVELS = ("medium", "severe")


def degrade(grey_rgb, level, seed):
    """Deterministically degrade a greyscale RGB image like an old photograph."""
    params = LEVELS[level]
    if params is None:
        return grey_rgb
    rng = np.random.default_rng(seed)
    g = grey_rgb[..., 0].astype(np.float32)
    height, width = g.shape
    if params["blur"] > 0:
        g = cv2.GaussianBlur(g, (0, 0), params["blur"])
    g = 128.0 + (g - 128.0) * params["contrast"] + params["lift"]           # fade
    g = g + rng.normal(0.0, params["noise"], g.shape)                        # grain
    for _ in range(params["scratches"]):                                     # vertical scratches
        x = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height // 2))
        y1 = int(rng.integers(y0 + height // 4, height))
        g[y0:y1, x] += float(rng.choice([-1.0, 1.0])) * float(rng.uniform(50, 110))
    g = np.clip(g, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg"]])
    g = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return np.repeat(g[..., None], 3, axis=2)


def colour_metrics(engine, rgb, degraded):
    result = np.asarray(engine.colorize_pil(Image.fromarray(degraded), 1.0))
    truth = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    pred = cv2.cvtColor(result.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    pred[..., 0] = truth[..., 0]                         # score colour only
    error = np.abs(pred[..., 1:] - truth[..., 1:]).mean(axis=-1)
    edges = cv2.dilate(cv2.Canny(np.clip(truth[..., 0] * 2.55, 0, 255).astype(np.uint8), 80, 160),
                       np.ones((5, 5), np.uint8)) > 0
    return {"delta_e_colour": float(deltaE_ciede2000(truth, pred).mean()),
            "ab_mae": float(error.mean()),
            "edge_ab_mae": float(error[edges].mean()) if edges.any() else float(error.mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reports" / "degradation_robustness")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    sets = list(BLOCK)
    images = {s: load_set(s, args.stride, args.max_side) for s in sets}
    values = {s: {} for s in sets}
    names = []
    for item in args.models:
        name, _, spec = item.partition("=")
        names.append(name)
        engine = load_engine(spec, args.device)
        for s in sets:
            per_level = {level: {m: [] for m in METRICS} for level in LEVELS}
            for index, rgb in enumerate(images[s]):
                clean = grey_input(rgb)
                for number, level in enumerate(LEVELS):
                    row = colour_metrics(engine, rgb, degrade(clean, level, 1000 * number + index))
                    for m in METRICS:
                        per_level[level][m].append(row[m])
            values[s][name] = {l: {m: np.array(v) for m, v in d.items()} for l, d in per_level.items()}
            print(f"{name:11s} {s}: " + "  ".join(f"{l}={values[s][name][l]['delta_e_colour'].mean():.2f}" for l in LEVELS),
                  flush=True)
        del engine
        import torch
        torch.cuda.empty_cache()

    report = {"stride": args.stride, "criterion": __doc__.split("Pre-registered criterion")[1].split("Usage:")[0].strip(),
              "sets": {}}
    primary = []
    for s in sets:
        entry = {"images": len(images[s]), "levels": {}}
        for level in LEVELS:
            cell = {"models": {}, "increase": {}, "pairs": []}
            for name in names:
                v = values[s][name][level]
                cell["models"][name] = {m: {"mean": float(v[m].mean()), "ci95": interval(v[m], BLOCK[s], args.resamples, rng)}
                                        for m in METRICS}
                if level != "clean":
                    inc = v["delta_e_colour"] - values[s][name]["clean"]["delta_e_colour"]
                    cell["increase"][name] = {"mean": float(inc.mean()), "ci95": interval(inc, BLOCK[s], args.resamples, rng)}
            if level != "clean":
                inc = {n: values[s][n][level]["delta_e_colour"] - values[s][n]["clean"]["delta_e_colour"] for n in names}
                b = "ddcolor"
                for a in (n for n in names if n != b):
                    diff = inc[a] - inc[b]              # negative = a degrades less than b
                    pair = {"a": a, "b": b, "mean_diff": float(diff.mean()),
                            "ci95": interval(diff, BLOCK[s], args.resamples, rng),
                            "block_perm_p": block_permutation_p(diff, BLOCK[s], args.permutations, rng)}
                    cell["pairs"].append(pair)
                    if a == "historical" and level in PRIMARY_LEVELS:
                        primary.append((s, level, pair))
            entry["levels"][level] = cell
        report["sets"][s] = entry
    adjusted = holm(np.array([p["block_perm_p"] for _, _, p in primary]))
    report["primary"] = [{"set": s, "level": level, "mean_diff": p["mean_diff"], "ci95": p["ci95"],
                          "block_perm_p_holm": float(adj),
                          "more_robust": bool(p["mean_diff"] < 0 and p["ci95"][1] < 0)}
                         for (s, level, p), adj in zip(primary, adjusted)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"),
                        **{f"{s}|{n}|{l}|{m}": values[s][n][l][m] for s in sets for n in names for l in LEVELS for m in METRICS})
    print(json.dumps(report["primary"], indent=2))


if __name__ == "__main__":
    main()
