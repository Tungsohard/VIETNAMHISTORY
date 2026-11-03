"""Statistical comparison of colourisation models on the held-out test sets.

Reports, for every model and set, the mean of each per-image metric with a 95% bootstrap
confidence interval, and for every pair of models the mean paired difference, its 95%
bootstrap interval, a Wilcoxon signed-rank p-value and the share of images where the first
model is better. This is what answers "is the improvement stable and reliable?" rather than
"is the average lower?".

Models colour the greyscale version of each test image and are scored against the original
colours. By default raw model output is compared (no SAM or post-processing), so differences
come from the networks. ``--mode natural`` applies the region-refinement post-processing and
``--sam`` adds SAM anti-bleeding, identically for every model, to measure the full pipeline. Test frames of one film are
strongly correlated, so ``vicow-test`` is resampled in blocks of consecutive frames.

Usage:
    python scripts/evaluate/statistical_comparison.py \
        historical=checkpoints/models/vietcolornet_historical.pth \
        vivid=checkpoints/models/vietcolornet_vivid.pth \
        ddcolor=@selection --output outputs/reports/statistical_comparison
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon
from skimage.color import deltaE_ciede2000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import ROOT, load_set  # noqa: E402
from colorization.inference import ColorizationEngine  # noqa: E402

METRICS = ("delta_e2000", "ab_mae", "edge_ab_mae", "chroma_ratio")
LOWER_IS_BETTER = {"delta_e2000": True, "ab_mae": True, "edge_ab_mae": True, "chroma_ratio": None}
BLOCK = {"modern-test": 1, "vicow-test": 12}
SELECTION = ROOT / "outputs" / "reports" / "baseline_selection.json"


class CachedSegmenter:
    """Share SAM object maps between models: the greyscale input is identical for all of them."""

    def __init__(self, segmenter):
        self.segmenter, self.cache = segmenter, {}

    def labels(self, rgb):
        key = hash(rgb.tobytes())
        if key not in self.cache:
            self.cache[key] = self.segmenter.labels(rgb).astype(np.int16)
        return self.cache[key].astype(np.int32)


def grey_input(rgb):
    """The image as the model would receive an archival photo: luminance only, no colour."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab[..., 1:] = 0.0
    return np.rint(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)


def image_metrics(engine, rgb, mode=None):
    """Per-image metrics of one model on one RGB uint8 image."""
    result = np.asarray(engine.colorize_pil(Image.fromarray(grey_input(rgb)), 1.0, mode=mode))
    truth = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    predicted = cv2.cvtColor(result.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    gt, ab = truth[..., 1:], predicted[..., 1:]
    error = np.abs(ab - gt).mean(axis=-1)
    edges = cv2.dilate(cv2.Canny(np.clip(truth[..., 0] * 2.55, 0, 255).astype(np.uint8), 80, 160),
                       np.ones((5, 5), np.uint8)) > 0
    chroma_gt = np.hypot(gt[..., 0], gt[..., 1]).mean()
    return {"delta_e2000": float(deltaE_ciede2000(truth, predicted).mean()),
            "ab_mae": float(error.mean()),
            "edge_ab_mae": float(error[edges].mean()) if edges.any() else float(error.mean()),
            "chroma_ratio": float(np.hypot(ab[..., 0], ab[..., 1]).mean() / max(chroma_gt, 1e-3))}


def block_bootstrap(values, block, resamples, rng):
    """Bootstrap distribution of the mean, resampling blocks of consecutive items."""
    n = len(values)
    blocks = [values[i:i + block] for i in range(0, n, block)]
    means = np.empty(resamples)
    for r in range(resamples):
        picked = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        means[r] = picked.mean()
    return means


def interval(values, block, resamples, rng):
    means = block_bootstrap(values, block, resamples, rng)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def load_engine(spec, device, segmenter=None):
    if spec == "@selection":
        engine = ColorizationEngine.from_selection(SELECTION, device=device)
    else:
        engine = ColorizationEngine(ROOT / spec, device=device)
    engine.segmenter = segmenter
    return engine


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="NAME=checkpoint path, or NAME=@selection for the DDColor baseline")
    parser.add_argument("--set", dest="sets", action="append", choices=tuple(BLOCK), default=None)
    parser.add_argument("--mode", choices=("natural", "modern", "vivid"), default=None,
                        help="post-processing mode applied to every model (default: raw output)")
    parser.add_argument("--sam", action="store_true", help="SAM anti-bleeding for every model (needs --mode)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reports" / "statistical_comparison")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    sets = args.sets or list(BLOCK)
    rng = np.random.default_rng(args.seed)

    segmenter = None
    if args.sam:
        from colorization.segmentation import ObjectSegmenter
        if args.mode is None:
            parser.error("--sam requires --mode")
        segmenter = CachedSegmenter(ObjectSegmenter.try_create(device=args.device))
        if segmenter.segmenter is None:
            parser.error("SAM is unavailable")

    images = {name: load_set(name, args.stride, args.max_side) for name in sets}
    per_image = {s: {} for s in sets}
    for item in args.models:
        name, _, spec = item.partition("=")
        engine = load_engine(spec, args.device, segmenter)
        for s in sets:
            rows = [image_metrics(engine, rgb, args.mode) for rgb in images[s]]
            per_image[s][name] = {m: np.array([r[m] for r in rows]) for m in METRICS}
            print(f"{name:12s} {s}: {len(rows)} images, dE={per_image[s][name]['delta_e2000'].mean():.3f}", flush=True)
        del engine
        import torch
        torch.cuda.empty_cache()

    names = [item.partition("=")[0] for item in args.models]
    report = {"settings": {"stride": args.stride, "max_side": args.max_side, "resamples": args.resamples,
                           "seed": args.seed, "block": BLOCK, "inference": ("raw model output (no SAM/post-processing)" if args.mode is None else
                                      f"mode={args.mode}" + (" + SAM" if args.sam else ""))},
              "sets": {}}
    for s in sets:
        entry = {"images": len(images[s]), "models": {}, "pairs": []}
        for name in names:
            entry["models"][name] = {m: {"mean": float(per_image[s][name][m].mean()),
                                         "std": float(per_image[s][name][m].std(ddof=1)),
                                         "ci95": interval(per_image[s][name][m], BLOCK[s], args.resamples, rng)}
                                     for m in METRICS}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = {"a": a, "b": b}
                for m in METRICS:
                    diff = per_image[s][a][m] - per_image[s][b][m]
                    stat = {"mean_diff": float(diff.mean()),
                            "ci95": interval(diff, BLOCK[s], args.resamples, rng)}
                    if np.any(diff != 0):
                        stat["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
                    if LOWER_IS_BETTER[m] is not None:
                        stat["a_better_share"] = float((diff < 0).mean())
                    pair[m] = stat
                entry["pairs"].append(pair)
        report["sets"][s] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report, names), encoding="utf-8")
    print(render_markdown(report, names))


def render_markdown(report, names):
    lines = []
    labels = {"delta_e2000": "ΔE2000 ↓", "ab_mae": "AB-MAE ↓", "edge_ab_mae": "Edge AB-MAE ↓", "chroma_ratio": "Chroma ratio"}
    for s, entry in report["sets"].items():
        lines += [f"### {s} ({entry['images']} ảnh)", "", "| Model | " + " | ".join(labels.values()) + " |",
                  "|---|" + "---|" * len(labels)]
        for name in names:
            cells = []
            for m in METRICS:
                v = entry["models"][name][m]
                cells.append(f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:", "",
                  "| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |", "|---|---|---|---|---|---|"]
        for pair in entry["pairs"]:
            for m in METRICS[:3]:
                d = pair[m]
                lines.append(f"| {pair['a']} vs {pair['b']} | {labels[m]} | {d['mean_diff']:+.3f} | "
                             f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d.get('wilcoxon_p', float('nan')):.2g} | "
                             f"{d['a_better_share'] * 100:.0f}% |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
