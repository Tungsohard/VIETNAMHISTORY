"""Estimate the ab colour prior of a training mix and its class-rebalancing weights.

Regression colourisers hedge towards the mean of plausible colours, which for natural
images is a desaturated yellow-brown (Zhang et al., "Colorful Image Colorization",
ECCV 2016). Following that paper, pixels whose ground-truth ab is rare in training
are up-weighted: w = 1 / ((1 - lambda) * p_smoothed + lambda / Q), normalised so that
the expected weight under the prior is 1.

Images are sampled from each source in proportion to ``train_source_weights`` of the
config, i.e. as the training loader sees them.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256


ROOT = Path(__file__).resolve().parents[2]
BIN = 10.0
EDGE = 110.0


def ab_histogram(rgb, bins):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    index = np.clip(((lab[..., 1:] + EDGE) / BIN).astype(int), 0, bins - 1)
    return np.bincount(index[..., 0].ravel() * bins + index[..., 1].ravel(), minlength=bins * bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--imagenet-manifest", type=Path,
                        default=ROOT / "data" / "processed" / "imagenet_targeted_v10.json")
    parser.add_argument("--craw-root", type=Path, default=ROOT / "data" / "sources")
    parser.add_argument("--wikimedia-root", type=Path, default=ROOT / "data" / "sources" / "wikimedia_vietnam")
    parser.add_argument("--military-root", type=Path, default=ROOT / "data" / "sources" / "wikimedia_vietnam_military")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--rebalance-lambda", type=float, default=0.5)
    parser.add_argument("--smoothing-sigma", type=float, default=0.5, help="Gaussian sigma in bins")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = config["train_source_weights"]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    with (args.protocol.parent / protocol["splits"]["train"]["path"]).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    roots = {"viet_cultural_vqa": args.craw_root / "viet_cultural_vqa", "ktvic": args.craw_root / "ktvic",
             "wikimedia_commons": args.wikimedia_root, "vietnam_military": args.military_root}
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)

    bins = int(2 * EDGE / BIN)
    histogram = np.zeros(bins * bins, dtype=np.float64)
    total = sum(weights.values())
    counts = {}
    for source, weight in weights.items():
        n = int(round(args.samples * weight / total))
        counts[source] = n
        source_hist = np.zeros_like(histogram)
        if source == "imagenet_replay":
            manifest = json.loads(args.imagenet_manifest.read_text(encoding="utf-8"))
            index = np.load(args.imagenet_manifest.parent / manifest["splits"]["train"]["path"])
            shards = {}
            for shard, row in index[rng.choice(len(index), n, replace=False)]:
                if shard not in shards:
                    shards[shard] = np.load(args.imagenet_manifest.parent / manifest["shards"][shard]["path"],
                                            mmap_mode="r")
                source_hist += ab_histogram(np.asarray(shards[shard][row]).reshape(3, 64, 64).transpose(1, 2, 0), bins)
        else:
            for row in rng.choice(by_source[source], min(n, len(by_source[source])), replace=False):
                with Image.open(roots[source] / row["colorPath"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((128, 128))
                    source_hist += ab_histogram(np.asarray(image), bins)
        # Each source contributes its sampling share, independent of image sizes.
        histogram += weight / total * source_hist / max(source_hist.sum(), 1)

    probability = histogram.reshape(bins, bins)
    smoothed = cv2.GaussianBlur(probability, (0, 0), args.smoothing_sigma)
    smoothed /= smoothed.sum()
    in_gamut = probability > 0
    q = int(in_gamut.sum())
    rebalance = 1.0 / ((1.0 - args.rebalance_lambda) * smoothed + args.rebalance_lambda / q)
    rebalance /= (smoothed * rebalance).sum()
    rebalance[~in_gamut] = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "bin_size": BIN, "edge": EDGE, "bins": bins, "lambda": args.rebalance_lambda,
        "smoothing_sigma_bins": args.smoothing_sigma, "in_gamut_bins": q,
        "samples_per_source": counts, "config": str(args.config), "config_sha256": sha256(args.config),
        "protocol_sha256": sha256(args.protocol),
        "prior": smoothed.tolist(), "weights": rebalance.tolist(),
    }), encoding="utf-8")
    centre = lambda a, b: rebalance[int((a + EDGE) / BIN), int((b + EDGE) / BIN)]
    print(json.dumps({"in_gamut_bins": q, "weight_range": [float(rebalance[in_gamut].min()), float(rebalance.max())],
                      "weight_grey(0,0)": float(centre(0, 0)), "weight_yellow_brown(5,20)": float(centre(5, 20)),
                      "weight_green(-25,25)": float(centre(-25, 25)), "weight_red(45,30)": float(centre(45, 30)),
                      "weight_blue(0,-35)": float(centre(0, -35))}, indent=1))


if __name__ == "__main__":
    main()
