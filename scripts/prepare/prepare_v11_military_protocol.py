"""Extend the V9 modern protocol with licensed Vietnamese military photographs (V11).

Images come from ``collect_wikimedia_vietnam.py`` run on Vietnam People's Army
categories. Commons categories also hold flags, insignia, camouflage swatches,
maps and scanned documents, which teach nothing about photographic colour, so
only photographs are kept: no graphic keywords in the title/description, not a
PNG, and colours not concentrated in a small palette. Validation and test stay
exactly those of V9; military images are removed when near-identical to them.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from prepare_v8_craw_protocol import read_csv, write_csv
from prepare_v9_modern_protocol import describe, near_any


ROOT = Path(__file__).resolve().parents[2]
GRAPHIC_WORDS = re.compile(
    r"logo|emblem|insignia|badge|flag|ensign|map|document|certificate|letter|rank|camo|pattern|seal|"
    r"coat.of.arms|huy.hi|bản.đồ|giấy|stamp|postcard|poster|diagram|chart|medal|ribbon|award|"
    r"shoulder|collar|epaulet|sketch|drawing|painting|tranh|vector|plate|plan|zone|regions|"
    r"vanity.fair|petit.parisien|bataille|battle.of", re.I)


def mean_chroma(path):
    """Mean Lab chroma; black-and-white or sepia scans sit below ~5 and teach desaturation."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((256, 256))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def palette_concentration(path):
    """Share of pixels covered by the 16 most frequent 15-bit colours (graphics are near 1)."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((256, 256))
        quantized = (np.asarray(image) >> 3).reshape(-1, 3).astype(np.int32)
    counts = np.sort(np.bincount(quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]))[::-1]
    return float(counts[:16].sum() / counts.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-protocol", type=Path, default=ROOT / "data" / "protocols" / "vietnam_modern_v9" / "manifest.json")
    parser.add_argument("--military-root", type=Path, default=ROOT / "data" / "sources" / "wikimedia_vietnam_military")
    parser.add_argument("--craw-root", type=Path, default=ROOT / "data" / "sources")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "vietnam_modern_military_v11")
    parser.add_argument("--max-palette-concentration", type=float, default=0.66)
    parser.add_argument("--min-chroma", type=float, default=5.0)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"V11 protocol already exists: {manifest_path}")

    v9 = json.loads(args.v9_protocol.read_text(encoding="utf-8"))
    splits = {name: read_csv(args.v9_protocol.parent / v9["splits"][name]["path"]) for name in ("train", "val", "test")}
    roots = {"viet_cultural_vqa": args.craw_root / "viet_cultural_vqa", "ktvic": args.craw_root / "ktvic"}
    eval_hashes = np.asarray([describe(roots[row["source"]] / row["colorPath"])[1]
                              for name in ("val", "test") for row in splits[name]], dtype=np.uint64)

    accepted = [json.loads(line) for line in (args.military_root / "accepted.jsonl").open(encoding="utf-8")]
    rejected, rows, hashes = Counter(), [], []
    for item in accepted:
        path = args.military_root / item["local_path"]
        text = f"{item['title']} {item.get('description', '')[:300]}"
        if GRAPHIC_WORDS.search(text):
            rejected["graphic_keyword"] += 1
        elif item["title"].lower().endswith(".png"):
            rejected["png_graphic"] += 1
        elif palette_concentration(path) >= args.max_palette_concentration:
            rejected["palette_concentration"] += 1
        elif mean_chroma(path) < args.min_chroma:
            rejected["low_chroma"] += 1
        else:
            rows.append({"source": "vietnam_military", "fileName": item["title"],
                         "video": f"Commons-{item['source_category']}",
                         "colorPath": item["local_path"], "grayPath": ""})
            # Recompute with the same dHash as the V9 evaluation hashes (the crawler uses its own).
            hashes.append(describe(path)[1])
    near = near_any(np.asarray(hashes, dtype=np.uint64), eval_hashes, args.near_duplicate_distance)
    rejected["near_eval"] = int(near.sum())
    military = [row for row, flag in zip(rows, near) if not flag]

    args.output.mkdir(parents=True, exist_ok=False)
    output_rows = {"train": splits["train"] + military, "val": splits["val"], "test": splits["test"]}
    split_info = {}
    for name, split_rows in output_rows.items():
        path = args.output / f"{name}.csv"
        write_csv(path, split_rows)
        split_info[name] = {"path": path.name, "sha256": sha256(path), "count": len(split_rows),
                            "videos": sorted({row["video"] for row in split_rows}),
                            "source_counts": dict(sorted(Counter(row["source"] for row in split_rows).items()))}
    manifest = {
        "protocol": "vietnam_modern_military_v11",
        "base_protocol": str(args.v9_protocol.resolve()),
        "base_protocol_sha256": sha256(args.v9_protocol),
        "policy": {"military_source": str(args.military_root.resolve()),
                   "military_accepted_by_crawler": len(accepted),
                   "military_kept": len(military),
                   "max_palette_concentration": args.max_palette_concentration,
                   "rejected": dict(rejected),
                   "validation_and_test": "unchanged from V9"},
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"military_kept": len(military), "rejected": dict(rejected),
                      "train": split_info["train"]["source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
