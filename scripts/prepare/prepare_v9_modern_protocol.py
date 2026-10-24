"""Build the VietColorNet V9 protocol from modern colour photographs only.

Historical and film sources carry sepia, olive or tinted casts that the model
learns as "old colour". V9 trains and selects on modern Vietnamese photographs
(Viet-Cultural-VQA, KTVIC, Wikimedia Commons) so that predictions are natural
and vivid. Validation and test use the official VQA val/test and KTVIC test
splits; every evaluation image is removed from train by exact and dHash
near-duplicate checks.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from prepare_v8_craw_protocol import image_dhash, read_csv, write_csv


ROOT = Path(__file__).resolve().parents[2]
MODERN_SOURCES = ("viet_cultural_vqa", "ktvic", "wikimedia_commons")


def popcount64(values):
    return np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_any(candidates, reference, distance, chunk=512):
    """Boolean mask: candidate hash within `distance` bits of any reference hash."""
    result = np.zeros(len(candidates), dtype=bool)
    if not len(reference):
        return result
    for start in range(0, len(candidates), chunk):
        block = candidates[start:start + chunk, None] ^ reference[None, :]
        bits = popcount64(block.reshape(-1)).reshape(block.shape)
        result[start:start + chunk] = (bits <= distance).any(axis=1)
    return result


def describe(path):
    """Return (sha256, dhash) or None when the image is unusable."""
    try:
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if min(image.size) < 128:
                return None
            return hashlib.sha256(payload).hexdigest(), image_dhash(image)
    except Exception:
        return None


def vqa_rows(vqa_root, split):
    items = json.loads((vqa_root / "splits" / f"{split}_data.json").read_text(encoding="utf-8"))
    rows, seen = [], set()
    for item in items:
        relative = item["image_path"].replace("data/images/", "images/", 1).replace("\\", "/")
        if relative in seen:
            continue
        seen.add(relative)
        category = str(item.get("category", "unknown"))
        rows.append({"source": "viet_cultural_vqa",
                     "fileName": f"{category}/{Path(relative).parent.name}/{Path(relative).stem}",
                     "video": f"VQA-{category}", "colorPath": relative, "grayPath": ""})
    return rows


def ktvic_test_rows(ktvic_root):
    import pyarrow.parquet as pq
    output = ktvic_root / "images" / "test"
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(ktvic_root / "data" / "test-00000-of-00001.parquet", columns=["image_uid", "image"])
    rows, seen = [], set()
    for item in table.to_pylist():
        uid = str(item["image_uid"])
        if uid in seen:
            continue
        seen.add(uid)
        payload = item["image"]["bytes"]
        with Image.open(io.BytesIO(payload)) as image:
            extension = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get((image.format or "jpeg").lower(), ".jpg")
        relative = Path("images") / "test" / f"{uid}{extension}"
        if not (ktvic_root / relative).exists():
            (ktvic_root / relative).write_bytes(payload)
        rows.append({"source": "ktvic", "fileName": uid, "video": "KTVIC-test",
                     "colorPath": relative.as_posix(), "grayPath": ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_historical_craw_v8" / "manifest.json")
    parser.add_argument("--craw-root", type=Path, default=ROOT / "data" / "sources")
    parser.add_argument("--wikimedia-root", type=Path, default=ROOT / "data" / "sources" / "wikimedia_vietnam")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "vietnam_modern_v9")
    parser.add_argument("--val-per-source", type=int, default=400)
    parser.add_argument("--test-per-source", type=int, default=600)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"V9 protocol already exists: {manifest_path}")
    rng = random.Random(args.seed)

    roots = {"viet_cultural_vqa": args.craw_root / "viet_cultural_vqa",
             "ktvic": args.craw_root / "ktvic", "wikimedia_commons": args.wikimedia_root}
    v8 = json.loads(args.v8_protocol.read_text(encoding="utf-8"))
    v8_train = read_csv(args.v8_protocol.parent / v8["splits"]["train"]["path"])
    train_pool = [row for row in v8_train if row["source"] in MODERN_SOURCES]

    eval_pools = {
        "val": {"viet_cultural_vqa": vqa_rows(roots["viet_cultural_vqa"], "val"),
                "ktvic": ktvic_test_rows(roots["ktvic"])},
        "test": {"viet_cultural_vqa": vqa_rows(roots["viet_cultural_vqa"], "test")},
    }
    # KTVIC has only one held-out split: share it between val and test without overlap.
    ktvic_eval = eval_pools["val"]["ktvic"]
    rng.shuffle(ktvic_eval)
    eval_pools["val"]["ktvic"] = ktvic_eval[:args.val_per_source]
    eval_pools["test"]["ktvic"] = ktvic_eval[args.val_per_source:args.val_per_source + args.test_per_source]

    rejected = Counter()
    seen_sha = {}

    def described(rows, tag):
        kept, hashes = [], []
        for row in rows:
            info = describe(roots[row["source"]] / row["colorPath"])
            if info is None:
                rejected[f"{tag}_invalid"] += 1
                continue
            digest, dhash = info
            if digest in seen_sha:
                rejected[f"{tag}_exact_duplicate"] += 1
                continue
            seen_sha[digest] = tag
            kept.append(row)
            hashes.append(dhash)
        return kept, np.asarray(hashes, dtype=np.uint64)

    # Evaluation images are claimed first; anything equal or near-equal in train is dropped.
    splits, eval_hashes = {}, []
    for split in ("val", "test"):
        rows = []
        for source, pool in eval_pools[split].items():
            pool = list(pool)
            rng.shuffle(pool)
            limit = args.val_per_source if split == "val" else args.test_per_source
            kept, hashes = described(pool[: limit * 2], split)
            if eval_hashes:
                near = near_any(hashes, np.concatenate(eval_hashes), args.near_duplicate_distance)
                rejected[f"{split}_near_other_eval"] += int(near.sum())
                kept = [row for row, flag in zip(kept, near) if not flag]
                hashes = hashes[~near]
            rows.extend(kept[:limit])
            eval_hashes.append(hashes[:limit])
        splits[split] = rows

    train_rows, train_hashes = described(train_pool, "train")
    near = near_any(train_hashes, np.concatenate(eval_hashes), args.near_duplicate_distance)
    rejected["train_near_eval"] = int(near.sum())
    splits["train"] = [row for row, flag in zip(train_rows, near) if not flag]

    args.output.mkdir(parents=True, exist_ok=False)
    split_info = {}
    for split in ("train", "val", "test"):
        path = args.output / f"{split}.csv"
        write_csv(path, splits[split])
        split_info[split] = {
            "path": path.name, "sha256": sha256(path), "count": len(splits[split]),
            "videos": sorted({row["video"] for row in splits[split]}),
            "source_counts": dict(sorted(Counter(row["source"] for row in splits[split]).items())),
        }
    manifest = {
        "protocol": "vietnam_modern_v9",
        "seed": args.seed,
        "base_protocol": str(args.v8_protocol.resolve()),
        "base_protocol_sha256": sha256(args.v8_protocol),
        "policy": {
            "sources": list(MODERN_SOURCES),
            "excluded": ["historical_video", "historical_wikimedia_still", "vietnam_video"],
            "reason": "historical/film targets carry sepia, olive or tinted casts",
            "vqa_eval": "official val/test splits, deduplicated by image",
            "ktvic_eval": "official test split, divided between val and test",
            "near_duplicate_distance": args.near_duplicate_distance,
            "rejected": dict(rejected),
        },
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["source_counts"] for k, v in split_info.items()} | {"rejected": dict(rejected)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
