"""Build the leakage-safe VietColorNet V8 protocol from crawled Vietnam datasets."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "fileName", "video", "colorPath", "grayPath"))
        writer.writeheader()
        writer.writerows(rows)


def image_dhash(image):
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def validate_image(path):
    with Image.open(path) as image:
        image.load()
        if min(image.size) < 128:
            return None, "too_small"
        return image_dhash(image), None


def source_path(row, roots):
    return roots[row["source"]] / row["colorPath"]


def eval_hashes(rows, roots):
    hashes = []
    for row in rows:
        value, reason = validate_image(source_path(row, roots))
        if reason:
            raise ValueError(f"Invalid frozen evaluation image ({reason}): {row}")
        hashes.append(value)
    return hashes


def near_eval(candidate, frozen, distance):
    return any((candidate ^ other).bit_count() <= distance for other in frozen)


def extract_ktvic_train(parquet_path, ktvic_root):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Install pyarrow from requirements.txt before preparing V8") from error
    output = ktvic_root / "images" / "train"
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path, columns=["image_uid", "image"])
    rows = []
    for item in table.to_pylist():
        uid = str(item["image_uid"])
        payload = item["image"]["bytes"]
        if not payload:
            raise ValueError(f"KTVIC image has no embedded bytes: {uid}")
        with Image.open(io.BytesIO(payload)) as image:
            fmt = (image.format or "JPEG").lower()
            extension = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, ".jpg")
            image.verify()
        relative = Path("images") / "train" / f"{uid}{extension}"
        target = ktvic_root / relative
        if not target.exists():
            target.write_bytes(payload)
        rows.append({
            "source": "ktvic", "fileName": uid, "video": "KTVIC-train",
            "colorPath": relative.as_posix(), "grayPath": "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_historical_still_replay_v4" / "manifest.json")
    parser.add_argument("--craw-root", type=Path, default=ROOT / "data" / "sources")
    parser.add_argument("--vietnam-root", type=Path, default=ROOT / "data" / "sources" / "vicow")
    parser.add_argument("--wikimedia-root", type=Path, default=ROOT / "data" / "sources" / "wikimedia_vietnam")
    parser.add_argument("--historical-root", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_v4_candidates_v2")
    parser.add_argument("--still-root", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_images")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "vietnam_historical_craw_v8")
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"V8 protocol already exists: {manifest_path}")

    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    if base.get("protocol") != "vietnam_historical_still_replay_v4":
        raise ValueError("V8 must extend the frozen V4 historical replay protocol")
    base_dir = args.base_protocol.parent
    base_rows = {split: read_csv(base_dir / base["splits"][split]["path"])
                 for split in ("train", "val", "test")}
    roots = {
        "vietnam_video": args.vietnam_root,
        "wikimedia_commons": args.wikimedia_root,
        "historical_video": args.historical_root,
        "historical_wikimedia_still": args.still_root,
    }
    frozen_hashes = eval_hashes(base_rows["val"] + base_rows["test"], roots)

    vqa_root = args.craw_root / "viet_cultural_vqa"
    vqa_annotations = json.loads((vqa_root / "splits" / "train_data.json").read_text(encoding="utf-8"))
    new_rows, rejected = [], Counter()
    seen_exact = set()

    def accept(path, row):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_exact:
                rejected["exact_duplicate"] += 1
                return
            value, reason = validate_image(path)
            if reason:
                rejected[reason] += 1
                return
            if near_eval(value, frozen_hashes, args.near_duplicate_distance):
                rejected["near_frozen_eval"] += 1
                return
            seen_exact.add(digest)
            new_rows.append(row)
        except Exception:
            rejected["invalid_image"] += 1

    for item in vqa_annotations:
        relative = item["image_path"].replace("data/images/", "images/", 1).replace("\\", "/")
        category = str(item.get("category", "unknown"))
        keyword = str(item.get("keyword", "unknown"))
        row = {
            "source": "viet_cultural_vqa",
            "fileName": f"{category}/{keyword}/{item.get('image_id', Path(relative).stem)}",
            "video": f"VQA-{category}", "colorPath": relative, "grayPath": "",
        }
        path = vqa_root / relative
        if path.is_file():
            accept(path, row)
        else:
            rejected["missing_vqa"] += 1

    ktvic_root = args.craw_root / "ktvic"
    ktvic_rows = extract_ktvic_train(ktvic_root / "data" / "train-00000-of-00001.parquet", ktvic_root)
    for row in ktvic_rows:
        accept(ktvic_root / row["colorPath"], row)

    train = base_rows["train"] + new_rows
    args.output.mkdir(parents=True, exist_ok=False)
    output_rows = {"train": train, "val": base_rows["val"], "test": base_rows["test"]}
    split_info = {}
    for split, rows in output_rows.items():
        path = args.output / f"{split}.csv"
        write_csv(path, rows)
        split_info[split] = {
            "path": path.name,
            "sha256": sha256(path),
            "count": len(rows),
            "videos": sorted({row["video"] for row in rows}),
            "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        }
    manifest = {
        "protocol": "vietnam_historical_craw_v8",
        "seed": 2026,
        "base_protocol": str(args.base_protocol.resolve()),
        "base_protocol_sha256": sha256(args.base_protocol),
        "policy": {
            "vqa_split": "train_only",
            "ktvic_split": "train_only",
            "frozen_validation_and_test": True,
            "near_duplicate_distance": args.near_duplicate_distance,
            "new_exact_duplicates_removed": rejected["exact_duplicate"],
            "new_near_frozen_eval_removed": rejected["near_frozen_eval"],
            "rejected": dict(rejected),
        },
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "train_count": split_info["train"]["count"],
        "new_train_count": len(new_rows),
        "source_counts": split_info["train"]["source_counts"],
        "rejected": dict(rejected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
