"""Combine Vietnam V2 with curated historical frames using video-disjoint splits."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent
FIELDS = ["source", "fileName", "video", "colorPath", "grayPath"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_base(rows):
    return [{"source": row.get("source") or "vietnam_video",
             "fileName": row["fileName"], "video": row["video"],
             "colorPath": row["colorPath"], "grayPath": row.get("grayPath", "")}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=ROOT / "data" / "vietnam_protocol_v2" / "manifest.json")
    parser.add_argument("--historical", type=Path,
                        default=ROOT / "data" / "historical_vietnam_v4_curated" / "manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "vietnam_protocol_v3")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Protocol output already exists: {args.output}")

    base = json.loads(args.base.read_text(encoding="utf-8"))
    if base.get("protocol") != "vietnam_video_disjoint_wikimedia_train_v2":
        raise ValueError("Expected Vietnam Wikimedia protocol V2")
    for split in ("train", "val", "test"):
        path = args.base.parent / base["splits"][split]["path"]
        if sha256(path) != base["splits"][split]["sha256"]:
            raise ValueError(f"Base {split} fingerprint mismatch")

    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    if historical.get("protocol") != "historical_vietnam_adaptation_curated_v1":
        raise ValueError("Expected curated historical adaptation data")
    review_path = args.historical.parent / historical["manual_review_csv"]
    if sha256(review_path) != historical["manual_review_sha256"]:
        raise ValueError("Historical review fingerprint mismatch")
    reviewed = read_csv(review_path)
    kept = [row for row in reviewed if row["manual_decision"] == "keep"]
    if len(kept) != historical["kept_count"]:
        raise ValueError("Historical kept count mismatch")
    train_groups = {row["source_group"] for row in kept if row["preferred_split"] == "train"}
    val_groups = {row["source_group"] for row in kept if row["preferred_split"] == "val"}
    if train_groups & val_groups:
        raise ValueError("Historical source groups overlap train and validation")

    base_rows = {split: normalize_base(read_csv(args.base.parent / base["splits"][split]["path"]))
                 for split in ("train", "val", "test")}
    historical_rows = {"train": [], "val": []}
    for row in kept:
        split = row["preferred_split"]
        historical_rows[split].append({
            "source": "historical_video",
            "fileName": f"{row['video_id']}_{int(float(row['timestamp_seconds']) * 1000):09d}ms",
            "video": row["video_id"], "colorPath": row["frame_path"], "grayPath": "",
        })

    outputs = {
        "train": base_rows["train"] + historical_rows["train"],
        "val": base_rows["val"] + historical_rows["val"],
        "test": base_rows["test"],
    }
    random.Random(args.seed).shuffle(outputs["train"])
    args.output.mkdir(parents=True)
    split_info = {}
    for split, rows in outputs.items():
        path = args.output / f"{split}.csv"
        write_split(path, rows)
        split_info[split] = {
            "path": path.name, "sha256": sha256(path), "count": len(rows),
            "videos": sorted({row["video"] for row in rows if row["source"] != "wikimedia_commons"}),
            "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        }

    candidate_root = Path(historical["source_candidates"])
    manifest = {
        "protocol": "vietnam_video_disjoint_wikimedia_historical_v3",
        "seed": args.seed, "base_protocol": str(args.base.resolve()),
        "base_protocol_sha256": sha256(args.base),
        "historical_manifest": str(args.historical.resolve()),
        "historical_manifest_sha256": sha256(args.historical),
        "training_rule": "V2 train plus curated historical train videos; no final-test video reuse",
        "validation_rule": "joint modern-video and source-group-disjoint historical validation",
        "sources": {**base["sources"],
                    "historical_video": {"root": str(candidate_root.resolve()),
                                         "manual_review_sha256": historical["manual_review_sha256"]}},
        "splits": split_info, "total_count": sum(info["count"] for info in split_info.values()),
        "frozen_historical_final_test_reused": False,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
