"""Combine the existing V3 protocol with reviewed historical still train/val splits."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["source", "fileName", "video", "colorPath", "grayPath"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_protocol_v3" / "manifest.json")
    parser.add_argument("--stills", type=Path,
                        default=ROOT / "data" / "protocols" / "historical_vietnam_image_protocol_v1" / "manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_historical_still_replay_v4")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Protocol output already exists: {args.output}")

    base = json.loads(args.base.read_text(encoding="utf-8"))
    if base.get("protocol") != "vietnam_video_disjoint_wikimedia_historical_v3":
        raise ValueError("Expected Vietnam protocol V3")
    stills = json.loads(args.stills.read_text(encoding="utf-8"))
    if stills.get("protocol") != "historical_vietnam_still_collection_disjoint_v1":
        raise ValueError("Expected collection-disjoint historical still protocol")
    if stills.get("collection_leakage_count") != 0:
        raise ValueError("Historical still protocol contains collection leakage")

    manifests = ((args.base, base), (args.stills, stills))
    for manifest_path, manifest in manifests:
        for split in ("train", "val"):
            path = manifest_path.parent / manifest["splits"][split]["path"]
            if sha256(path) != manifest["splits"][split]["sha256"]:
                raise ValueError(f"{manifest['protocol']} {split} fingerprint mismatch")
    base_test = args.base.parent / base["splits"]["test"]["path"]
    if sha256(base_test) != base["splits"]["test"]["sha256"]:
        raise ValueError("Base test fingerprint mismatch")

    base_rows = {split: read_csv(args.base.parent / base["splits"][split]["path"])
                 for split in ("train", "val", "test")}
    still_rows = {split: read_csv(args.stills.parent / stills["splits"][split]["path"])
                  for split in ("train", "val")}
    train_still_groups = {row["video"] for row in still_rows["train"]}
    val_still_groups = {row["video"] for row in still_rows["val"]}
    if train_still_groups & val_still_groups:
        raise ValueError("Historical still collections overlap train and validation")

    outputs = {
        "train": base_rows["train"] + still_rows["train"],
        "val": base_rows["val"] + still_rows["val"],
        "test": base_rows["test"],
    }
    random.Random(args.seed).shuffle(outputs["train"])
    args.output.mkdir(parents=True)
    split_info = {}
    for split, rows in outputs.items():
        path = args.output / f"{split}.csv"
        write_split(path, rows)
        split_info[split] = {
            "path": path.name,
            "sha256": sha256(path),
            "count": len(rows),
            "videos": sorted({row["video"] for row in rows}),
            "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        }

    manifest = {
        "protocol": "vietnam_historical_still_replay_v4",
        "seed": args.seed,
        "base_protocol": str(args.base.resolve()),
        "base_protocol_sha256": sha256(args.base),
        "still_protocol": str(args.stills.resolve()),
        "still_protocol_sha256": sha256(args.stills),
        "training_rule": "V3 train plus reviewed historical still train; V3 checkpoint replay",
        "validation_rule": "source-balanced V3 validation plus collection-disjoint still validation",
        "sources": {**base["sources"], "historical_wikimedia_still": {
            "root": stills["input_root"],
            "manual_review_sha256": stills["manual_review_sha256"],
        }},
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
        "frozen_historical_final_test_reused": False,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
