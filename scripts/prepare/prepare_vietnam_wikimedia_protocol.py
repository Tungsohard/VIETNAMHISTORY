"""Build a mixed Vietnam training protocol while preserving frozen validation/test splits."""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["source", "fileName", "video", "colorPath", "grayPath"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_protocol_v1" / "manifest.json")
    parser.add_argument("--curated-manifest", type=Path,
                        default=ROOT / "outputs" / "reports" / "wikimedia_vietnam_curated_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "vietnam_protocol_v2")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Protocol already exists: {args.output}")

    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    if base.get("protocol") != "vietnam_video_disjoint_v1":
        raise ValueError("Expected the video-disjoint Vietnam V1 protocol")
    curated = json.loads(args.curated_manifest.read_text(encoding="utf-8-sig"))
    curated_csv = ROOT / curated["curated_attribution"]
    if sha256(curated_csv) != curated["curated_attribution_sha256"]:
        raise ValueError("Curated Wikimedia attribution fingerprint mismatch")
    wikimedia_root = ROOT / "data" / "sources" / "wikimedia_vietnam"

    base_train_path = args.base_protocol.parent / base["splits"]["train"]["path"]
    if sha256(base_train_path) != base["splits"]["train"]["sha256"]:
        raise ValueError("Base Vietnam training split fingerprint mismatch")
    train_rows = [{"source": "vietnam_video", **row} for row in read_csv(base_train_path)]
    commons_rows = []
    for row in read_csv(curated_csv):
        image_path = wikimedia_root / row["local_path"]
        if not image_path.is_file() or sha256(image_path) != row["local_sha256"]:
            raise ValueError(f"Missing or modified Wikimedia image: {image_path}")
        commons_rows.append({"source": "wikimedia_commons", "fileName": image_path.stem,
                             "video": "Wikimedia/" + row["source_category"].removeprefix("Category:"),
                             "colorPath": row["local_path"], "grayPath": ""})
    if len(commons_rows) != curated["approved_count"]:
        raise ValueError("Curated Wikimedia count mismatch")
    train_rows.extend(commons_rows)
    random.Random(args.seed).shuffle(train_rows)

    args.output.mkdir(parents=True)
    train_path = args.output / "train.csv"
    write_csv(train_path, train_rows)
    split_info = {
        "train": {"path": train_path.name, "sha256": sha256(train_path), "count": len(train_rows),
                  "videos": base["splits"]["train"]["videos"],
                  "source_counts": dict(Counter(row["source"] for row in train_rows))}
    }
    for split in ("val", "test"):
        source = args.base_protocol.parent / base["splits"][split]["path"]
        if sha256(source) != base["splits"][split]["sha256"]:
            raise ValueError(f"Base Vietnam {split} fingerprint mismatch")
        destination = args.output / f"{split}.csv"
        shutil.copyfile(source, destination)
        if sha256(destination) != base["splits"][split]["sha256"]:
            raise RuntimeError(f"Frozen {split} split changed while copying")
        split_info[split] = {**base["splits"][split], "path": destination.name}

    manifest = {
        "protocol": "vietnam_video_disjoint_wikimedia_train_v2",
        "seed": args.seed,
        "base_protocol": str(args.base_protocol.resolve()),
        "base_protocol_sha256": sha256(args.base_protocol),
        "curated_manifest": str(args.curated_manifest.resolve()),
        "curated_manifest_sha256": sha256(args.curated_manifest),
        "training_rule": "shuffle all video-train frames and manually curated Wikimedia photographs",
        "evaluation_rule": "validation and development-test CSV files are byte-identical to protocol V1",
        "sources": {
            "vietnam_video": {"root": str((ROOT / "data" / "sources" / "vicow").resolve())},
            "wikimedia_commons": {"root": str(wikimedia_root.resolve()),
                                  "curated_attribution_sha256": curated["curated_attribution_sha256"]},
        },
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
