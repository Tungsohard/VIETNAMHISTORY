"""Freeze manually approved historical-video frames as an immutable external test protocol."""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parents[2]
TEST_FIELDS = ["fileName", "video", "colorPath", "grayPath"]
ATTRIBUTION_FIELDS = ["frame_path", "frame_sha256", "video_id", "timestamp_seconds",
                      "source_title", "source_page_url", "author", "license", "license_url"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_candidates")
    parser.add_argument("--videos", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_videos")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "historical_vietnam_final_test_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Final test already exists: {args.output}")

    review_path = args.candidates / "manual_review.csv"
    rows = read_csv(review_path)
    selected = [row for row in rows if row["manual_decision"] == "keep"]
    if any(row["manual_decision"] not in ("keep", "exclude") for row in rows):
        raise ValueError("Every candidate must have a manual decision")
    counts = Counter(row["video_id"] for row in selected)
    if len(counts) < 2 or len(set(counts.values())) != 1:
        raise ValueError(f"Final test must be balanced across at least two videos: {dict(counts)}")

    source_manifest_path = args.videos / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    video_metadata = {row["id"]: row for row in source_manifest["videos"]}
    args.output.mkdir(parents=True)
    test_rows = []
    attribution_rows = []
    for row in selected:
        source = args.candidates / row["frame_path"]
        if sha256(source) != row["frame_sha256"]:
            raise ValueError(f"Candidate frame fingerprint mismatch: {source}")
        relative = Path("images") / row["video_id"] / source.name
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_hash = sha256(destination)
        if copied_hash != row["frame_sha256"]:
            raise RuntimeError(f"Frame changed while copying: {source}")
        metadata = video_metadata[row["video_id"]]
        test_rows.append({"fileName": destination.stem, "video": row["video_id"],
                          "colorPath": relative.as_posix(), "grayPath": ""})
        attribution_rows.append({"frame_path": relative.as_posix(), "frame_sha256": copied_hash,
                                 "video_id": row["video_id"],
                                 "timestamp_seconds": row["timestamp_seconds"],
                                 "source_title": metadata["title"],
                                 "source_page_url": metadata["page_url"], "author": metadata["author"],
                                 "license": metadata["license"], "license_url": metadata["license_url"]})

    test_csv = args.output / "test.csv"
    attribution_csv = args.output / "attribution.csv"
    write_csv(test_csv, TEST_FIELDS, test_rows)
    write_csv(attribution_csv, ATTRIBUTION_FIELDS, attribution_rows)
    manifest = {
        "protocol": "historical_vietnam_final_test_v1",
        "status": "FROZEN_DO_NOT_USE_FOR_TRAINING_OR_MODEL_SELECTION",
        "input": "CIE-L derived from each original color frame; RGB frame is ground truth",
        "selection": "fixed 15-second sampling followed by manual exposure, diversity and content review",
        "source_video_manifest": str(source_manifest_path.resolve()),
        "source_video_manifest_sha256": sha256(source_manifest_path),
        "manual_review": str(review_path.resolve()), "manual_review_sha256": sha256(review_path),
        "test": {"path": test_csv.name, "sha256": sha256(test_csv), "count": len(test_rows),
                 "videos": sorted(counts), "count_per_video": dict(sorted(counts.items()))},
        "attribution": {"path": attribution_csv.name, "sha256": sha256(attribution_csv)},
        "frame_sha256": {row["colorPath"]: sha256(args.output / row["colorPath"]) for row in test_rows},
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"protocol": manifest["protocol"], "status": manifest["status"],
                      **manifest["test"]}, indent=2))


if __name__ == "__main__":
    main()
