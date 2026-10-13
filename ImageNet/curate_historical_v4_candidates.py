"""Record reproducible manual and image-quality decisions for V4 historical frames."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent
VIDEO_EXCLUSIONS = {
    "foulke_ammo_depot_1967": "reject_severe_magenta_monochrome_cast",
    "korean_landing_1969": "reject_black_and_white_footage",
}
TIME_EXCLUSIONS = {
    "foulke_aircraft_damage_1972": [(0, 13, "reject_slate_or_leader"),
                                      (192, 213, "reject_sensitive_or_ambiguous_closeup")],
    "foulke_ics_redeployment": [(0, 9, "reject_slate_or_leader"),
                                  (168, 177, "reject_slate_or_leader"),
                                  (188, 197, "reject_text_card"),
                                  (232, 241, "reject_text_card"),
                                  (336, 345, "reject_slate_or_leader"),
                                  (476, 485, "reject_slate_or_leader")],
    "foulke_inauguration_thieu": [(336, 345, "reject_slate_or_leader")],
    "operation_san_angelo_1968": [(176, 185, "reject_slate_or_leader"),
                                    (916, 933, "reject_slate_or_leader")],
}


def interval_reason(video_id, timestamp, exclusions=None):
    exclusions = TIME_EXCLUSIONS if exclusions is None else exclusions
    for start, end, reason in exclusions.get(video_id, []):
        if start <= timestamp < end:
            return reason
    return ""


def quality_reason(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mean, contrast = float(luminance.mean()), float(luminance.std())
    dark_fraction = float((luminance < 10).mean())
    bright_fraction = float((luminance > 245).mean())
    if mean < 24 or dark_fraction > 0.65:
        return "reject_too_dark", mean, contrast, dark_fraction, bright_fraction
    if mean > 238 or bright_fraction > 0.70:
        return "reject_overexposed", mean, contrast, dark_fraction, bright_fraction
    if contrast < 11:
        return "reject_low_information", mean, contrast, dark_fraction, bright_fraction
    return "keep", mean, contrast, dark_fraction, bright_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "historical_vietnam_v4_candidates_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "historical_vietnam_v4_curated")
    parser.add_argument("--policy", type=Path,
                        help="Optional reproducible manual-review exclusions")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Curated output already exists: {args.output}")

    source_manifest_path = args.input / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("protocol") != "historical_vietnam_adaptation_candidates_v1":
        raise ValueError("Expected historical adaptation candidates")
    candidate_path = args.input / "candidates.csv"
    if sha256(candidate_path) != source_manifest["candidates_csv_sha256"]:
        raise ValueError("Candidate CSV fingerprint mismatch")
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    video_exclusions = VIDEO_EXCLUSIONS
    time_exclusions = TIME_EXCLUSIONS
    policy_sha256 = None
    if args.policy:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if policy.get("protocol") != "historical_manual_review_policy_v1":
            raise ValueError("Unexpected manual-review policy")
        video_exclusions = policy.get("video_exclusions", {})
        time_exclusions = policy.get("time_exclusions", {})
        policy_sha256 = sha256(args.policy)

    reviewed = []
    for row in rows:
        frame_path = args.input / row["frame_path"]
        if sha256(frame_path) != row["frame_sha256"]:
            raise ValueError(f"Frame fingerprint mismatch: {frame_path}")
        reason = video_exclusions.get(row["video_id"], "")
        if not reason:
            reason = interval_reason(row["video_id"], float(row["timestamp_seconds"]), time_exclusions)
        quality, mean, contrast, dark_fraction, bright_fraction = quality_reason(frame_path)
        if not reason:
            reason = quality
        reviewed.append({**row, "manual_decision": "keep" if reason == "keep" else "reject",
                         "review_reason": reason, "luminance_mean": round(mean, 4),
                         "luminance_std": round(contrast, 4),
                         "dark_fraction": round(dark_fraction, 6),
                         "bright_fraction": round(bright_fraction, 6)})

    args.output.mkdir(parents=True)
    review_path = args.output / "manual_review.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    kept = [row for row in reviewed if row["manual_decision"] == "keep"]
    per_video = {}
    for video in source_manifest["videos"]:
        video_rows = [row for row in reviewed if row["video_id"] == video["id"]]
        per_video[video["id"]] = {
            "preferred_split": video["preferred_split"],
            "candidate_count": len(video_rows),
            "kept_count": sum(row["manual_decision"] == "keep" for row in video_rows),
        }
    manifest = {
        "protocol": "historical_vietnam_adaptation_curated_v1",
        "source_candidates": str(args.input.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "review_method": "manual contact-sheet review plus deterministic luminance/contrast filters",
        "manual_review_policy": str(args.policy.resolve()) if args.policy else None,
        "manual_review_policy_sha256": policy_sha256,
        "candidate_count": len(reviewed), "kept_count": len(kept),
        "rejected_count": len(reviewed) - len(kept),
        "decision_counts": dict(sorted(Counter(row["review_reason"] for row in reviewed).items())),
        "split_counts": {split: sum(row["preferred_split"] == split for row in kept)
                         for split in ("train", "val")},
        "videos": per_video, "manual_review_csv": review_path.name,
        "manual_review_sha256": sha256(review_path),
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
