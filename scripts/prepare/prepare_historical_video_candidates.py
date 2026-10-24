"""Extract sparse color-frame candidates from downloaded historical videos for manual review."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from collect_wikimedia_vietnam import difference_hash, image_colorfulness
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parents[2]


def write_contact_sheet(root, rows, path, columns=4):
    thumb = (256, 180)
    cell_height = 210
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], max(1, row_count) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(root / row["frame_path"]) as image:
            image = ImageOps.contain(image.convert("RGB"), thumb)
            left = index % columns * thumb[0] + (thumb[0] - image.width) // 2
            top = index // columns * cell_height + (thumb[1] - image.height) // 2
            sheet.paste(image, (left, top))
        draw.text((index % columns * thumb[0] + 4, index // columns * cell_height + 184),
                  f"{row['video_id']}  t={float(row['timestamp_seconds']):.1f}s", fill="black")
    sheet.save(path, "JPEG", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_videos")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_candidates")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--start-seconds", type=float, default=5.0)
    parser.add_argument("--end-margin-seconds", type=float, default=5.0)
    parser.add_argument("--min-colorfulness", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Candidate output already exists: {args.output}")

    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("downloaded_count") != manifest.get("video_count"):
        raise ValueError("Not all configured videos have been downloaded")
    args.output.mkdir(parents=True)
    rows = []
    seen_hashes = []
    video_info = []
    for video in manifest["videos"]:
        path = args.input / video["local_path"]
        if sha256(path) != video["local_sha256"]:
            raise ValueError(f"Video fingerprint mismatch: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot decode {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        frame_dir = args.output / "frames" / video["id"]
        frame_dir.mkdir(parents=True)
        accepted = rejected_gray = rejected_duplicate = 0
        timestamp = args.start_seconds
        while timestamp < duration - args.end_margin_seconds and accepted < args.max_frames_per_video:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, bgr = capture.read()
            if not ok:
                timestamp += args.interval_seconds
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            colorfulness = image_colorfulness(image)
            dhash = int(difference_hash(image), 16)
            if colorfulness < args.min_colorfulness:
                rejected_gray += 1
            elif any((dhash ^ previous).bit_count() <= 3 for previous in seen_hashes):
                rejected_duplicate += 1
            else:
                destination = frame_dir / f"frame_{accepted + 1:04d}_{int(timestamp * 1000):09d}ms.jpg"
                image.save(destination, "JPEG", quality=95)
                row = {"video_id": video["id"], "source_title": video["title"],
                       "preferred_split": video.get("preferred_split", ""),
                       "source_group": video.get("source_group", video["id"]),
                       "timestamp_seconds": round(timestamp, 3),
                       "frame_path": destination.relative_to(args.output).as_posix(),
                       "frame_sha256": sha256(destination), "colorfulness": colorfulness,
                       "dhash": f"{dhash:016x}", "manual_decision": "pending"}
                rows.append(row)
                seen_hashes.append(dhash)
                accepted += 1
            timestamp += args.interval_seconds
        capture.release()
        video_info.append({"id": video["id"],
                           "preferred_split": video.get("preferred_split", ""),
                           "source_group": video.get("source_group", video["id"]),
                           "duration_seconds": duration, "fps": fps,
                           "decoded_frame_count": frame_count, "candidate_count": accepted,
                           "low_color_rejected": rejected_gray,
                           "near_duplicate_rejected": rejected_duplicate})

    csv_path = args.output / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sheet = args.output / "contact_sheet.jpg"
    write_contact_sheet(args.output, rows, sheet)
    for video_id in sorted({row["video_id"] for row in rows}):
        video_rows = [row for row in rows if row["video_id"] == video_id]
        # Keep review sheets readable while covering the whole timeline.
        if len(video_rows) > 48:
            indices = np.linspace(0, len(video_rows) - 1, 48, dtype=int)
            video_rows = [video_rows[index] for index in indices]
        write_contact_sheet(args.output, video_rows, args.output / f"contact_{video_id}.jpg")
    adaptation = any(video.get("preferred_split") for video in manifest["videos"])
    output_manifest = {"protocol": ("historical_vietnam_adaptation_candidates_v1" if adaptation
                                    else "historical_vietnam_final_test_candidates_v1"),
                       "source_manifest": str((args.input / "manifest.json").resolve()),
                       "source_manifest_sha256": sha256(args.input / "manifest.json"),
                       "selection": "fixed temporal interval, minimum colorfulness, near-duplicate dHash filter",
                       "interval_seconds": args.interval_seconds, "min_colorfulness": args.min_colorfulness,
                       "candidate_count": len(rows), "videos": video_info,
                       "candidates_csv_sha256": sha256(csv_path), "manual_review_complete": False}
    write_json(args.output / "manifest.json", output_manifest)
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
