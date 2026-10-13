"""Create a deterministic video-disjoint Vietnam adaptation protocol."""
import argparse
import csv
import json
import random
from pathlib import Path

from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent
FIELDS = ["fileName", "video", "colorPath", "grayPath"]


def write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "dataset" / "VietNam_dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "vietnam_protocol_v1")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Protocol already exists: {args.output}")

    color_root = args.source / "Color_image"
    gray_root = args.source / "Grayscale_image"
    videos = sorted(path.name for path in color_root.iterdir() if path.is_dir())
    if len(videos) < 4:
        raise ValueError("At least four videos are required for a 2/1/1 group split")
    shuffled = videos.copy()
    random.Random(args.seed).shuffle(shuffled)
    assignments = {
        "train": sorted(shuffled[:-2]),
        "val": [shuffled[-2]],
        "test": [shuffled[-1]],
    }

    args.output.mkdir(parents=True)
    split_info = {}
    all_ids = set()
    for split, split_videos in assignments.items():
        rows = []
        for video in split_videos:
            color_files = sorted((color_root / video).glob("*.jpg"))
            gray_files = {path.name: path for path in (gray_root / video).glob("*.jpg")}
            if not color_files or {path.name for path in color_files} != set(gray_files):
                raise ValueError(f"Color/grayscale pairs do not match for {video}")
            for color_path in color_files:
                relative_color = color_path.relative_to(args.source).as_posix()
                relative_gray = gray_files[color_path.name].relative_to(args.source).as_posix()
                if relative_color in all_ids:
                    raise ValueError(f"Duplicate image: {relative_color}")
                all_ids.add(relative_color)
                rows.append({"fileName": color_path.stem, "video": video,
                             "colorPath": relative_color, "grayPath": relative_gray})
        csv_path = args.output / f"{split}.csv"
        write_split(csv_path, rows)
        split_info[split] = {"path": csv_path.name, "sha256": sha256(csv_path),
                             "count": len(rows), "videos": split_videos}

    source_csv = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        path = args.source / name
        if path.exists():
            source_csv[name] = sha256(path)
    manifest = {
        "protocol": "vietnam_video_disjoint_v1",
        "seed": args.seed,
        "assignment_rule": "sorted video names; Python random.Random(seed).shuffle; all but last two train, penultimate val, last test",
        "source_root": str(args.source.resolve()),
        "source_csv_sha256": source_csv,
        "input": "CIE-L derived from paired color target; grayPath retained for later real-grayscale evaluation",
        "splits": split_info,
        "total_count": len(all_ids),
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
