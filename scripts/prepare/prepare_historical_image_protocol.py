"""Create a manually reviewed, collection-disjoint historical still-image protocol."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from colorization.image_dataset import infer_collection_id, infer_content_category, manipulation_reason
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parents[2]
REVIEW_FIELDS = [
    "review_index", "decision", "notes", "content_category", "collection_id",
    "local_path", "title", "page_url", "license", "author", "source_category",
    "colorfulness", "dhash", "local_sha256",
]
PROTOCOL_FIELDS = [
    "source", "fileName", "video", "colorPath", "grayPath", "content_category",
    "collection_id", "page_url", "license",
]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def initialize_review(input_root, rows, review_path, sheet_dir, columns=4):
    review_rows = []
    for index, row in enumerate(rows, 1):
        review_rows.append({
            **row,
            "review_index": index,
            "decision": "",
            "notes": "suspected_manipulated_color" if manipulation_reason(row) else "",
            "content_category": row.get("content_category") or infer_content_category(row),
            "collection_id": row.get("collection_id") or infer_collection_id(row),
        })
    write_csv(review_path, review_rows, REVIEW_FIELDS)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cell_width, image_height, text_height = 280, 210, 58
    per_sheet = columns * 5
    for page, start in enumerate(range(0, len(review_rows), per_sheet), 1):
        selected = review_rows[start:start + per_sheet]
        rows_count = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, rows_count * (image_height + text_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(selected):
            with Image.open(input_root / row["local_path"]) as source:
                image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"),
                                         (cell_width, image_height))
            x = (offset % columns) * cell_width
            y = (offset // columns) * (image_height + text_height)
            sheet.paste(image, (x + (cell_width - image.width) // 2, y + (image_height - image.height) // 2))
            draw.text((x + 5, y + image_height + 3),
                      f"#{row['review_index']}  {row['content_category']}", fill="black")
            draw.text((x + 5, y + image_height + 23), row["title"][:42], fill="black")
        sheet.save(sheet_dir / f"review_{page:03d}.jpg", quality=92)
    return review_rows


def split_by_collection(rows, validation_ratio, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["collection_id"], []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two independent collections are required")
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    target = max(1, round(len(rows) * validation_ratio))
    validation, training = [], []
    for collection, selected in items:
        if len(validation) < target and len(items) > 1:
            validation.extend(selected)
        else:
            training.extend(selected)
    if not training:
        moved_collection = validation[-1]["collection_id"]
        training = [row for row in validation if row["collection_id"] == moved_collection]
        validation = [row for row in validation if row["collection_id"] != moved_collection]
    return training, validation


def protocol_row(row):
    return {
        "source": "historical_wikimedia_still",
        "fileName": Path(row["local_path"]).stem,
        "video": row["collection_id"],
        "colorPath": row["local_path"],
        "grayPath": "",
        "content_category": row["content_category"],
        "collection_id": row["collection_id"],
        "page_url": row["page_url"],
        "license": row["license"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "sources" / "historical_vietnam_images")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "protocols" / "historical_vietnam_image_protocol_v1")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 0.5:
        parser.error("--validation-ratio must be between 0 and 0.5")
    accepted_path = args.input / "accepted.jsonl"
    if not accepted_path.is_file():
        raise FileNotFoundError(f"Missing collector journal: {accepted_path}")
    review_path = args.review or args.input / "manual_review.csv"
    accepted = read_jsonl(accepted_path)
    if not accepted:
        raise ValueError("No collected images found")
    if not review_path.exists():
        initialize_review(args.input, accepted, review_path, args.input / "review_sheets")
        print(f"Review template created: {review_path}")
        print("Fill every decision with keep or reject, then run this command again.")
        return

    review = read_csv(review_path)
    decisions = {row["decision"].strip().lower() for row in review}
    invalid = decisions - {"keep", "reject"}
    if invalid:
        raise ValueError(f"Every review decision must be keep or reject; found: {sorted(invalid)}")
    kept = [row for row in review if row["decision"].strip().lower() == "keep"]
    if not kept:
        raise ValueError("Manual review approved no images")
    accepted_by_path = {row["local_path"]: row for row in accepted}
    if len(review) != len(accepted) or set(accepted_by_path) != {row["local_path"] for row in review}:
        raise ValueError("Manual review rows do not match accepted.jsonl")
    for row in kept:
        source = accepted_by_path[row["local_path"]]
        image_path = args.input / row["local_path"]
        if not image_path.is_file() or sha256(image_path) != source["local_sha256"]:
            raise ValueError(f"Missing or modified reviewed image: {image_path}")
        if manipulation_reason({**source, **row}):
            raise ValueError(f"Manipulated/colorized metadata cannot be approved: {row['title']}")

    training, validation = split_by_collection(kept, args.validation_ratio, args.seed)
    if args.output.exists():
        raise FileExistsError(f"Protocol output already exists: {args.output}")
    args.output.mkdir(parents=True)
    split_rows = {"train": [protocol_row(row) for row in training],
                  "val": [protocol_row(row) for row in validation]}
    split_info = {}
    for split, rows in split_rows.items():
        path = args.output / f"{split}.csv"
        write_csv(path, rows, PROTOCOL_FIELDS)
        split_info[split] = {
            "path": path.name,
            "sha256": sha256(path),
            "count": len(rows),
            "collection_count": len({row["collection_id"] for row in rows}),
            "content_counts": dict(Counter(row["content_category"] for row in rows)),
        }
    overlap = ({row["collection_id"] for row in split_rows["train"]}
               & {row["collection_id"] for row in split_rows["val"]})
    if overlap:
        raise RuntimeError(f"Collection leakage detected: {sorted(overlap)}")
    manifest = {
        "protocol": "historical_vietnam_still_collection_disjoint_v1",
        "role": "training and model-selection validation only; not a final test",
        "seed": args.seed,
        "validation_ratio_target": args.validation_ratio,
        "input_root": str(args.input.resolve()),
        "accepted_journal_sha256": sha256(accepted_path),
        "manual_review_sha256": sha256(review_path),
        "approved_count": len(kept),
        "rejected_count": len(review) - len(kept),
        "collection_leakage_count": 0,
        "splits": split_info,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
