"""Audit a collected Wikimedia Vietnam image set without modifying it."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


REVIEW_CATEGORY_TERMS = ("drawings", "maps", "postcards", "genetic studies")


def write_contact_sheets(root, rows, output_stem, per_sheet=12):
    paths = []
    thumb_size = (300, 220)
    columns = 3
    cell_height = 270
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet), 1):
        selected = rows[start:start + per_sheet]
        row_count = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_size[0], row_count * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(selected):
            index = start + offset + 1
            with Image.open(root / row["local_path"]) as image:
                image = ImageOps.contain(image.convert("RGB"), thumb_size)
                left = (offset % columns) * thumb_size[0] + (thumb_size[0] - image.width) // 2
                top = (offset // columns) * cell_height + (thumb_size[1] - image.height) // 2
                sheet.paste(image, (left, top))
            text_left = (offset % columns) * thumb_size[0] + 5
            text_top = (offset // columns) * cell_height + thumb_size[1] + 4
            category = row["source_category"].removeprefix("Category:")
            draw.text((text_left, text_top), f"#{index} {category[:38]}", fill="black")
            draw.text((text_left, text_top + 18), row["review_reasons"], fill="black")
        path = output_stem.with_name(f"{output_stem.name}_review_{sheet_index}.jpg")
        sheet.save(path, "JPEG", quality=92)
        paths.append(path)
    return paths


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()

    accepted = load_jsonl(args.input / "accepted.jsonl")
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    issues = []
    review = []
    dimensions = []
    modes = Counter()

    for row in accepted:
        path = args.input / row["local_path"]
        reasons = []
        if not path.is_file():
            issues.append({"title": row["title"], "reason": "missing_file"})
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["local_sha256"]:
            issues.append({"title": row["title"], "reason": "sha256_mismatch"})
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                modes[image.mode] += 1
        except (OSError, ValueError) as exc:
            issues.append({"title": row["title"], "reason": f"decode_error: {exc}"})
            continue
        dimensions.append((width, height))
        ratio = width / height
        if ratio > 3 or ratio < 1 / 3:
            reasons.append("extreme_aspect_ratio")
        if any(term in row["source_category"].lower() for term in REVIEW_CATEGORY_TERMS):
            reasons.append("potential_non_photo_or_off_domain")
        if reasons:
            review.append({"local_path": row["local_path"], "title": row["title"],
                           "source_category": row["source_category"],
                           "width": width, "height": height,
                           "review_reasons": ";".join(reasons), "page_url": row["page_url"]})

    attribution_sha = hashlib.sha256((args.input / "attribution.csv").read_bytes()).hexdigest()
    report = {
        "input": str(args.input.resolve()),
        "accepted_count": len(accepted),
        "image_file_count": len(list((args.input / "images").glob("*"))),
        "decoded_count": len(dimensions),
        "integrity_issue_count": len(issues),
        "unique_titles": len({row["title"] for row in accepted}),
        "unique_commons_sha1": len({row["commons_sha1"] for row in accepted}),
        "unique_dhash": len({row["dhash"] for row in accepted}),
        "unique_local_sha256": len({row["local_sha256"] for row in accepted}),
        "image_modes": dict(modes),
        "review_candidate_count": len(review),
        "attribution_sha256_matches_manifest": attribution_sha == manifest["attribution_sha256"],
        "source_category_counts": dict(Counter(row["source_category"] for row in accepted)),
        "license_counts": dict(Counter(row["license"] for row in accepted)),
        "integrity_issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    review_path = args.output.with_name(args.output.stem + "_review.csv")
    for index, row in enumerate(review, 1):
        row["review_index"] = index
    with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["review_index", "local_path", "title", "source_category", "width", "height",
                  "review_reasons", "page_url"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps({key: report[key] for key in (
        "accepted_count", "image_file_count", "decoded_count", "integrity_issue_count",
        "unique_titles", "unique_commons_sha1", "unique_dhash", "unique_local_sha256",
        "image_modes", "review_candidate_count", "attribution_sha256_matches_manifest")}, indent=2))
    print(f"Report: {args.output}")
    print(f"Review list: {review_path}")
    if args.contact_sheets:
        for path in write_contact_sheets(args.input, review, args.output.with_suffix("")):
            print(f"Contact sheet: {path}")


if __name__ == "__main__":
    main()
