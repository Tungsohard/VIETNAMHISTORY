"""Create auditable review decisions for historical Vietnam still photographs."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
REJECT_CATEGORY_TERMS = (
    " maps of ", "topographical maps", "drawings", "illustrations", "paintings",
    "posters", "postcards", "stamps", "books from", "newspapers", "magazines",
    "documents", "diagrams", "charts", "logos", "coats of arms", "banknotes",
)
REJECT_TITLE_TERMS = (
    " map ", " map.", " map,", "topographic map", "administrative divisions",
    "ethnolinguistic groups", "boundary map", "location map", "locator map",
)
REVIEW_TERMS = (
    "dead", "corpse", "casualt", "wounded", "killed", "execution", "self-immolation",
    "body bag", "massacre", "victim", "graphic", "scan", "document", "newspaper",
    "magazine", "book cover", "poster", "drawing", "painting", "illustration",
    "diagram", "flag", "emblem", "insignia", "patch", "medal", "certificate",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value):
    return " " + re.sub(r"\s+", " ", str(value).lower()) + " "


def propose(row, metadata, root):
    category = normalized(row["source_category"])
    title = normalized(row["title"])
    text = normalized(" ".join((row["title"], row["source_category"],
                                metadata.get("description", ""), metadata.get("credit", ""))))
    if any(term in category for term in REJECT_CATEGORY_TERMS):
        return "reject", "non_photograph_source_category"
    if any(term in title for term in REJECT_TITLE_TERMS):
        return "reject", "non_photograph_title"
    if any(term in text for term in REVIEW_TERMS):
        return "review", "sensitive_or_ambiguous_metadata"
    with Image.open(root / row["local_path"]) as image:
        ratio = image.width / image.height
    if ratio > 3 or ratio < 1 / 3:
        return "review", "extreme_aspect_ratio"
    return "keep", "automatic_photo_filters_passed"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sheets(root, rows, output, per_sheet=20):
    output.mkdir(parents=True, exist_ok=True)
    columns, cell_width, image_height, label_height = 4, 280, 210, 60
    for page, start in enumerate(range(0, len(rows), per_sheet), 1):
        selected = rows[start:start + per_sheet]
        row_count = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, row_count * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(selected):
            with Image.open(root / row["local_path"]) as source:
                image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"),
                                         (cell_width, image_height))
            x = (offset % columns) * cell_width
            y = (offset // columns) * (image_height + label_height)
            sheet.paste(image, (x + (cell_width - image.width) // 2,
                                y + (image_height - image.height) // 2))
            draw.text((x + 4, y + image_height + 2),
                      f"#{row['review_index']} {row['proposal_reason']}", fill="black")
            draw.text((x + 4, y + image_height + 21), row["title"][:43], fill="black")
        sheet.save(output / f"manual_candidates_{page:03d}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "historical_vietnam_images")
    parser.add_argument("--apply", action="store_true",
                        help="Write keep/reject proposals to manual_review.csv; review items remain blank")
    args = parser.parse_args()
    review_path = args.input / "manual_review.csv"
    rows = read_csv(review_path)
    metadata = {row["local_path"]: row for row in read_jsonl(args.input / "accepted.jsonl")}
    proposals = []
    for row in rows:
        decision, reason = propose(row, metadata[row["local_path"]], args.input)
        proposals.append({**row, "proposal": decision, "proposal_reason": reason})
    proposal_path = args.input / "review_proposals.csv"
    write_csv(proposal_path, proposals, list(proposals[0]))
    manual = [row for row in proposals if row["proposal"] == "review"]
    write_sheets(args.input, manual, args.input / "manual_candidate_sheets")
    if args.apply:
        for row, proposal in zip(rows, proposals):
            if proposal["proposal"] in {"keep", "reject"}:
                row["decision"] = proposal["proposal"]
                row["notes"] = proposal["proposal_reason"]
        backup = review_path.with_suffix(".before_triage.csv")
        if not backup.exists():
            backup.write_bytes(review_path.read_bytes())
        write_csv(review_path, rows, list(rows[0]))
    report = {"total": len(proposals), "counts": dict(Counter(row["proposal"] for row in proposals)),
              "applied": args.apply, "proposal_csv": str(proposal_path),
              "manual_candidate_sheets": str(args.input / "manual_candidate_sheets")}
    (args.input / "review_proposals.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
