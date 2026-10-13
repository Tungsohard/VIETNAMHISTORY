"""Apply the completed contact-sheet visual review to manual_review.csv."""
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "historical_vietnam_images"
REVIEW_PATH = DATA_ROOT / "manual_review.csv"
DECISIONS_PATH = DATA_ROOT / "visual_review_rejects.json"
BACKUP_PATH = DATA_ROOT / "manual_review.before_visual_review.csv"


def main():
    with DECISIONS_PATH.open(encoding="utf-8") as handle:
        review_log = json.load(handle)
    with REVIEW_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or "review_index" not in fieldnames:
        raise ValueError("manual_review.csv is missing review_index")
    expected_count = len(rows)
    if review_log.get("reviewed_through_index") != expected_count:
        raise ValueError(
            "Visual review is incomplete: "
            f"reviewed={review_log.get('reviewed_through_index')}, rows={expected_count}"
        )

    reject_indices = [
        int(index)
        for indices in review_log.get("batches", {}).values()
        for index in indices
    ]
    reject_set = set(reject_indices)
    if len(reject_set) != len(reject_indices):
        raise ValueError("Duplicate reject indices found in visual review log")
    invalid = sorted(index for index in reject_set if not 1 <= index <= expected_count)
    if invalid:
        raise ValueError(f"Reject indices outside CSV range: {invalid}")

    csv_indices = [int(row["review_index"]) for row in rows]
    if csv_indices != list(range(1, expected_count + 1)):
        raise ValueError("manual_review.csv review_index values are not contiguous and ordered")

    if not BACKUP_PATH.exists():
        shutil.copy2(REVIEW_PATH, BACKUP_PATH)

    for row in rows:
        index = int(row["review_index"])
        rejected = index in reject_set
        row["decision"] = "reject" if rejected else "keep"
        row["notes"] = (
            "visual_review_contact_sheet_reject"
            if rejected
            else "visual_review_contact_sheet_keep"
        )

    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "reviewed": expected_count,
        "kept": expected_count - len(reject_set),
        "rejected": len(reject_set),
        "backup": str(BACKUP_PATH),
        "output": str(REVIEW_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
