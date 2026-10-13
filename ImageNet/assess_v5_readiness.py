"""Audit whether independent historical data is sufficient to start V5."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from colorization.data import sha256
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def curated_group_counts(manifest_paths):
    groups = {split: set() for split in ("train", "val")}
    frames = Counter()
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        review_path = manifest_path.parent / manifest["manual_review_csv"]
        if sha256(review_path) != manifest["manual_review_sha256"]:
            raise ValueError("Curated historical review fingerprint mismatch")
        kept = [row for row in read_rows(review_path) if row["manual_decision"] == "keep"]
        for row in kept:
            split = row["preferred_split"]
            if split not in groups:
                raise ValueError(f"Unexpected historical split: {split}")
            groups[split].add(row["source_group"])
            frames[split] += 1
    if groups["train"] & groups["val"]:
        raise ValueError("Historical source groups overlap train and validation")
    return groups, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, nargs="+", default=[
        ROOT / "data" / "historical_vietnam_v4_curated" / "manifest.json"
    ])
    parser.add_argument("--new-final-protocol", type=Path)
    parser.add_argument("--min-train-groups", type=int, default=6)
    parser.add_argument("--min-val-groups", type=int, default=3)
    parser.add_argument("--min-final-groups", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "v5_readiness.json")
    args = parser.parse_args()

    groups, frames = curated_group_counts(args.curated)
    final_groups = set()
    final_protocol_valid = False
    if args.new_final_protocol:
        final = json.loads(args.new_final_protocol.read_text(encoding="utf-8"))
        final_groups = set(final.get("test", {}).get("videos", []))
        final_protocol_valid = (
            final.get("status") == "FROZEN_DO_NOT_USE_FOR_TRAINING_OR_MODEL_SELECTION"
            and not (final_groups & groups["train"])
            and not (final_groups & groups["val"])
        )

    checks = {
        "enough_historical_train_source_groups": len(groups["train"]) >= args.min_train_groups,
        "enough_historical_validation_source_groups": len(groups["val"]) >= args.min_val_groups,
        "new_untouched_final_protocol_present": final_protocol_valid,
        "enough_new_final_source_groups": final_protocol_valid and len(final_groups) >= args.min_final_groups,
    }
    report = {
        "status": "READY" if all(checks.values()) else "BLOCKED_DATA",
        "checks": checks,
        "requirements": {
            "minimum_train_source_groups": args.min_train_groups,
            "minimum_validation_source_groups": args.min_val_groups,
            "minimum_new_final_source_groups": args.min_final_groups,
        },
        "current": {
            "historical_train_source_groups": sorted(groups["train"]),
            "historical_train_group_count": len(groups["train"]),
            "historical_train_frame_count": frames["train"],
            "historical_validation_source_groups": sorted(groups["val"]),
            "historical_validation_group_count": len(groups["val"]),
            "historical_validation_frame_count": frames["val"],
            "new_final_source_groups": sorted(final_groups),
        },
        "policy": {
            "previous_48_frame_benchmark_role": "development_only",
            "allow_v5_training_when_blocked": False,
            "selection_rule": "source-group-balanced validation only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
