"""Add the ViCoW historical films to the V12 modern protocol (V15), split by film.

ViCoW's own split is random per frame: 75% of its test frames have a training frame at
most two frames away in the same film, so it overstates quality. This protocol keeps
the project's film-disjoint split instead:

* train: *Những người viết huyền thoại* and *Vào Nam ra Bắc*, colour-corrected frames
  from ``data/sources/vicow_corrected`` (source ``vicow``);
* validation: *Mùi cỏ cháy*, original frames (source ``vicow_val``);
* test: *Hà Nội 12 ngày đêm*, original frames (source ``vicow_test``), never loaded by
  training.

Evaluation frames keep their original colours because they are the ground truth the
model is measured against. Modern validation/test splits are exactly those of V12.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256
from prepare_v8_craw_protocol import read_csv, write_csv


ROOT = Path(__file__).resolve().parents[2]
FILM_SPLIT = {"train": ("NhungNguoiVietLenHuyenThoai", "VaoNamRaBac"),
              "val": ("MuiCoChay",), "test": ("HaNoi12NgayDem",)}
SOURCE_OF_SPLIT = {"train": "vicow", "val": "vicow_val", "test": "vicow_test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_modern_military_v12" / "manifest.json")
    parser.add_argument("--vicow-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_protocol_v1" / "manifest.json")
    parser.add_argument("--vicow-root", type=Path, default=ROOT / "data" / "sources" / "vicow")
    parser.add_argument("--corrected-root", type=Path, default=ROOT / "data" / "sources" / "vicow_corrected")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "protocols" / "vietnam_modern_vicow_v15")
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"V15 protocol already exists: {manifest_path}")

    v12 = json.loads(args.v12_protocol.read_text(encoding="utf-8"))
    vicow = json.loads(args.vicow_protocol.read_text(encoding="utf-8"))
    rows = {split: read_csv(args.v12_protocol.parent / v12["splits"][split]["path"])
            for split in ("train", "val", "test")}
    vicow_rows = {split: read_csv(args.vicow_protocol.parent / vicow["splits"][split]["path"])
                  for split in ("train", "val", "test")}

    for split, films in FILM_SPLIT.items():
        found = {row["video"] for row in vicow_rows[split]}
        if found != set(films):
            raise ValueError(f"Unexpected ViCoW films in {split}: {sorted(found)}")
        root = args.corrected_root if split == "train" else args.vicow_root
        for row in vicow_rows[split]:
            if not (root / row["colorPath"]).is_file():
                raise FileNotFoundError(f"Missing ViCoW frame: {root / row['colorPath']}")
            rows[split].append({"source": SOURCE_OF_SPLIT[split], "fileName": row["fileName"],
                                "video": row["video"], "colorPath": row["colorPath"], "grayPath": ""})
    all_films = [set(films) for films in FILM_SPLIT.values()]
    assert not (all_films[0] & all_films[1]) and not (all_films[0] & all_films[2]) and not (all_films[1] & all_films[2])

    args.output.mkdir(parents=True, exist_ok=False)
    split_info = {}
    for split, split_rows in rows.items():
        path = args.output / f"{split}.csv"
        write_csv(path, split_rows)
        split_info[split] = {"path": path.name, "sha256": sha256(path), "count": len(split_rows),
                             "videos": sorted({row["video"] for row in split_rows}),
                             "source_counts": dict(sorted(Counter(row["source"] for row in split_rows).items()))}
    manifest = {
        "protocol": "vietnam_modern_vicow_v15",
        "base_protocol": str(args.v12_protocol.resolve()),
        "base_protocol_sha256": sha256(args.v12_protocol),
        "policy": {
            "vicow_split": "by film (the original ViCoW per-frame split leaks neighbouring frames)",
            "vicow_films": {split: list(films) for split, films in FILM_SPLIT.items()},
            "vicow_train_colour": "mildly corrected and sharpened (scripts/prepare/correct_vicow_colors.py)",
            "vicow_eval_colour": "original frames",
            "modern_validation_and_test": "unchanged from V12",
        },
        "splits": split_info,
        "total_count": sum(info["count"] for info in split_info.values()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["source_counts"] for k, v in split_info.items()}, indent=1))


if __name__ == "__main__":
    main()
