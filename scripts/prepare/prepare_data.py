"""Prepare fixed ImageNet64 train/validation splits and official validation test."""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256


ROOT = Path(__file__).resolve().parents[2]


def batch_number(path):
    return int(re.search(r"(\d+)$", path.name).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "sources" / "imagenet64")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--official-validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.02 <= args.official_validation_fraction <= 0.5:
        parser.error("official-validation-fraction must be between 0.02 and 0.5")
    output = args.processed.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "protocol_manifest_v3.json"
    if destination.exists():
        raise FileExistsError(f"Protocol already frozen: {destination}")

    sources = sorted(args.raw.glob("train_part*/train_data_batch_*"), key=batch_number)
    if len(sources) != 10:
        raise FileNotFoundError(f"Expected 10 ImageNet64 batches, found {len(sources)}")
    old_manifest_path = output / "manifest.json"
    old = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.exists() else None
    rng = np.random.default_rng(args.seed)
    train_rows, shards = [], []
    for shard_id, source in enumerate(sources):
        print(f"Checking {source.name}", flush=True)
        source_hash = sha256(source)
        if old:
            matches = [item for item in old["shards"] if item["source_sha256"] == source_hash]
            if len(matches) != 1:
                raise ValueError(f"Cannot match existing processed shard for {source}")
            shard_path = output / matches[0]["path"]
        else:
            shard_path = output / f"shard_{shard_id:02d}.npy"
        with source.open("rb") as handle:
            content = pickle.load(handle, encoding="latin1")
        data = content.get("data", content.get(b"data"))
        labels = np.asarray(content.get("labels", content.get(b"labels")))
        if data.shape != (len(labels), 12288):
            raise ValueError(f"Invalid batch shape: {source}")
        if not shard_path.exists():
            np.save(shard_path, data.astype(np.uint8, copy=False))
        processed = np.load(shard_path, mmap_mode="r")
        if processed.shape != data.shape or processed.dtype != np.uint8:
            raise ValueError(f"Invalid processed shard: {shard_path}")
        train_rows.extend((shard_id, int(row)) for row in range(len(labels)))
        shards.append({"path": shard_path.name, "count": len(data),
                       "raw_path": str(source.resolve()), "raw_sha256": source_hash})
        del content, data, processed

    rng.shuffle(train_rows)
    split_files = {}
    train_path = output / "protocol_train_v3.npy"
    np.save(train_path, np.asarray(train_rows, dtype=np.int32))
    split_files["train"] = {"path": train_path.name, "count": len(train_rows),
                            "sha256": sha256(train_path), "source": "train_shards"}

    official_npz = args.raw / "official_val" / "val_data.npz"
    if not official_npz.exists():
        raise FileNotFoundError(official_npz)
    print("Converting official ImageNet64 validation", flush=True)
    with np.load(official_npz) as archive:
        official_data = archive["data"]
        official_labels = archive["labels"]
        if official_data.shape != (50000, 12288) or len(official_labels) != 50000:
            raise ValueError("Unexpected official validation shape")
        official_path = output / "official_val.npy"
        if not official_path.exists():
            np.save(official_path, official_data.astype(np.uint8, copy=False))
        validation_rows, test_rows = [], []
        for label in np.unique(official_labels):
            indices = np.flatnonzero(official_labels == label)
            rng.shuffle(indices)
            count = max(1, round(len(indices) * args.official_validation_fraction))
            validation_rows.extend(map(int, indices[:count]))
            test_rows.extend(map(int, indices[count:]))
    rng.shuffle(validation_rows)
    rng.shuffle(test_rows)
    for name, rows in (("val", validation_rows), ("test", test_rows)):
        path = output / f"official_{name}_v3.npy"
        np.save(path, np.asarray(rows, dtype=np.int32))
        split_files[name] = {"path": path.name, "count": len(rows), "sha256": sha256(path),
                            "source": "official_validation"}
    manifest = {
        "version": 3,
        "dataset": "Downsampled ImageNet 64x64",
        "seed": args.seed,
        "official_validation_fraction": args.official_validation_fraction,
        "protocol": "all ImageNet train -> training; stratified official validation -> model-selection validation/final test",
        "shards": shards,
        "splits": split_files,
        "official_data": {"path": official_path.name, "count": 50000,
                          "sha256": sha256(official_path), "raw_path": str(official_npz.resolve()),
                          "raw_sha256": sha256(official_npz)},
    }
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Train={len(train_rows):,} Val={len(validation_rows):,} Test={len(test_rows):,}")
    print(destination)


if __name__ == "__main__":
    main()
