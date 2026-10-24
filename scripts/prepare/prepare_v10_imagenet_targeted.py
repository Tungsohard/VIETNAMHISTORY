"""Select ImageNet64 training images of military and Vietnam-relevant classes for V10.

V9 (modern Vietnamese photographs only) colours uniforms, tanks and artillery
brown because those objects are almost absent from its data. V10 replays the
ImageNet classes that contain them. Labels come from the raw Downsampled
ImageNet batches; label k maps to line k of ``map_clsloc.txt``.

Outputs (next to the v3 manifest):
- ``train_labels_v3.npy``: label of every row of every shard (cached).
- ``imagenet_targeted_v10.npy``: shuffled (shard, row) pairs of the chosen classes.
- ``imagenet_targeted_v10.json``: manifest readable by ``ImageNetDataset``.
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import sha256


ROOT = Path(__file__).resolve().parents[2]
CLASSES = {
    # military
    231: "warplane", 249: "half_track", 250: "tank", 264: "amphibian", 269: "jeep",
    539: "cannon", 540: "assault_rifle", 541: "rifle", 542: "projectile",
    778: "crash_helmet", 866: "military_uniform",
    # Vietnamese countryside, streets and landmarks
    108: "ox", 162: "water_buffalo", 989: "thatch", 359: "cliff", 360: "valley",
    361: "alp", 366: "lakeside", 367: "seashore", 826: "paddle", 244: "liner",
    235: "fireboat", 254: "bicycle-built-for-two", 260: "motor_scooter", 277: "moped",
    287: "streetcar", 882: "trolleybus", 680: "steel_arch_bridge", 685: "palace",
    686: "monastery", 692: "stupa", 933: "bell_cote",
}


def mean_chroma(flat_rows):
    rgb = np.asarray(flat_rows).reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb.reshape(-1, 64, 3), cv2.COLOR_RGB2LAB).reshape(rgb.shape)
    return np.hypot(lab[..., 1], lab[..., 2]).mean(axis=(1, 2))


def shard_labels(raw_path):
    with open(raw_path, "rb") as handle:
        content = pickle.load(handle, encoding="latin1")
    return np.asarray(content.get("labels", content.get(b"labels")), dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-chroma", type=float, default=5.0,
                        help="Drop black-and-white/sepia images; they teach desaturated colour")
    args = parser.parse_args()
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_manifest = root / "imagenet_targeted_v10.json"
    if output_manifest.exists():
        raise FileExistsError(f"V10 ImageNet selection already exists: {output_manifest}")

    labels_path = root / "train_labels_v3.npy"
    if labels_path.exists():
        labels = np.load(labels_path, allow_pickle=True)
    else:
        labels = []
        for shard in manifest["shards"]:
            if sha256(shard["raw_path"]) != shard["raw_sha256"]:
                raise ValueError(f"Raw batch fingerprint mismatch: {shard['raw_path']}")
            values = shard_labels(shard["raw_path"])
            if len(values) != shard["count"]:
                raise ValueError(f"Label count mismatch for {shard['path']}")
            labels.append(values)
            print(f"labels: {shard['path']} ({len(values)})", flush=True)
        labels = np.array(labels, dtype=object)
        np.save(labels_path, labels, allow_pickle=True)

    wanted = np.array(sorted(CLASSES))
    rows = [(shard_id, int(row)) for shard_id, shard_labels_ in enumerate(labels)
            for row in np.flatnonzero(np.isin(shard_labels_, wanted))]
    rows = np.asarray(rows, dtype=np.int32)
    keep = np.ones(len(rows), dtype=bool)
    for shard_id in np.unique(rows[:, 0]):
        shard = np.load(root / manifest["shards"][shard_id]["path"], mmap_mode="r")
        positions = np.flatnonzero(rows[:, 0] == shard_id)
        keep[positions] = mean_chroma(shard[rows[positions, 1]]) >= args.min_chroma
    dropped = int((~keep).sum())
    rows = rows[keep]
    np.random.default_rng(args.seed).shuffle(rows)
    index_path = root / "imagenet_targeted_v10.npy"
    np.save(index_path, rows)

    counts = Counter(int(labels[s][r]) for s, r in rows)
    output = {
        **{k: manifest[k] for k in ("version", "dataset", "seed", "shards", "official_data")},
        "protocol": "ImageNet64 train rows of military and Vietnam-relevant classes (V10 replay)",
        "min_chroma": args.min_chroma,
        "dropped_low_chroma": dropped,
        "base_manifest": str(args.manifest.resolve()),
        "base_manifest_sha256": sha256(args.manifest),
        "classes": {str(k): v for k, v in sorted(CLASSES.items())},
        "class_counts": {CLASSES[k]: counts[k] for k in sorted(CLASSES)},
        "splits": {"train": {"path": index_path.name, "count": len(rows),
                             "sha256": sha256(index_path), "source": "train_shards"}},
    }
    output_manifest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "dropped_low_chroma": dropped,
                      "class_counts": output["class_counts"]}, indent=1))


if __name__ == "__main__":
    main()
