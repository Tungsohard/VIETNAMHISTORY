import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .color import rgb_to_tensors


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageNetDataset(Dataset):
    def __init__(self, manifest_path, split, image_size=64, limit=0):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.split, self.image_size = split, image_size
        split_info = self.manifest["splits"][split]
        self.source = split_info["source"]
        index_path = self.root / split_info["path"]
        if sha256(index_path) != split_info["sha256"]:
            raise ValueError(f"Split fingerprint mismatch: {split}")
        self.index_path = index_path
        self.rows = np.load(self.index_path, mmap_mode="r")
        if limit:
            self.rows = self.rows[:limit]
        self.length = len(self.rows)
        self.limit = limit
        self.shards = {}
        self.official_array = None
        self.official_path = None
        if self.source == "official_validation":
            official = self.manifest["official_data"]
            self.official_path = self.root / official["path"]
            if sha256(self.official_path) != official["sha256"]:
                raise ValueError("Official validation fingerprint mismatch")

    def __len__(self):
        return self.length

    def __getstate__(self):
        """Keep Windows DataLoader worker payloads small and pickle-safe."""
        state = self.__dict__.copy()
        state["rows"] = None
        state["shards"] = {}
        state["official_array"] = None
        return state

    def _ensure_rows(self):
        if self.rows is None:
            self.rows = np.load(self.index_path, mmap_mode="r")
            if self.limit:
                self.rows = self.rows[:self.limit]

    def __getitem__(self, index):
        self._ensure_rows()
        if self.source == "official_validation":
            if self.official_array is None:
                self.official_array = np.load(self.official_path, mmap_mode="r")
            flat = self.official_array[int(self.rows[index])]
            identifier = f"official-val:{int(self.rows[index])}"
        else:
            shard, row = map(int, self.rows[index])
            if shard not in self.shards:
                self.shards[shard] = np.load(self.root / self.manifest["shards"][shard]["path"], mmap_mode="r")
            flat = self.shards[shard][row]
            identifier = f"train:{shard}:{row}"
        rgb = flat.reshape(3, 64, 64).transpose(1, 2, 0)
        sample = rgb_to_tensors(rgb, self.image_size, augment=self.split == "train")
        sample.update(id=identifier, group="ImageNet64")
        return sample


class VietnamDataset(Dataset):
    def __init__(self, root, image_size=64, limit=0, split="test", csv_path=None, augment=False):
        self.root, self.image_size = Path(root), image_size
        self.split, self.augment = split, augment
        self.csv_path = Path(csv_path) if csv_path else self.root / f"{split}.csv"
        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Vietnam split is empty: {self.csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["colorPath"]) as image:
            sample = rgb_to_tensors(np.asarray(image.convert("RGB")), self.image_size, self.augment)
        sample.update(id=row["colorPath"], group=row["video"])
        return sample


class MultiSourceVietnamDataset(Dataset):
    """Load a CSV whose rows select one of several independently rooted sources."""
    def __init__(self, roots, image_size=64, limit=0, csv_path=None, augment=False, degrade=0.0):
        self.roots = {name: Path(root) for name, root in roots.items()}
        self.image_size, self.augment, self.degrade = image_size, augment, degrade
        self.csv_path = Path(csv_path)
        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"Multi-source split is empty: {self.csv_path}")
        unknown = {row["source"] for row in self.rows} - self.roots.keys()
        if unknown:
            raise ValueError(f"Unknown image sources: {sorted(unknown)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.roots[row["source"]] / row["colorPath"]
        with Image.open(path) as image:
            sample = rgb_to_tensors(np.asarray(image.convert("RGB")), self.image_size, self.augment,
                                    self.degrade)
        sample.update(id=f"{row['source']}:{row['colorPath']}", group=row["video"], source=row["source"])
        return sample


class ImageNetVietnamReplayDataset(Dataset):
    """Present ImageNet replay and Vietnam sources as one source-aware dataset."""
    def __init__(self, imagenet, vietnam):
        self.imagenet = imagenet
        self.vietnam = vietnam
        self.source_names = (["imagenet_replay"] * len(imagenet)
                             + [row["source"] for row in vietnam.rows])

    def __len__(self):
        return len(self.source_names)

    def __getitem__(self, index):
        if index < len(self.imagenet):
            sample = self.imagenet[index]
            sample["source"] = "imagenet_replay"
            return sample
        return self.vietnam[index - len(self.imagenet)]
