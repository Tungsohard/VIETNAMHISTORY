import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size, workers, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model", "image_size", "epochs", "batch_size",
                "learning_rate", "weight_decay", "num_workers", "seed"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    return config


def load_checkpoint(path, map_location="cpu"):
    """Safely load checkpoints produced by this project, including PyTorch 2.8 metadata."""
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location=map_location, weights_only=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
