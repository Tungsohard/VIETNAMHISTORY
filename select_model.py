"""Freeze the best complete run using the model-selection validation subset only."""
import argparse
import json
from pathlib import Path

import torch

from colorization.data import sha256
from colorization.runtime import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/model_selection.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Model selection is already frozen")
    candidates = []
    fingerprint = None
    for run in args.runs:
        checkpoint_path = run / "best.pth"
        checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint["run_config"]
        if config.get("protocol") != "imagenet_only_v3":
            raise ValueError(f"Unsupported experiment protocol: {run}")
        if config["smoke_test"]:
            raise ValueError(f"Smoke run cannot be selected: {run}")
        current = (config["manifest_sha256"], config["image_size"])
        if fingerprint is not None and current != fingerprint:
            raise ValueError("Candidates used different data or image sizes")
        fingerprint = current
        architecture = config.get("backbone", config.get("encoder_name", config["model"]))
        candidates.append({"checkpoint": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path),
                           "model": config["model"], "architecture": architecture,
                           "val_loss": checkpoint["validation"]["val_loss"],
                           "val_ab_mae": checkpoint["validation"]["val_ab_mae"]})
    winner = min(candidates, key=lambda item: item["val_loss"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"criterion": "minimum official-validation loss", "winner": winner,
                             "candidates": candidates, "test_opened": False})
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
