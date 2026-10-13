"""Score an existing local run checkpoint on the full frozen validation subset."""
import argparse
import json
from pathlib import Path

import torch

from colorization.data import ImageNetDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_checkpoint, make_loader, write_json
from train import validate


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["run_config"]
    if config["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Checkpoint and validation manifest differ")
    build_config = dict(config)
    build_config["pretrained_encoder"] = False
    model = build_model(build_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    dataset = ImageNetDataset(args.manifest, "val", config["image_size"], args.val_limit)
    batches = make_loader(dataset, args.batch_size, args.workers, config["seed"])
    metrics = validate(model, batches, device)
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "validation_count": len(dataset), "partial": bool(args.val_limit), **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
