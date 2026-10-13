"""Register an official DDColor checkpoint as a validation-scored candidate."""
import argparse
import json
from pathlib import Path

import torch

from colorization.data import ImageNetDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_config, make_loader, set_seed, write_json
from train import validate


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ddcolor_tiny.json")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "pretrained" / "ddcolor_paper_tiny.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-limit", type=int, default=0, help="Diagnostic run only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Candidate output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = load_config(args.config)
    if config["model"] != "ddcolor":
        raise ValueError("This command only registers official DDColor checkpoints")
    set_seed(config["seed"])
    dataset = ImageNetDataset(args.manifest, "val", config["image_size"], args.val_limit)
    batches = make_loader(dataset, config["batch_size"], config["num_workers"], config["seed"])
    device = torch.device(args.device)
    model = build_model(config)
    model.load_official(args.checkpoint)
    model.to(device)
    metrics = validate(model, batches, device)
    run_config = {
        **config,
        "protocol": "imagenet_only_v3",
        "manifest_sha256": sha256(args.manifest),
        "train_count": 0,
        "val_count": len(dataset),
        "smoke_test": bool(args.val_limit),
        "selection": "minimum_official_validation_loss",
        "candidate_type": "official_pretrained_no_local_finetuning",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "torch_version": str(torch.__version__),
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "run_config.json", run_config)
    torch.save({"epoch": 0, "model_state_dict": model.state_dict(), "run_config": run_config,
                "validation": metrics, "best_val_loss": metrics["val_loss"], "stale_epochs": 0},
               args.output / "best.pth")
    write_json(args.output / "validation.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
