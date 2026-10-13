"""Train on ImageNet and select epochs on the frozen official-validation subset."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from colorization.data import ImageNetDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_checkpoint, load_config, make_loader, set_seed, write_csv, write_json


ROOT = Path(__file__).resolve().parent


def color_loss(prediction, target):
    robust = F.smooth_l1_loss(prediction, target, beta=0.1)
    pixel = F.l1_loss(prediction, target)
    mean = F.l1_loss(prediction.mean((2, 3)), target.mean((2, 3)))
    pred_chroma = torch.linalg.vector_norm(prediction, dim=1)
    target_chroma = torch.linalg.vector_norm(target, dim=1)
    chroma = F.l1_loss(pred_chroma, target_chroma)
    return robust + 0.35 * pixel + 0.10 * mean + 0.15 * chroma


@torch.inference_mode()
def validate(model, batches, device):
    model.eval()
    loss_sum = ab_sum = 0.0
    count = 0
    for batch in batches:
        luminance, target = batch["L"].to(device), batch["ab"].to(device)
        prediction = model(luminance)
        loss_sum += color_loss(prediction, target).item() * len(luminance)
        ab_sum += F.l1_loss(prediction, target, reduction="sum").item()
        count += len(luminance)
    return {"val_loss": loss_sum / count, "val_ab_mae": ab_sum / (count * 2 * batches.dataset.image_size ** 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "resnet34.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke test only")
    parser.add_argument("--val-limit", type=int, default=0, help="Smoke test only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch build cannot access CUDA")
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("Limits must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pth").exists() and not args.resume:
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    set_seed(config["seed"])
    train_ds = ImageNetDataset(args.manifest, "train", config["image_size"], args.train_limit)
    val_ds = ImageNetDataset(args.manifest, "val", config["image_size"], args.val_limit)
    val_loader = make_loader(val_ds, config["batch_size"], config.get("validation_workers", 0), config["seed"])
    device = torch.device(args.device)
    model = build_model(config).to(device)
    initial = config.get("initial_checkpoint")
    if initial and not args.resume:
        initial_path = (ROOT / initial).resolve()
        if not initial_path.exists():
            raise FileNotFoundError(f"Download the official checkpoint first: {initial_path}")
        if not hasattr(model, "load_official"):
            raise ValueError("initial_checkpoint is only supported by the DDColor adapter")
        model.load_official(initial_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    samples_per_epoch = config.get("samples_per_epoch", 0)
    if samples_per_epoch < 0 or samples_per_epoch > len(train_ds):
        raise ValueError("samples_per_epoch must be 0 or no larger than the training set")
    run_config = {**config, "protocol": "imagenet_only_v3", "manifest_sha256": sha256(args.manifest),
                  "train_count": len(train_ds), "samples_per_epoch": samples_per_epoch or len(train_ds),
                  "val_count": len(val_ds),
                  "smoke_test": bool(args.train_limit or args.val_limit), "selection": "minimum_official_validation_loss",
                  "torch_version": str(torch.__version__)}
    start, best, stale, history = 0, float("inf"), 0, []
    pending_epoch = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Resume config does not exactly match this run")
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume in the original output directory")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best, stale = checkpoint["best_val_loss"], checkpoint["stale_epochs"]
        if checkpoint.get("stage") == "awaiting_validation":
            pending_epoch = checkpoint["epoch"]
            start = pending_epoch - 1
        else:
            start = checkpoint["epoch"]
        history_path = args.output / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))[:start]
    write_json(args.output / "run_config.json", run_config)
    started = time.time()
    accumulation = config.get("gradient_accumulation_steps", 1)
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    for epoch in range(start, config["epochs"]):
        set_seed(config["seed"] + epoch)
        if pending_epoch == epoch + 1:
            total = checkpoint["train_loss_sum"]
            seen = checkpoint["train_seen"]
            pending_epoch = None
            print(f"Resuming epoch {epoch + 1} at validation", flush=True)
        else:
            epoch_dataset = train_ds
            if samples_per_epoch:
                generator = torch.Generator().manual_seed(config["seed"] + epoch)
                indices = torch.randperm(len(train_ds), generator=generator)[:samples_per_epoch].tolist()
                epoch_dataset = Subset(train_ds, indices)
            train_loader = make_loader(epoch_dataset, config["batch_size"], config["num_workers"],
                                       config["seed"] + epoch, True)
            model.train()
            if hasattr(model, "encoder"):
                frozen = epoch < config.get("freeze_encoder_epochs", 0)
                for parameter in model.encoder.parameters():
                    parameter.requires_grad_(not frozen)
            total = seen = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress):
                luminance, target = batch["L"].to(device), batch["ab"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(luminance)
                    loss = color_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss / accumulation).backward()
                update = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
                if update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item() * len(luminance)
                seen += len(luminance)
                progress.set_postfix(loss=f"{total / seen:.5f}")
            torch.save({"stage": "awaiting_validation", "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                        "run_config": run_config, "best_val_loss": best, "stale_epochs": stale,
                        "train_loss_sum": total, "train_seen": seen}, args.output / "pending_validation.pth")
        metrics = validate(model, val_loader, device)
        scheduler.step()
        improved = metrics["val_loss"] < best
        best, stale = (metrics["val_loss"], 0) if improved else (best, stale + 1)
        row = {"epoch": epoch + 1, "train_loss": total / seen, **metrics,
               "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_minutes": (time.time() - started) / 60}
        history.append(row)
        write_csv(args.output / "history.csv", history)
        state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                 "scaler_state_dict": scaler.state_dict(), "run_config": run_config,
                 "validation": metrics, "best_val_loss": best, "stale_epochs": stale}
        torch.save(state, args.output / "latest.pth")
        if improved:
            torch.save(state, args.output / "best.pth")
        (args.output / "pending_validation.pth").unlink(missing_ok=True)
        print(json.dumps(row, indent=2))
        if stale >= config.get("early_stopping_patience", 0) > 0:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
