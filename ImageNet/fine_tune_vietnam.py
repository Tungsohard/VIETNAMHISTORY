"""Fine-tune the frozen ImageNet winner on video-disjoint Vietnam train/val data."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from colorization.data import MultiSourceVietnamDataset, VietnamDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_checkpoint, load_config, make_loader, set_seed, write_csv, write_json
from train import color_loss, validate


ROOT = Path(__file__).resolve().parent


def chroma_aware_loss(prediction, target, config, luminance=None, sources=None):
    """Discourage the low-chroma shortcut while retaining the stable base loss."""
    base = color_loss(prediction, target)
    pred_chroma = torch.linalg.vector_norm(prediction, dim=1)
    target_chroma = torch.linalg.vector_norm(target, dim=1)
    scale = config.get("color_weight_scale", 0.12)
    pixel_weight = 1.0 + config.get("color_pixel_boost", 2.0) * torch.clamp(target_chroma / scale, 0, 1)
    pixel_error = torch.abs(prediction - target).mean(dim=1)
    weighted_ab = (pixel_error * pixel_weight).sum() / pixel_weight.sum()
    chroma = F.smooth_l1_loss(pred_chroma, target_chroma, beta=0.03)
    chroma_mean = torch.abs(pred_chroma.mean() - target_chroma.mean())
    colorful = target_chroma > config.get("hue_chroma_threshold", 4.0 / 128.0)
    hue_error = 1.0 - F.cosine_similarity(prediction, target, dim=1, eps=1e-6)
    hue_weight = target_chroma * colorful
    hue = (hue_error * hue_weight).sum() / hue_weight.sum().clamp_min(1e-6)
    loss = (base
            + config.get("weighted_ab_weight", 0.5) * weighted_ab
            + config.get("chroma_loss_weight", 0.75) * chroma
            + config.get("mean_chroma_weight", 0.5) * chroma_mean
            + config.get("hue_loss_weight", 0.02) * hue)
    if config.get("loss_version") in {"hue_chroma_edge_v5", "chroma_calibration_v5", "chroma_calibration_v6", "palette_contrastive_v7"} and luminance is not None:
        gx = F.pad((luminance[..., :, 1:] - luminance[..., :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((luminance[..., 1:, :] - luminance[..., :-1, :]).abs(), (0, 0, 0, 1))
        edge = torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(1)
        edge = edge / edge.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        edge_weight = edge.pow(config.get("edge_gamma", 0.7))
        edge_error = torch.abs(prediction - target).mean(dim=1)
        edge_loss = (edge_error * edge_weight).sum() / edge_weight.sum().clamp_min(1e-6)
        loss = loss + config.get("edge_chroma_loss_weight", 0.25) * edge_loss
    if config.get("loss_version") in {"chroma_calibration_v5", "chroma_calibration_v6", "palette_contrastive_v7"} and luminance is not None:
        # Phạt bất đối xứng: thiếu màu ở pixel có chroma thật cao và thừa màu ở
        # pixel gần trung tính. Mean tính theo từng ảnh để tránh các ảnh rực màu
        # và ảnh nhạt màu triệt tiêu lẫn nhau trong cùng batch.
        normalized_target = torch.clamp(target_chroma / config.get("color_weight_scale", 0.12), 0, 1)
        under = F.relu(target_chroma - pred_chroma)
        over = F.relu(pred_chroma - target_chroma)
        under_weight = 1.0 + config.get("chroma_under_weight", 3.0) * normalized_target
        over_weight = 1.0 + config.get("chroma_over_weight", 3.0) * (1.0 - normalized_target)
        asymmetric_chroma = (under * under_weight + over * over_weight).mean()
        per_image_mean = torch.abs(
            pred_chroma.mean(dim=(1, 2)) - target_chroma.mean(dim=(1, 2))
        ).mean()
        pred_image_mean = pred_chroma.mean(dim=(1, 2)).clamp_min(1e-5)
        target_image_mean = target_chroma.mean(dim=(1, 2)).clamp_min(1e-5)
        per_image_log_ratio = torch.abs(torch.log(pred_image_mean / target_image_mean)).mean()
        loss = (loss
                + config.get("asymmetric_chroma_weight", 1.0) * asymmetric_chroma
                + config.get("per_image_chroma_weight", 0.75) * per_image_mean
                + config.get("per_image_ratio_weight", 0.0) * per_image_log_ratio)
    if config.get("loss_version") in {"chroma_calibration_v6", "palette_contrastive_v7"}:
        neutral_threshold = config.get("neutral_chroma_threshold", 3.0 / 128.0)
        colorful_threshold = config.get("colorful_chroma_threshold", 10.0 / 128.0)
        neutral = target_chroma < neutral_threshold
        colorful = target_chroma > colorful_threshold
        neutral_loss = pred_chroma[neutral].mean() if neutral.any() else pred_chroma.new_zeros(())
        colorful_under = (F.relu(target_chroma - pred_chroma)[colorful].mean()
                          if colorful.any() else pred_chroma.new_zeros(()))
        loss = (loss
                + config.get("neutral_region_weight", 1.0) * neutral_loss
                + config.get("colorful_under_weight", 0.75) * colorful_under)
    if config.get("loss_version") == "palette_contrastive_v7":
        # Soft hue histograms align the predicted and reference palettes while
        # remaining differentiable. Chroma weighting prevents unstable hue
        # gradients in neutral pixels.
        bins = int(config.get("palette_bins", 12))
        temperature = float(config.get("palette_temperature", 0.18))
        centers = torch.linspace(-torch.pi, torch.pi, bins + 1, device=prediction.device)[:-1]
        pred_angle = torch.atan2(prediction[:, 1], prediction[:, 0]).unsqueeze(1)
        target_angle = torch.atan2(target[:, 1], target[:, 0]).unsqueeze(1)
        pred_distance = 1.0 - torch.cos(pred_angle - centers.view(1, -1, 1, 1))
        target_distance = 1.0 - torch.cos(target_angle - centers.view(1, -1, 1, 1))
        pred_assign = F.softmax(-pred_distance / temperature, dim=1)
        target_assign = F.softmax(-target_distance / temperature, dim=1)
        pred_weight = pred_chroma.unsqueeze(1)
        target_weight = target_chroma.unsqueeze(1)
        pred_hist = (pred_assign * pred_weight).sum((2, 3)) / pred_weight.sum((2, 3)).clamp_min(1e-6)
        target_hist = (target_assign * target_weight).sum((2, 3)) / target_weight.sum((2, 3)).clamp_min(1e-6)
        palette_alignment = F.smooth_l1_loss(pred_hist, target_hist)
        # Match palette spread rather than forcing hard semantic colours.
        pred_spread = prediction.flatten(2).std(dim=2)
        target_spread = target.flatten(2).std(dim=2)
        palette_spread = F.smooth_l1_loss(pred_spread, target_spread)
        loss = (loss
                + config.get("palette_loss_weight", 0.35) * palette_alignment
                + config.get("palette_spread_weight", 0.10) * palette_spread)
    if config.get("source_aware_chroma", False) and sources is not None:
        # Source-specific per-image constraints address the known domain split:
        # historical videos are often already saturated while reviewed stills
        # tend to remain under-colourized. This is a soft penalty, not a hard
        # colour rule, and therefore preserves scene-level variation.
        pred_mean = pred_chroma.mean(dim=(1, 2))
        target_mean = target_chroma.mean(dim=(1, 2)).clamp_min(1e-5)
        ratios = pred_mean / target_mean
        source_penalty = prediction.new_zeros(())
        for index, source in enumerate(sources):
            if source == "historical_video":
                ceiling = float(config.get("historical_video_ratio_ceiling", 1.10))
                source_penalty = source_penalty + float(config.get("historical_video_over_weight", 2.0)) * F.relu(ratios[index] - ceiling).square()
            elif source == "historical_wikimedia_still":
                floor = float(config.get("historical_still_ratio_floor", 0.90))
                ceiling = float(config.get("historical_still_ratio_ceiling", 1.15))
                source_penalty = source_penalty + float(config.get("historical_still_under_weight", 1.25)) * F.relu(floor - ratios[index]).square()
                source_penalty = source_penalty + float(config.get("historical_still_over_weight", 0.5)) * F.relu(ratios[index] - ceiling).square()
            else:
                floor = float(config.get("default_ratio_floor", 0.90))
                ceiling = float(config.get("default_ratio_ceiling", 1.10))
                source_penalty = source_penalty + float(config.get("default_ratio_weight", 0.5)) * (
                    F.relu(floor - ratios[index]).square() + F.relu(ratios[index] - ceiling).square())
        loss = loss + float(config.get("source_aware_weight", 0.25)) * source_penalty / max(len(sources), 1)
    return loss


def chroma_selection_score(val_loss, ratio, config, under_fraction=0.0, over_fraction=0.0):
    ratio = max(ratio, 1e-3)
    log_penalty = abs(float(torch.log(torch.tensor(ratio))))
    lower = config.get("selection_chroma_min_ratio", 0.8)
    upper = config.get("selection_chroma_max_ratio", 1.2)
    band_penalty = max(0.0, lower - ratio) + max(0.0, ratio - upper)
    distribution_penalty = (
        config.get("selection_under_fraction_weight", 0.0) * max(0.0, under_fraction - 0.30)
        + config.get("selection_over_fraction_weight", 0.0) * max(0.0, over_fraction - 0.15)
    )
    return (val_loss
            + config.get("selection_chroma_weight", 0.03) * log_penalty
            + config.get("selection_chroma_band_weight", 0.2) * band_penalty
            + distribution_penalty)


def add_source_balanced_score(metrics):
    source_scores = [value for key, value in metrics.items()
                     if key.startswith("val_") and key.endswith("_selection_score")
                     and key not in {"val_selection_score", "val_source_balanced_selection_score"}]
    if source_scores:
        metrics["val_source_balanced_selection_score"] = sum(source_scores) / len(source_scores)
    elif "val_selection_score" in metrics:
        metrics["val_source_balanced_selection_score"] = metrics["val_selection_score"]


@torch.inference_mode()
def validate_chroma_aware(model, batches, device, config):
    model.eval()
    loss_sum = ab_sum = pred_chroma_sum = target_chroma_sum = 0.0
    hue_sum = hue_weight_sum = 0.0
    count = pixels = under_count = over_count = 0
    threshold = config.get("hue_chroma_threshold", 4.0 / 128.0)
    for batch in batches:
        luminance, target = batch["L"].to(device), batch["ab"].to(device)
        prediction = model(luminance)
        loss_sum += chroma_aware_loss(prediction, target, config, luminance, batch.get("source")).item() * len(luminance)
        ab_sum += F.l1_loss(prediction, target, reduction="sum").item()
        pred_chroma = torch.linalg.vector_norm(prediction, dim=1)
        target_chroma = torch.linalg.vector_norm(target, dim=1)
        pred_chroma_sum += pred_chroma.sum().item()
        target_chroma_sum += target_chroma.sum().item()
        image_pred = pred_chroma.mean(dim=(1, 2))
        image_target = target_chroma.mean(dim=(1, 2)).clamp_min(1e-8)
        image_ratios = image_pred / image_target
        under_count += int((image_ratios < 0.8).sum().item())
        over_count += int((image_ratios > 1.25).sum().item())
        weights = target_chroma * (target_chroma > threshold)
        hue_sum += ((1.0 - F.cosine_similarity(prediction, target, dim=1, eps=1e-6)) * weights).sum().item()
        hue_weight_sum += weights.sum().item()
        count += len(luminance)
        pixels += target_chroma.numel()
    pred_mean = pred_chroma_sum / pixels * 128.0
    target_mean = target_chroma_sum / pixels * 128.0
    ratio = pred_mean / max(target_mean, 1e-8)
    val_loss = loss_sum / count
    under_fraction = under_count / count
    over_fraction = over_count / count
    selection_score = chroma_selection_score(val_loss, ratio, config, under_fraction, over_fraction)
    return {"val_loss": val_loss,
            "val_ab_mae": ab_sum / (pixels * 2),
            "val_pred_chroma": pred_mean,
            "val_target_chroma": target_mean,
            "val_chroma_ratio": ratio,
            "val_under_chroma_fraction": under_fraction,
            "val_over_chroma_fraction": over_fraction,
            "val_hue_cosine_error": hue_sum / max(hue_weight_sum, 1e-8),
            "val_selection_score": selection_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ddcolor_vietnam_adaptation.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "data" / "vietnam_protocol_v1" / "manifest.json")
    parser.add_argument("--source-selection", type=Path, default=ROOT / "reports" / "model_selection.json")
    parser.add_argument("--source-checkpoint", type=Path, default=None,
                        help="Direct path to checkpoint to initialize model from")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None,
                        help="Path to pre-trained teacher checkpoint (e.g. DDColor V6) for cross-architecture knowledge distillation")
    parser.add_argument("--distill-weight", type=float, default=0.5,
                        help="Weight for teacher distillation loss (default: 0.5)")
    parser.add_argument("--from-torchvision", action="store_true",
                        help="Initialize model directly with torchvision ImageNet weights without needing a prior checkpoint")
    parser.add_argument("--vietnam-root", type=Path, default=ROOT.parent / "dataset" / "VietNam_dataset")
    parser.add_argument("--wikimedia-root", type=Path, default=ROOT / "data" / "wikimedia_vietnam")
    parser.add_argument("--historical-root", type=Path,
                        default=ROOT / "data" / "historical_vietnam_v4_candidates_v2")
    parser.add_argument("--still-root", type=Path,
                        default=ROOT / "data" / "historical_vietnam_images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("Limits must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pth").exists() and not args.resume:
        raise FileExistsError("Run already exists; use --resume or a new output directory")

    config = load_config(args.config)
    loss_version = config.get("loss_version", "v1")
    if loss_version not in ("v1", "chroma_aware_v2", "hue_chroma_edge_v5", "chroma_calibration_v5", "chroma_calibration_v6", "palette_contrastive_v7"):
        raise ValueError(f"Unsupported loss_version: {loss_version}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_name = protocol.get("protocol")
    supported_protocols = {"vietnam_video_disjoint_v1", "vietnam_video_disjoint_wikimedia_train_v2",
                           "vietnam_video_disjoint_wikimedia_historical_v3",
                           "vietnam_historical_still_replay_v4",
                           "historical_vietnam_still_collection_disjoint_v1"}
    if protocol_name not in supported_protocols:
        raise ValueError("Unsupported Vietnam protocol")
    still_protocol = protocol_name == "historical_vietnam_still_collection_disjoint_v1"
    split_names = ("train", "val") if still_protocol else ("train", "val", "test")
    csv_paths = {split: args.protocol.parent / protocol["splits"][split]["path"]
                 for split in split_names}
    for split, path in csv_paths.items():
        if sha256(path) != protocol["splits"][split]["sha256"]:
            raise ValueError(f"Vietnam {split} split fingerprint mismatch")

    if args.source_checkpoint and args.source_checkpoint.is_file():
        source_path = args.source_checkpoint
        source = load_checkpoint(source_path)
        source_config = source.get("run_config", {})
        model_config = {**source_config, **config, "pretrained_encoder": False}
    elif args.from_torchvision:
        source_path = None
        source = None
        model_config = {**config, "pretrained_encoder": True}
    else:
        selection = json.loads(args.source_selection.read_text(encoding="utf-8"))
        source_path = Path(selection["winner"]["checkpoint"])
        if sha256(source_path) != selection["winner"]["sha256"]:
            raise ValueError("ImageNet source checkpoint fingerprint mismatch")
        source = load_checkpoint(source_path)
        source_config = source["run_config"]
        model_config = {**source_config, **config, "pretrained_encoder": False}
    set_seed(config["seed"])
    if still_protocol:
        train_ds = VietnamDataset(args.still_root, config["image_size"], args.train_limit,
                                  "train", csv_paths["train"], augment=True)
        val_ds = VietnamDataset(args.still_root, config["image_size"], args.val_limit,
                                "val", csv_paths["val"])
    elif protocol_name in {"vietnam_video_disjoint_wikimedia_train_v2",
                           "vietnam_video_disjoint_wikimedia_historical_v3",
                           "vietnam_historical_still_replay_v4"}:
        roots = {"vietnam_video": args.vietnam_root, "wikimedia_commons": args.wikimedia_root,
                 "historical_video": args.historical_root,
                 "historical_wikimedia_still": args.still_root}
        train_ds = MultiSourceVietnamDataset(roots, config["image_size"], args.train_limit,
                                             csv_paths["train"], augment=True)
        if protocol_name in {"vietnam_video_disjoint_wikimedia_historical_v3",
                             "vietnam_historical_still_replay_v4"}:
            val_ds = MultiSourceVietnamDataset(roots, config["image_size"], args.val_limit,
                                               csv_paths["val"], augment=False)
        else:
            val_ds = VietnamDataset(args.vietnam_root, config["image_size"], args.val_limit,
                                    "val", csv_paths["val"])
    else:
        train_ds = VietnamDataset(args.vietnam_root, config["image_size"], args.train_limit,
                                  "train", csv_paths["train"], augment=True)
        val_ds = VietnamDataset(args.vietnam_root, config["image_size"], args.val_limit,
                                "val", csv_paths["val"])
    val_loader = make_loader(val_ds, config["batch_size"], config.get("validation_workers", 0), config["seed"])
    val_source_loaders = {}
    if config.get("validation_source_breakdown") and isinstance(val_ds, MultiSourceVietnamDataset):
        for source_name in sorted({row["source"] for row in val_ds.rows}):
            indices = [index for index, row in enumerate(val_ds.rows)
                       if row["source"] == source_name]
            val_source_loaders[source_name] = make_loader(
                Subset(val_ds, indices), config["batch_size"],
                config.get("validation_workers", 0), config["seed"])
    device = torch.device(args.device)
    model = build_model(model_config).to(device)
    teacher = None
    if args.teacher_checkpoint and args.teacher_checkpoint.is_file():
        print(f"Loading Teacher model for distillation from: {args.teacher_checkpoint}")
        teacher_ckpt = load_checkpoint(args.teacher_checkpoint)
        teacher_cfg = {**teacher_ckpt["run_config"], "pretrained_encoder": False}
        teacher = build_model(teacher_cfg).to(device).eval()
        teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=True)
        for p in teacher.parameters():
            p.requires_grad = False
    if hasattr(model, "semantic_parameters") and config.get("semantic_learning_rate"):
        semantic = list(model.semantic_parameters())
        semantic_ids = {id(parameter) for parameter in semantic}
        dino = [parameter for parameter in model.semantic_bridge.encoder.parameters() if parameter.requires_grad]
        dino_ids = {id(parameter) for parameter in dino}
        base = [parameter for parameter in model.parameters()
                if id(parameter) not in semantic_ids and id(parameter) not in dino_ids]
        groups = [{"params": base, "lr": config["learning_rate"]},
                  {"params": semantic, "lr": config["semantic_learning_rate"]}]
        if dino:
            groups.append({"params": dino, "lr": config.get("dino_learning_rate", 1e-6)})
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.get("scheduler_t_max", config["epochs"]), eta_min=2e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_config = {**model_config, "protocol": protocol_name,
                  "protocol_manifest_sha256": sha256(args.protocol),
                  "source_checkpoint": str(source_path.resolve()) if source_path else "torchvision_imagenet_pretrained",
                  "source_checkpoint_sha256": sha256(source_path) if source_path else "torchvision_weights",
                  "train_count": len(train_ds), "val_count": len(val_ds),
                  "train_source_counts": protocol["splits"]["train"].get(
                      "source_counts", {"vietnam_video": len(train_ds)}),
                  "val_source_counts": protocol["splits"]["val"].get(
                      "source_counts", {"vietnam_video": len(val_ds)}),
                  "smoke_test": bool(args.train_limit or args.val_limit),
                  "selection": ("minimum_video_disjoint_vietnam_validation_loss" if loss_version == "v1"
                                else ("minimum_source_balanced_chroma_penalized_validation_score"
                                      if config.get("selection_source_balanced")
                                      else "minimum_chroma_penalized_video_disjoint_validation_score")),
                  "torch_version": str(torch.__version__)}
    if still_protocol:
        run_config.update({
            "still_root": str(args.still_root.resolve()),
            "test_count": 0,
            "train_collections": protocol["splits"]["train"]["collection_count"],
            "val_collections": protocol["splits"]["val"]["collection_count"],
        })
    else:
        run_config.update({
            "test_count": protocol["splits"]["test"]["count"],
            "train_videos": protocol["splits"]["train"]["videos"],
            "val_videos": protocol["splits"]["val"]["videos"],
            "test_videos": protocol["splits"]["test"]["videos"],
        })
    start, best, stale, history, pending_epoch = 0, float("inf"), 0, [], None
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
            pending_epoch, start = checkpoint["epoch"], checkpoint["epoch"] - 1
        else:
            start = checkpoint["epoch"]
        history_path = args.output / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))[:start]
    elif source is not None:
        model.load_state_dict(source["model_state_dict"], strict=False)
    if source is not None:
        del source


    if loss_version != "v1" and not args.resume:
        baseline_metrics = validate_chroma_aware(model, val_loader, device, config)
        for source, source_loader in val_source_loaders.items():
            source_metrics = validate_chroma_aware(model, source_loader, device, config)
            baseline_metrics.update({f"val_{source}_{key.removeprefix('val_')}": value
                                     for key, value in source_metrics.items()})
        add_source_balanced_score(baseline_metrics)
        selection_key = ("val_source_balanced_selection_score"
                         if (config.get("selection_source_balanced") and "val_source_balanced_selection_score" in baseline_metrics)
                         else "val_selection_score")
        best = baseline_metrics.get(selection_key, baseline_metrics.get("val_selection_score", baseline_metrics["val_loss"]))
        if teacher is not None:
            with torch.no_grad():
                val_distill_sum = 0.0
                val_distill_count = 0
                for b in val_loader:
                    v_lum = b["L"].to(device)
                    v_pred = model(v_lum)
                    v_teach = teacher(v_lum)
                    val_distill_sum += F.l1_loss(v_pred, v_teach).item() * len(v_lum)
                    val_distill_count += len(v_lum)
                baseline_metrics["val_distill_l1"] = val_distill_sum / max(val_distill_count, 1)
                w = float(config.get("distill_weight", args.distill_weight))
                if w > 0:
                    baseline_metrics["val_distill_selection_score"] = (1.0 - w) * best + w * (baseline_metrics["val_distill_l1"] * 2.0)
                    best = baseline_metrics["val_distill_selection_score"]
        write_json(args.output / "baseline_validation.json", baseline_metrics)
        baseline_state = {"epoch": 0, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "scheduler_state_dict": scheduler.state_dict(),
                          "scaler_state_dict": scaler.state_dict(), "run_config": run_config,
                          "validation": baseline_metrics, "best_val_loss": best,
                          "best_validation_score": best, "stale_epochs": 0,
                          "candidate_type": "frozen_imagenet_source_baseline"}
        torch.save(baseline_state, args.output / "best.pth")
        print("Source baseline validation:")
        print(json.dumps(baseline_metrics, indent=2))

    write_json(args.output / "run_config.json", run_config)
    started = time.time()
    accumulation = config.get("gradient_accumulation_steps", 1)
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    for epoch in range(start, config["epochs"]):
        set_seed(config["seed"] + epoch)
        if pending_epoch == epoch + 1:
            total, seen = checkpoint["train_loss_sum"], checkpoint["train_seen"]
            pending_epoch = None
            print(f"Resuming epoch {epoch + 1} at validation", flush=True)
        else:
            train_loader = make_loader(train_ds, config["batch_size"], config["num_workers"],
                                       config["seed"] + epoch, True)
            model.train()
            if hasattr(model, "set_training_epoch"):
                model.set_training_epoch(epoch)
            frozen = epoch < config.get("freeze_encoder_epochs", 0)
            if hasattr(model, "encoder_parameters"):
                for parameter in model.encoder_parameters():
                    parameter.requires_grad_(not frozen)
            elif hasattr(model, "encoder"):
                for parameter in model.encoder.parameters():
                    parameter.requires_grad_(not frozen)
            total = seen = 0
            progress = tqdm(train_loader, desc=f"Vietnam epoch {epoch + 1}/{config['epochs']}")
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress):
                luminance, target = batch["L"].to(device), batch["ab"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(luminance)
                    base_loss = (color_loss(prediction, target) if loss_version == "v1"
                                 else chroma_aware_loss(prediction, target, config, luminance, batch.get("source")))
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_ab = teacher(luminance)
                        t_chroma = torch.linalg.vector_norm(teacher_ab, dim=1, keepdim=True)
                        saliency_boost = 1.0 + float(config.get("distill_saliency_boost", 2.5)) * torch.clamp(t_chroma / 0.15, 0.0, 1.0)
                        kd_l1 = (torch.abs(prediction - teacher_ab) * saliency_boost).mean()
                        kd_hue = (1.0 - F.cosine_similarity(prediction, teacher_ab, dim=1, eps=1e-6))
                        colorful_mask = (t_chroma.squeeze(1) > float(config.get("hue_chroma_threshold", 0.03125)))
                        kd_hue = (kd_hue * colorful_mask).sum() / colorful_mask.sum().clamp_min(1.0)
                        distill_loss = kd_l1 + float(config.get("distill_hue_weight", 0.10)) * kd_hue
                        w = float(config.get("distill_weight", args.distill_weight))
                        loss = (1.0 - w) * base_loss + w * distill_loss
                    else:
                        loss = base_loss
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
                        "run_config": run_config, "best_val_loss": best, "best_validation_score": best,
                        "stale_epochs": stale,
                        "train_loss_sum": total, "train_seen": seen}, args.output / "pending_validation.pth")

        metrics = (validate(model, val_loader, device) if loss_version == "v1"
                   else validate_chroma_aware(model, val_loader, device, config))
        if loss_version != "v1":
            for source, source_loader in val_source_loaders.items():
                source_metrics = validate_chroma_aware(model, source_loader, device, config)
                metrics.update({f"val_{source}_{key.removeprefix('val_')}": value
                                for key, value in source_metrics.items()})
            add_source_balanced_score(metrics)
        scheduler.step()
        selection_key = ("val_source_balanced_selection_score"
                         if (config.get("selection_source_balanced") and "val_source_balanced_selection_score" in metrics)
                         else "val_selection_score")
        score = metrics.get(selection_key, metrics.get("val_selection_score", metrics["val_loss"]))
        if teacher is not None:
            with torch.no_grad():
                val_distill_sum = 0.0
                val_distill_count = 0
                for b in val_loader:
                    v_lum = b["L"].to(device)
                    v_pred = model(v_lum)
                    v_teach = teacher(v_lum)
                    val_distill_sum += F.l1_loss(v_pred, v_teach).item() * len(v_lum)
                    val_distill_count += len(v_lum)
                metrics["val_distill_l1"] = val_distill_sum / max(val_distill_count, 1)
                w = float(config.get("distill_weight", args.distill_weight))
                if w > 0:
                    score = (1.0 - w) * score + w * (metrics["val_distill_l1"] * 2.0)
                    metrics["val_distill_selection_score"] = score
        improved = score < best

        best, stale = (score, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch + 1, "train_loss": total / seen, **metrics,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_minutes": (time.time() - started) / 60}
        history.append(row)
        write_csv(args.output / "history.csv", history)
        state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                 "scaler_state_dict": scaler.state_dict(), "run_config": run_config,
                 "validation": metrics, "best_val_loss": best, "best_validation_score": best,
                 "stale_epochs": stale}
        torch.save(state, args.output / "latest.pth")
        if config.get("save_every_epoch"):
            torch.save(state, args.output / f"epoch_{epoch + 1:02d}.pth")
        if improved:
            torch.save(state, args.output / "best.pth")
        (args.output / "pending_validation.pth").unlink(missing_ok=True)
        print(json.dumps(row, indent=2))
        if stale >= config.get("early_stopping_patience", 0) > 0:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
