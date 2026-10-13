"""Evaluate the frozen winner on official ImageNet validation and Vietnam test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from colorization.color import lab_to_rgb
from colorization.data import ImageNetDataset, VietnamDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_checkpoint, make_loader, write_csv, write_json


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate_domain(model, batches, device, output, lpips_model=None):
    from colorization.metrics import image_metrics
    model.eval()
    rows = []
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True)
    for batch in tqdm(batches, desc=output.name):
        luminance = batch["L"].to(device)
        prediction = model(luminance)
        pred_rgb, target_rgb = [], []
        for index in range(len(luminance)):
            pred_rgb.append(lab_to_rgb(batch["L"][index], prediction[index]))
            target_rgb.append(batch["rgb"][index].permute(1, 2, 0).numpy())
        lpips_values = [None] * len(pred_rgb)
        if lpips_model is not None:
            pred_tensor = torch.from_numpy(np.stack(pred_rgb)).permute(0, 3, 1, 2).to(device)
            target_tensor = batch["rgb"].to(device)
            lpips_values = lpips_model(pred_tensor * 2 - 1, target_tensor * 2 - 1).flatten().cpu().tolist()
        for index, (prediction_rgb, target) in enumerate(zip(pred_rgb, target_rgb)):
            metrics = image_metrics(prediction_rgb, target)
            if lpips_values[index] is not None:
                metrics["lpips"] = lpips_values[index]
            row = {"id": batch["id"][index], "group": batch["group"][index], **metrics}
            rows.append(row)
            if len(rows) <= 12:
                gray = lab_to_rgb(batch["L"][index], torch.zeros_like(prediction[index]))
                grid = np.concatenate((gray, prediction_rgb, target), axis=1)
                Image.fromarray(np.rint(grid * 255).astype(np.uint8)).save(sample_dir / f"{len(rows):03d}.png")
    metric_names = [key for key in rows[0] if key not in ("id", "group")]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metric_names}
    groups = []
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        groups.append({"group": group, "count": len(selected),
                       **{key: float(np.mean([row[key] for row in selected])) for key in metric_names}})
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "groups.csv", groups)
    return {"count": len(rows), **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "reports" / "model_selection.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--vietnam", type=Path, default=ROOT.parent / "dataset" / "VietNam_dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic run only")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists; final test results are immutable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    checkpoint_path = Path(selection["winner"]["checkpoint"])
    if sha256(checkpoint_path) != selection["winner"]["sha256"]:
        raise ValueError("Selected checkpoint fingerprint mismatch")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["run_config"]
    if config["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Evaluation manifest differs from training")
    build_config = dict(config)
    build_config["pretrained_encoder"] = False
    model = build_model(build_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    datasets = {
        "imagenet": ImageNetDataset(args.manifest, "test", config["image_size"], args.limit),
        "vietnam": VietnamDataset(args.vietnam, config["image_size"], args.limit),
    }
    args.output.mkdir(parents=True)
    results = {}
    for name, dataset in datasets.items():
        domain_output = args.output / name
        domain_output.mkdir()
        batches = make_loader(dataset, args.batch_size, args.workers, config["seed"])
        results[name] = evaluate_domain(model, batches, device, domain_output, lpips_model)
    metric_names = [key for key in results["imagenet"] if key != "count"]
    report = {
        "selection": selection,
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest_sha256": sha256(args.manifest),
        "vietnam_test_csv_sha256": sha256(args.vietnam / "test.csv"),
        "protocol": "same frozen checkpoint; synthetic CIE-L input in both domains",
        "partial_evaluation": bool(args.limit),
        "lpips_computed": not args.skip_lpips,
        "results": results,
        "vietnam_minus_imagenet": {key: results["vietnam"][key] - results["imagenet"][key]
                                    for key in metric_names},
    }
    write_json(args.output / "comparison.json", report)
    write_csv(args.output / "comparison.csv", [{"dataset": name, **values} for name, values in results.items()])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
