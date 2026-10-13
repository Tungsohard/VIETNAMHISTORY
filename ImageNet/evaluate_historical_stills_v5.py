"""Compare V3 and a candidate on the collection-disjoint historical still validation set."""
import argparse
import json
from pathlib import Path

import torch

from colorization.data import VietnamDataset, sha256
from colorization.runtime import load_checkpoint, make_loader, write_csv, write_json
from evaluate import evaluate_domain
from evaluate_vietnam_adaptation import load_model


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-selection", type=Path,
                        default=ROOT / "reports" / "vietnam_v3_selection.json")
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-name", default="v5")
    parser.add_argument("--training-protocol", type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data" / "historical_vietnam_image_protocol_v1" / "manifest.json")
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data" / "historical_vietnam_images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.candidate_name.replace("_", "").isalnum():
        raise ValueError("Candidate name may contain only letters, numbers, and underscores")
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "historical_vietnam_still_collection_disjoint_v1":
        raise ValueError("Unexpected historical still protocol")
    if protocol.get("collection_leakage_count") != 0:
        raise ValueError("Historical still protocol contains collection leakage")
    val_info = protocol["splits"]["val"]
    val_csv = args.protocol.parent / val_info["path"]
    if sha256(val_csv) != val_info["sha256"]:
        raise ValueError("Historical validation CSV fingerprint mismatch")
    review_path = args.dataset_root / "manual_review.csv"
    if sha256(review_path) != protocol["manual_review_sha256"]:
        raise ValueError("Manual review fingerprint mismatch")

    selection = json.loads(args.baseline_selection.read_text(encoding="utf-8"))
    baseline_path = Path(selection["winner"]["checkpoint"])
    if sha256(baseline_path) != selection["winner"]["sha256"]:
        raise ValueError("V3 checkpoint fingerprint mismatch")
    adapted = load_checkpoint(args.adapted_checkpoint)
    adapted_config = adapted["run_config"]
    if adapted_config.get("source_checkpoint_sha256") != sha256(baseline_path):
        raise ValueError("Candidate did not originate from the registered V3 checkpoint")
    training_protocol = args.training_protocol or args.protocol
    if adapted_config.get("protocol_manifest_sha256") != sha256(training_protocol):
        raise ValueError("Candidate training protocol fingerprint mismatch")
    del adapted

    dataset = VietnamDataset(args.dataset_root, adapted_config["image_size"], args.limit,
                             "val", val_csv)
    device = torch.device(args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    args.output.mkdir(parents=True)
    paths = {"v3": baseline_path, args.candidate_name: args.adapted_checkpoint}
    results = {}
    for name, checkpoint_path in paths.items():
        model = load_model(checkpoint_path, device)
        destination = args.output / name
        destination.mkdir()
        loader = make_loader(dataset, args.batch_size, args.workers, adapted_config["seed"])
        results[name] = evaluate_domain(model, loader, device, destination, lpips_model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = [key for key in results["v3"] if key != "count"]
    report = {
        "benchmark_role": "collection-disjoint model-selection validation; not a final test",
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256(args.protocol),
        "training_protocol_sha256": sha256(training_protocol),
        "validation_csv_sha256": sha256(val_csv),
        "validation_count": len(dataset),
        "validation_collection_count": val_info["collection_count"],
        "partial_evaluation": bool(args.limit),
        "lpips_computed": not args.skip_lpips,
        "checkpoint_sha256": {name: sha256(path) for name, path in paths.items()},
        "results": results,
        f"{args.candidate_name}_minus_v3": {
            metric: results[args.candidate_name][metric] - results["v3"][metric]
            for metric in metrics
        },
    }
    write_json(args.output / "comparison.json", report)
    write_csv(args.output / "comparison.csv",
              [{"model": name, **values} for name, values in results.items()])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
