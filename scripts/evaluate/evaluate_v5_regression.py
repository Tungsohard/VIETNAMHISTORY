"""Check a candidate for regressions against V3 on the established Vietnam development test."""
import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import VietnamDataset, sha256
from colorization.runtime import load_checkpoint, make_loader, write_csv, write_json
from evaluate import evaluate_domain
from evaluate_vietnam_adaptation import load_model


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-selection", type=Path,
                        default=ROOT / "outputs" / "reports" / "vietnam_v3_selection.json")
    parser.add_argument("--candidate", type=Path,
                        default=ROOT / "checkpoints" / "runs" / "ddcolor_historical_stills_v5" / "best.pth")
    parser.add_argument("--candidate-name", default="v5")
    parser.add_argument("--training-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "historical_vietnam_image_protocol_v1" / "manifest.json")
    parser.add_argument("--benchmark", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_protocol_v1" / "manifest.json")
    parser.add_argument("--vietnam-root", type=Path,
                        default=ROOT / "data" / "sources" / "vicow")
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
        raise FileExistsError(f"Regression output already exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if benchmark.get("protocol") != "vietnam_video_disjoint_v1":
        raise ValueError("Unexpected Vietnam regression protocol")
    test_info = benchmark["splits"]["test"]
    test_csv = args.benchmark.parent / test_info["path"]
    if sha256(test_csv) != test_info["sha256"]:
        raise ValueError("Vietnam regression CSV fingerprint mismatch")

    selection = json.loads(args.baseline_selection.read_text(encoding="utf-8"))
    baseline = Path(selection["winner"]["checkpoint"])
    if sha256(baseline) != selection["winner"]["sha256"]:
        raise ValueError("V3 checkpoint fingerprint mismatch")
    candidate_state = load_checkpoint(args.candidate)
    candidate_config = candidate_state["run_config"]
    if candidate_config.get("source_checkpoint_sha256") != sha256(baseline):
        raise ValueError("Candidate did not originate from the registered V3 checkpoint")
    if candidate_config.get("protocol_manifest_sha256") != sha256(args.training_protocol):
        raise ValueError("Candidate training protocol fingerprint mismatch")
    del candidate_state

    dataset = VietnamDataset(args.vietnam_root, candidate_config["image_size"], args.limit,
                             "test", test_csv)
    device = torch.device(args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    args.output.mkdir(parents=True)
    paths = {"v3": baseline, args.candidate_name: args.candidate}
    results = {}
    for name, checkpoint in paths.items():
        model = load_model(checkpoint, device)
        destination = args.output / name
        destination.mkdir()
        loader = make_loader(dataset, args.batch_size, args.workers, candidate_config["seed"])
        results[name] = evaluate_domain(model, loader, device, destination, lpips_model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = [key for key in results["v3"] if key != "count"]
    report = {
        "benchmark_role": "previously used Vietnam development test; regression guard only",
        "benchmark_protocol": benchmark["protocol"],
        "benchmark_sha256": sha256(args.benchmark),
        "training_protocol_sha256": sha256(args.training_protocol),
        "test_csv_sha256": sha256(test_csv),
        "test_count": len(dataset),
        "held_out_videos": test_info["videos"],
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
