"""Compare a frozen baseline and successor on the historical development benchmark."""
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
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path,
                        default=ROOT / "data" / "protocols" / "vietnam_protocol_v3" / "manifest.json")
    parser.add_argument("--benchmark", type=Path,
                        default=ROOT / "data" / "protocols" / "historical_vietnam_final_test_v1" / "manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if benchmark.get("protocol") != "historical_vietnam_final_test_v1":
        raise ValueError("Unexpected historical benchmark protocol")
    test_csv = args.benchmark.parent / benchmark["test"]["path"]
    if sha256(test_csv) != benchmark["test"]["sha256"]:
        raise ValueError("Historical benchmark CSV fingerprint mismatch")
    for relative, expected in benchmark["frame_sha256"].items():
        if sha256(args.benchmark.parent / relative) != expected:
            raise ValueError(f"Historical frame fingerprint mismatch: {relative}")

    selection = json.loads(args.baseline_selection.read_text(encoding="utf-8"))
    baseline = Path(selection["winner"]["checkpoint"])
    if sha256(baseline) != selection["winner"]["sha256"]:
        raise ValueError("Baseline checkpoint fingerprint mismatch")
    adapted_state = load_checkpoint(args.adapted_checkpoint)
    adapted_config = adapted_state["run_config"]
    if adapted_config.get("source_checkpoint_sha256") != sha256(baseline):
        raise ValueError("Adapted checkpoint did not originate from the registered baseline")
    if adapted_config.get("protocol_manifest_sha256") != sha256(args.training_protocol):
        raise ValueError("Adapted checkpoint training protocol mismatch")
    del adapted_state

    device = torch.device(args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dataset = VietnamDataset(args.benchmark.parent, 256, split="test", csv_path=test_csv)
    args.output.mkdir(parents=True)
    paths = {"baseline_v3": baseline, "adapted_v4": args.adapted_checkpoint}
    results = {}
    for name, checkpoint in paths.items():
        model = load_model(checkpoint, device)
        destination = args.output / name
        destination.mkdir()
        results[name] = evaluate_domain(
            model, make_loader(dataset, args.batch_size, args.workers, 2026),
            device, destination, lpips_model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = [key for key in results["baseline_v3"] if key != "count"]
    report = {
        "benchmark_role": "development benchmark; V3 results informed V4 design, so this is not an untouched final test for V4",
        "protocol": benchmark["protocol"], "protocol_sha256": sha256(args.benchmark),
        "test_count": len(dataset), "videos": benchmark["test"]["videos"],
        "checkpoint_sha256": {name: sha256(path) for name, path in paths.items()},
        "lpips_computed": not args.skip_lpips, "results": results,
        "adapted_v4_minus_baseline_v3": {
            key: results["adapted_v4"][key] - results["baseline_v3"][key] for key in metrics},
    }
    write_json(args.output / "comparison.json", report)
    write_csv(args.output / "comparison.csv",
              [{"model": name, **values} for name, values in results.items()])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
