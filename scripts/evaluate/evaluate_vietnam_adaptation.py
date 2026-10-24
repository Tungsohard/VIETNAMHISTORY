"""Compare a registered baseline and its adapted successor on held-out Vietnam videos."""
import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from colorization.data import VietnamDataset, sha256
from colorization.model import build_model
from colorization.runtime import load_checkpoint, make_loader, write_csv, write_json
from evaluate import evaluate_domain


ROOT = Path(__file__).resolve().parents[2]


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    config = {**checkpoint["run_config"], "pretrained_encoder": False}
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-selection", type=Path, default=ROOT / "outputs" / "reports" / "model_selection.json")
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data" / "protocols" / "vietnam_protocol_v1" / "manifest.json")
    parser.add_argument("--vietnam-root", type=Path, default=ROOT / "data" / "sources" / "vicow")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    test_info = protocol["splits"]["test"]
    test_csv = args.protocol.parent / test_info["path"]
    if sha256(test_csv) != test_info["sha256"]:
        raise ValueError("Vietnam test split fingerprint mismatch")
    selection = json.loads(args.baseline_selection.read_text(encoding="utf-8"))
    baseline_path = Path(selection["winner"]["checkpoint"])
    if sha256(baseline_path) != selection["winner"]["sha256"]:
        raise ValueError("Baseline checkpoint fingerprint mismatch")
    adapted = load_checkpoint(args.adapted_checkpoint)
    adapted_config = dict(adapted["run_config"])
    if adapted_config.get("protocol_manifest_sha256") != sha256(args.protocol):
        raise ValueError("Adapted checkpoint used a different Vietnam protocol")
    if adapted_config.get("source_checkpoint_sha256") != sha256(baseline_path):
        raise ValueError("Adapted checkpoint did not originate from this baseline")
    del adapted

    device = torch.device(args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dataset = VietnamDataset(args.vietnam_root, adapted_config["image_size"], args.limit,
                             "test", test_csv)
    args.output.mkdir(parents=True)
    results = {}
    checkpoint_hashes = {}
    for name, path in (("baseline", baseline_path), ("adapted", args.adapted_checkpoint)):
        model = load_model(path, device)
        domain_output = args.output / name
        domain_output.mkdir()
        batches = make_loader(dataset, args.batch_size, args.workers, adapted_config["seed"])
        results[name] = evaluate_domain(model, batches, device, domain_output, lpips_model)
        checkpoint_hashes[name] = sha256(path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = [key for key in results["baseline"] if key != "count"]
    report = {"protocol": protocol["protocol"], "held_out_videos": test_info["videos"],
              "test_csv_sha256": sha256(test_csv), "checkpoint_sha256": checkpoint_hashes,
              "partial_evaluation": bool(args.limit), "lpips_computed": not args.skip_lpips,
              "results": results,
              "adapted_minus_baseline": {key: results["adapted"][key] - results["baseline"][key]
                                           for key in metrics}}
    write_json(args.output / "comparison.json", report)
    write_csv(args.output / "comparison.csv", [{"model": name, **values} for name, values in results.items()])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
