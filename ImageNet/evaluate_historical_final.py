"""Evaluate frozen V2 and V3 selections once on the external historical-video final test."""
import argparse
import json
from pathlib import Path

import torch

from colorization.data import VietnamDataset, sha256
from colorization.runtime import make_loader, write_csv, write_json
from evaluate import evaluate_domain
from evaluate_vietnam_adaptation import load_model


ROOT = Path(__file__).resolve().parent


def selected_checkpoint(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    path = Path(selection["winner"]["checkpoint"])
    if sha256(path) != selection["winner"]["sha256"]:
        raise ValueError(f"Checkpoint fingerprint mismatch: {selection_path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-selection", type=Path, default=ROOT / "reports" / "vietnam_v2_selection.json")
    parser.add_argument("--v3-selection", type=Path, default=ROOT / "reports" / "vietnam_v3_selection.json")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data" / "historical_vietnam_final_test_v1" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "historical_vietnam_final_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Final evaluation already exists: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "historical_vietnam_final_test_v1":
        raise ValueError("Unexpected final-test protocol")
    if protocol.get("status") != "FROZEN_DO_NOT_USE_FOR_TRAINING_OR_MODEL_SELECTION":
        raise ValueError("Historical test is not frozen")
    test_csv = args.protocol.parent / protocol["test"]["path"]
    if sha256(test_csv) != protocol["test"]["sha256"]:
        raise ValueError("Historical test CSV fingerprint mismatch")
    for relative, expected in protocol["frame_sha256"].items():
        if sha256(args.protocol.parent / relative) != expected:
            raise ValueError(f"Historical frame fingerprint mismatch: {relative}")

    paths = {"v2": selected_checkpoint(args.v2_selection),
             "v3": selected_checkpoint(args.v3_selection)}
    device = torch.device(args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dataset = VietnamDataset(args.protocol.parent, 256, split="test", csv_path=test_csv)
    args.output.mkdir(parents=True)
    results = {}
    for name, path in paths.items():
        model = load_model(path, device)
        destination = args.output / name
        destination.mkdir()
        loader = make_loader(dataset, args.batch_size, args.workers, 2026)
        results[name] = evaluate_domain(model, loader, device, destination, lpips_model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = [key for key in results["v2"] if key != "count"]
    report = {"protocol": protocol["protocol"], "protocol_sha256": sha256(args.protocol),
              "test_count": len(dataset), "videos": protocol["test"]["videos"],
              "checkpoint_sha256": {name: sha256(path) for name, path in paths.items()},
              "lpips_computed": not args.skip_lpips, "results": results,
              "v3_minus_v2": {key: results["v3"][key] - results["v2"][key] for key in metrics}}
    write_json(args.output / "comparison.json", report)
    write_csv(args.output / "comparison.csv", [{"model": name, **values} for name, values in results.items()])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
