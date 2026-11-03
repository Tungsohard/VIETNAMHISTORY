"""Verify files and checkpoint integrity required by the local still-image product."""
import argparse
import json
import sys
from pathlib import Path

import torch

from colorization.data import sha256


ROOT = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "outputs" / "reports" / "baseline_selection.json")
    parser.add_argument("--load-model", action="store_true",
                        help="Also construct the model and load all checkpoint weights")
    args = parser.parse_args()

    required = [
        ROOT / "demo.py",
        ROOT / "start_demo.bat",
        ROOT / "colorization",
        ROOT / "third_party" / "DDColor",
        args.selection,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing product files:\n" + "\n".join(missing))

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    winner = selection["winner"]
    checkpoint = Path(winner["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (args.selection.resolve().parent / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Product checkpoint not found: {checkpoint}")
    actual_sha = sha256(checkpoint)
    if actual_sha != winner["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual_sha}")

    print(f"Selection: {args.selection}")
    print(f"Model: {winner.get('display_name', winner.get('name', 'DDColor'))}")
    print(f"Checkpoint: {checkpoint}")
    print(f"SHA-256: {actual_sha}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.load_model:
        from colorization.inference import ColorizationEngine
        engine = ColorizationEngine.from_selection(args.selection)
        print(f"Model load: OK ({engine.device_name})")
    print("Product check: OK")


if __name__ == "__main__":
    main()
