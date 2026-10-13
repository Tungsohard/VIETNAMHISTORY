"""Check the local environment, data protocol and model imports without training."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from colorization.data import ImageNetDataset, sha256


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "protocol_manifest_v3.json")
    parser.add_argument("--check-ddcolor", action="store_true")
    args = parser.parse_args()
    dependency_status = {name: bool(importlib.util.find_spec(name))
                         for name in ("skimage", "lpips", "huggingface_hub")}
    print("Python dependencies: " + ", ".join(f"{key}={value}" for key, value in dependency_status.items()))
    print(f"PyTorch={torch.__version__}, CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"Protocol={manifest['protocol']}")
    print("Counts=" + ", ".join(f"{name}:{info['count']:,}" for name, info in manifest["splits"].items()))
    for split in ("train", "val", "test"):
        dataset = ImageNetDataset(args.manifest, split, 64, 1)
        sample = dataset[0]
        assert sample["L"].shape == (1, 64, 64) and sample["ab"].shape == (2, 64, 64)
        assert np.isfinite(sample["rgb"].numpy()).all()
    if args.check_ddcolor:
        from colorization.model import build_model
        model = build_model({"model": "ddcolor", "encoder_name": "convnext-t", "image_size": 256,
                             "nf": 512, "num_queries": 100, "decoder_layers": 9})
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 256, 256))
        assert output.shape == (1, 2, 256, 256) and torch.isfinite(output).all()
        print("Official DDColor-T forward check: OK")
        checkpoint = ROOT / "pretrained" / "ddcolor_paper_tiny.pth"
        if checkpoint.exists():
            model.load_official(checkpoint)
            with torch.inference_mode():
                output = model(torch.zeros(1, 1, 256, 256))
            assert output.shape == (1, 2, 256, 256) and torch.isfinite(output).all()
            print(f"Official DDColor-T checkpoint load: OK ({checkpoint.stat().st_size:,} bytes)")
    print(f"Manifest SHA-256={sha256(args.manifest)}")
    print("Data/model check: OK")
    missing = [name for name, installed in dependency_status.items() if not installed]
    if missing:
        print("Environment incomplete; install requirements before evaluation: " + ", ".join(missing))
    if not torch.cuda.is_available():
        print("Training unavailable on GPU in this Python environment; install a CUDA-enabled PyTorch build")


if __name__ == "__main__":
    main()
