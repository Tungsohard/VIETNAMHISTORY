"""Assemble a ready-to-upload Hugging Face Space (Gradio, CPU) in ``build/hf_space``.

The Space contains the code, the historical-model checkpoint stripped to inference weights (the training
checkpoint also stores optimizer state and is twice as large) and, with ``--with-sam``, the
SAM ViT-B weights. Checkpoints are not stored in the GitHub repository, so this is the only
place they are packaged for deployment.

Usage:
    python scripts/deploy/build_hf_space.py [--with-sam]
Then upload the folder (see the printed instructions).
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "models" / "vietcolornet_historical.pth"
SAM = ROOT / "checkpoints" / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"

README = """---
title: Sắc Việt
emoji: 🎨
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

Sắc Việt colourises Vietnamese historical photographs with VietColorNet.
Source code: https://github.com/Tungsohard/VIETNAMHISTORY
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
numpy>=1.24,<3
opencv-python-headless>=4.8,<5
Pillow>=10,<13
scikit-image>=0.22,<1
lpips==0.1.4
PyYAML>=6,<7
scipy>=1.10,<2
requests>=2.31,<3
gradio==5.49.1
imageio-ffmpeg>=0.5,<1
pyarrow>=17,<24
segment-anything==1.0
huggingface-hub>=0.23,<2
tqdm>=4.66,<5
"""

APP = '''"""Hugging Face Space entry point: CPU-only Sắc Việt demo."""
import os
import sys

import demo

sys.argv = ["demo.py", "--host", "0.0.0.0", "--port", os.environ.get("PORT", "7860"), "--device", "cpu"]
if not os.path.isfile("checkpoints/pretrained/sam/sam_vit_b_01ec64.pth"):
    sys.argv.append("--no-sam")
demo.main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "hf_space")
    parser.add_argument("--with-sam", action="store_true", help="Also package SAM ViT-B (slow on CPU)")
    args = parser.parse_args()
    out = args.output
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(ROOT / "colorization", out / "colorization", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "demo.py", out / "demo.py")
    shutil.copytree(ROOT / "examples", out / "examples")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "app.py").write_text(APP, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pth filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    folder = out / "checkpoints" / "models"
    folder.mkdir(parents=True)
    torch.save({"epoch": checkpoint["epoch"], "model_state_dict": checkpoint["model_state_dict"],
                "run_config": checkpoint["run_config"]}, folder / "vietcolornet_historical.pth")
    if args.with_sam:
        (out / "checkpoints" / "pretrained" / "sam").mkdir(parents=True)
        shutil.copy2(SAM, out / "checkpoints" / "pretrained" / "sam" / SAM.name)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Built {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
