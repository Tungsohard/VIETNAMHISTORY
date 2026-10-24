"""Download the official Segment Anything ViT-B checkpoint used for bleed-free colour refinement.

SAM (Kirillov et al., 2023) is released by Meta AI under Apache-2.0. The file name carries
the first six hex digits of its MD5, which is verified after download.
"""
import argparse
import hashlib
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "pretrained" / "sam" / "sam_vit_b_01ec64.pth")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        temporary = args.output.with_suffix(".part")
        with requests.get(URL, stream=True, timeout=60) as response, temporary.open("wb") as handle:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
        temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"SAM checkpoint checksum mismatch: {digest}")
    print(f"SAM ViT-B ready: {args.output}")


if __name__ == "__main__":
    main()
