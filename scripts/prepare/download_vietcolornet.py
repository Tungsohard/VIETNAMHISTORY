"""Download the two VietColorNet checkpoints (inference weights) from the GitHub Release of this repo.

Files go to checkpoints/models/ and are checked against the SHA-256 listed below.
Run: python scripts/prepare/download_vietcolornet.py
"""
import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://github.com/Tungsohard/VIETNAMHISTORY/releases/download/weights-v1/"
FILES = {
    "vietcolornet_historical.pth": "fa4b6eeb4d4de2dc8be4f3d683bdba02e490044a6b6523725e611d1948d41dda",
    "vietcolornet_vivid.pth": "0aa793c31c2c903eb6402af5c1842762b04b0f276988dd2430f03a20b635ef82",
}


def download(name, expected):
    target = ROOT / "checkpoints" / "models" / name
    if target.exists():
        print(f"{name}: already there, skipped")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    digest = hashlib.sha256()
    with requests.get(BASE + name, stream=True, timeout=60) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        part.unlink()
        sys.exit(f"{name}: wrong SHA-256 ({digest.hexdigest()}), file removed")
    part.rename(target)
    print(f"{name}: ok")


if __name__ == "__main__":
    for file_name, sha in FILES.items():
        download(file_name, sha)
