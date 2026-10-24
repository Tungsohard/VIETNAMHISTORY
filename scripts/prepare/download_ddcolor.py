"""Download an official ImageNet-only DDColor checkpoint from its model author."""
import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "DDColor"
sys.path.insert(0, str(REPO))
from ddcolor import DDColor


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ddcolor_paper_tiny", "ddcolor_paper"],
                        default="ddcolor_paper_tiny")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "checkpoints" / "pretrained" / f"{args.model}.pth"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = DDColorHF.from_pretrained(f"piddnad/{args.model}")
    torch.save({"params": model.state_dict(), "source": f"piddnad/{args.model}"}, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
