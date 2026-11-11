"""Build the blind survey: 20 greyscale images, each coloured by three models shown in random order.
Participants pick the most natural and real-looking one. Model names are kept only in key.json,
which must not be shared. All models use the same pipeline (natural mode + SAM).

Run: python scripts/evaluate/user_study/build_study.py --output build/user_study
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))
from compare_checkpoints import load_set  # noqa: E402
from statistical_comparison import CachedSegmenter, grey_input, load_engine  # noqa: E402

MODELS = {"historical": "checkpoints/models/vietcolornet_historical.pth",
          "vivid": "checkpoints/models/vietcolornet_vivid.pth",
          "ddcolor": "@selection"}
MAX_SIDE = 800


def pick(images, count):
    indices = np.linspace(0, len(images) - 1, count).round().astype(int)
    return [images[i] for i in indices]


def collect_images():
    items = []
    for path in sorted((ROOT / "examples").glob("*.jpg")):
        if path.stem == "dai_hoi_3_san_sang_1965":  # a colour film frame, not an archival photograph
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        items.append(("archive", np.asarray(image)))
    for rgb in pick(load_set("vicow-test", 1, MAX_SIDE), 8):
        items.append(("film", rgb))
    for rgb in pick(load_set("modern-test", 1, MAX_SIDE), 7):
        items.append(("modern", rgb))
    return items


def write_page(args, template, trials):
    page = (template.replace("__TRIALS__", json.dumps(trials))
            .replace("__FORM_URL__", args.form_url).replace("__FORM_ENTRY__", args.form_entry))
    (args.output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "user_study")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--form-url", default="", help="Google Form .../formResponse URL that receives the answers")
    parser.add_argument("--form-entry", default="", help="the form field name, e.g. entry.1234567890")
    parser.add_argument("--reuse-images", action="store_true",
                        help="only rewrite index.html from the existing trials.json (no model runs)")
    args = parser.parse_args()

    template = (Path(__file__).parent / "study_template.html").read_text(encoding="utf-8")
    if args.reuse_images:
        trials = json.loads((args.output / "trials.json").read_text(encoding="utf-8"))
        write_page(args, template, trials)
        return

    from colorization.segmentation import ObjectSegmenter
    segmenter = CachedSegmenter(ObjectSegmenter.try_create(device=args.device))
    if segmenter.segmenter is None:
        raise SystemExit("SAM is unavailable")

    items = collect_images()
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    grey = {i: grey_input(items[i][1]) for i in order}

    outputs = {}
    for name, spec in MODELS.items():
        engine = load_engine(spec, args.device, segmenter)
        for i in order:
            result = engine.colorize_pil(Image.fromarray(grey[i]), 1.0, mode="natural")
            outputs[(name, i)] = result
        del engine
        import torch
        torch.cuda.empty_cache()
        print(f"{name}: done", flush=True)

    key, trials = {}, []
    for position, i in enumerate(order):
        code = f"s{position + 1:02d}"
        labels = ["A", "B", "C"]
        names = list(MODELS)
        rng.shuffle(names)
        Image.fromarray(grey[i]).convert("L").save(images_dir / f"{code}_grey.jpg", quality=90)
        key[code] = {"group": items[i][0], "A": names[0], "B": names[1], "C": names[2]}
        for label, name in zip(labels, names):
            outputs[(name, i)].save(images_dir / f"{code}_{label}.jpg", quality=90)
        trials.append(code)

    (args.output / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (args.output / "trials.json").write_text(json.dumps(trials), encoding="utf-8")
    write_page(args, template, trials)
    print(f"Study written to {args.output} ({len(trials)} images). Share everything except key.json.")


if __name__ == "__main__":
    main()
