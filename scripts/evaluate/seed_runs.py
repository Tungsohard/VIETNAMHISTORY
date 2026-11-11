"""Fine-tune the "historical" model again with other seeds to see how much the result varies.

Every run starts from vietcolornet_vivid.pth and only the seed changes. Checkpoints go to
checkpoints/runs/seed_<n>/ (use epoch_03.pth, as for the released model). An interrupted run is resumed.

Run: python scripts/evaluate/seed_runs.py 1 2
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "vietcolornet_v15_vicow.json"


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2]
    for seed in seeds:
        output = ROOT / "checkpoints" / "runs" / f"seed_{seed}"
        config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        config["seed"] = seed
        history = output / "history.csv"
        epochs_done = max(0, len(history.read_text(encoding="utf-8").splitlines()) - 1) if history.exists() else 0
        if (output / "seed_done.json").exists() or epochs_done >= config["epochs"]:
            print(f"seed {seed}: already trained, skipping", flush=True)
            continue
        config_path = ROOT / "configs" / f"seed_run_{seed}.generated.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, str(ROOT / "fine_tune_vietnam.py"),
                   "--config", str(config_path),
                   "--protocol", str(ROOT / "data/protocols/vietnam_modern_vicow_v15/manifest.json"),
                   "--imagenet-manifest", str(ROOT / "data/processed/imagenet_targeted_v10.json"),
                   "--source-checkpoint", str(ROOT / "checkpoints/models/vietcolornet_vivid.pth"),
                   "--output", str(output)]
        # best.pth exists as soon as the baseline is validated, so an interrupted run is resumed
        # from its newest recovery checkpoint instead of being restarted or skipped.
        for name in ("resume_latest.pth", "latest.pth"):
            if (output / name).exists():
                command += ["--resume", str(output / name)]
                break
        print(f"seed {seed}: training -> {output}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        (output / "seed_done.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")


if __name__ == "__main__":
    main()
