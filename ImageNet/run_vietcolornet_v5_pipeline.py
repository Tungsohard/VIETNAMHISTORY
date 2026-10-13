"""Run all VietColorNet V5 ablations and register the best full candidate."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "data" / "vietnam_historical_still_replay_v4" / "manifest.json"
SOURCE = ROOT / "runs" / "vietcolornet_master_v4" / "best.pth"
VIETNAM_ROOT = ROOT.parent / "dataset" / "VietNam_dataset"
STATE_PATH = ROOT / "reports" / "vietcolornet_v5_pipeline_status.json"


BASE = {
    "model": "vietcolornet_v5",
    "backbone": "resnet34",
    "pretrained_encoder": False,
    "num_palette_queries": 32,
    "max_chroma": 48.0,
    "image_size": 256,
    "color_weight_scale": 0.12,
    "color_pixel_boost": 2.0,
    "weighted_ab_weight": 0.5,
    "chroma_loss_weight": 0.75,
    "mean_chroma_weight": 0.65,
    "hue_loss_weight": 0.05,
    "hue_chroma_threshold": 0.03125,
    "edge_chroma_loss_weight": 0.25,
    "edge_gamma": 0.7,
    "selection_chroma_weight": 0.04,
    "selection_chroma_min_ratio": 0.9,
    "selection_chroma_max_ratio": 1.1,
    "selection_chroma_band_weight": 0.3,
    "selection_source_balanced": True,
    "validation_source_breakdown": True,
    "save_every_epoch": False,
    "freeze_encoder_epochs": 1,
    "epochs": 15,
    "scheduler_t_max": 15,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.00005,
    "weight_decay": 0.0001,
    "num_workers": 2,
    "validation_workers": 0,
    "seed": 2026,
    "early_stopping_patience": 4,
}

VARIANTS = [
    ("v5_a_head_v1", {"use_palette_attention": False, "loss_version": "v1", "distill_weight": 0.0,
                       "ablation": "V5-A_head_only"}),
    ("v5_b_color_loss", {"use_palette_attention": False, "loss_version": "chroma_aware_v2", "distill_weight": 0.0,
                          "ablation": "V5-B_head_color_loss"}),
    ("v5_c_edge_loss", {"use_palette_attention": False, "loss_version": "hue_chroma_edge_v5", "distill_weight": 0.0,
                         "ablation": "V5-C_head_color_edge_loss"}),
    ("v5_d_cpca", {"use_palette_attention": True, "loss_version": "hue_chroma_edge_v5", "distill_weight": 0.0,
                    "ablation": "V5-D_head_color_edge_cpca"}),
    ("v5_e_distill", {"use_palette_attention": True, "loss_version": "hue_chroma_edge_v5", "distill_weight": 0.15,
                       "distill_saliency_boost": 2.0, "distill_hue_weight": 0.05,
                       "ablation": "V5-E_full_with_v4_regularizer"}),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_state(status: str, **extra):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at": datetime.now().astimezone().isoformat(), **extra}
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def completed(run_dir: Path, config: dict) -> bool:
    history = run_dir / "history.csv"
    log = run_dir / "training.log"
    if not (run_dir / "best.pth").is_file() or not history.is_file():
        return False
    epoch_count = max(0, sum(1 for _ in history.open(encoding="utf-8-sig")) - 1)
    return epoch_count >= config["epochs"] or (log.is_file() and "Early stopping" in log.read_text(encoding="utf-8", errors="replace"))


def train_variant(name: str, overrides: dict):
    config = {**BASE, **overrides}
    config_path = ROOT / "configs" / f"vietcolornet_{name}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_dir = ROOT / "runs" / f"vietcolornet_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if completed(run_dir, config):
        return
    command = [
        sys.executable, str(ROOT / "fine_tune_vietnam.py"), "--config", str(config_path),
        "--protocol", str(PROTOCOL), "--source-checkpoint", str(SOURCE),
        "--output", str(run_dir), "--vietnam-root", str(VIETNAM_ROOT), "--device", "cuda",
    ]
    if name == "v5_e_distill":
        command.extend(["--teacher-checkpoint", str(SOURCE), "--distill-weight", "0.15"])
    save_state("training", variant=name, run_dir=str(run_dir))
    with (run_dir / "training.log").open("w", encoding="utf-8") as stdout, \
            (run_dir / "training.err.log").open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def checkpoint_summary(name: str) -> dict:
    checkpoint = ROOT / "runs" / f"vietcolornet_{name}" / "best.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validation = state.get("validation", {})
    return {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
        "epoch": state.get("epoch"),
        "score": state.get("best_validation_score", state.get("best_val_loss")),
        "validation": validation,
    }


def main():
    if not SOURCE.is_file() or not PROTOCOL.is_file() or not VIETNAM_ROOT.is_dir():
        raise FileNotFoundError("V4 checkpoint, frozen protocol, or ViCoW dataset is missing")
    summaries = []
    for name, overrides in VARIANTS:
        train_variant(name, overrides)
        summaries.append(checkpoint_summary(name))
    # V5-D and V5-E are the only complete V5 candidates; earlier runs are ablations.
    finalists = [row for row in summaries if row["name"] in {"v5_d_cpca", "v5_e_distill"}]
    winner = min(finalists, key=lambda row: row["score"])
    selection = {
        "criterion": "Minimum source-balanced chroma-penalized validation score among complete V5 candidates",
        "status": "awaiting_direct_acceptance_audit",
        "winner": {
            "name": winner["name"],
            "display_name": "VietColorNet V5 Candidate",
            "checkpoint": "../" + Path(winner["checkpoint"]).relative_to(ROOT).as_posix(),
            "sha256": winner["sha256"],
            "epoch": winner["epoch"],
            "model": "vietcolornet_v5",
            "backbone": "resnet34",
        },
        "ablations": summaries,
    }
    selection_path = ROOT / "reports" / "vietcolornet_v5_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    save_state("training_complete", winner=winner, selection=str(selection_path), ablations=summaries)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save_state("failed", error=repr(exc))
        raise
