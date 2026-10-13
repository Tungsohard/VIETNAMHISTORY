"""Reproduce V7-A epoch 1, then train/audit source-aware V7.1 variants."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "reports" / "vietcolornet_v7_1_pipeline_status.json"
ROADMAP = ROOT / "VIETCOLORNET_SYSTEM_UPDATES.md"
SOURCE = ROOT / "runs" / "vietcolornet_v6_refined" / "best.pth"
PROTOCOL = ROOT / "data" / "vietnam_historical_still_replay_v4" / "manifest.json"
DATASET = ROOT / "data" / "historical_vietnam_images"

VARIANTS = {
    "V7.1-A": {
        "epochs": 1, "scheduler_t_max": 1, "early_stopping_patience": 0,
        "save_every_epoch": True, "semantic_gate_mode": "zero_scalar",
        "source_aware_chroma": False, "use_palette_moe": False,
    },
    "V7.1-B": {
        "epochs": 6, "scheduler_t_max": 6, "early_stopping_patience": 3,
        "save_every_epoch": True, "semantic_gate_mode": "channel_sigmoid",
        "semantic_gate_init_bias": -3.0, "semantic_gate_warmup_epochs": 2,
        "semantic_learning_rate": 1e-5, "source_aware_chroma": True,
        "source_aware_weight": 0.35, "historical_video_ratio_ceiling": 1.08,
        "historical_video_over_weight": 3.0, "historical_still_ratio_floor": 0.90,
        "historical_still_ratio_ceiling": 1.15, "historical_still_under_weight": 1.5,
        "historical_still_over_weight": 0.5, "default_ratio_floor": 0.90,
        "default_ratio_ceiling": 1.10, "default_ratio_weight": 0.5,
        "use_palette_moe": False,
    },
    "V7.1-C": {
        "epochs": 6, "scheduler_t_max": 6, "early_stopping_patience": 3,
        "save_every_epoch": True, "semantic_gate_mode": "channel_sigmoid",
        "semantic_gate_init_bias": -3.0, "semantic_gate_warmup_epochs": 2,
        "semantic_learning_rate": 1e-5, "dino_frozen": False,
        "dino_trainable_blocks": 2, "dino_learning_rate": 1e-6,
        "source_aware_chroma": True, "source_aware_weight": 0.35,
        "historical_video_ratio_ceiling": 1.08, "historical_video_over_weight": 3.0,
        "historical_still_ratio_floor": 0.90, "historical_still_ratio_ceiling": 1.15,
        "historical_still_under_weight": 1.5, "historical_still_over_weight": 0.5,
        "default_ratio_floor": 0.90, "default_ratio_ceiling": 1.10,
        "default_ratio_weight": 0.5, "use_palette_moe": False,
    },
}


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def update(**values):
    state = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if values.get("status") != "failed":
        state.pop("error", None); state.pop("failed_at", None)
    state.update(values, updated_at=now())
    STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def note(message):
    with ROADMAP.open("a", encoding="utf-8") as f: f.write(f"\n- {now()}: {message}\n")


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def learned_checkpoint(run):
    rows = list(csv.DictReader((run / "history.csv").open(encoding="utf-8", newline="")))
    key = "val_source_balanced_selection_score"
    winner = min(rows, key=lambda row: float(row[key]))
    checkpoint = run / f"epoch_{int(winner['epoch']):02d}.pth"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    return checkpoint, winner


def audit(name, checkpoint):
    slug = name.lower().replace(".", "_").replace("-", "_")
    selection = ROOT / "reports" / f"vietcolornet_{slug}_selection.json"
    selection.write_text(json.dumps({"criterion": "best learned epoch; epoch 0 excluded",
        "winner": {"name": name, "checkpoint": str(checkpoint), "sha256": sha(checkpoint),
                   "model": "vietcolornet_v7"}}, indent=2), encoding="utf-8")
    output = ROOT / "reports" / f"vietcolornet_{slug}_direct_audit_100"
    if not (output / "report.json").exists():
        execute([sys.executable, "evaluate_image_failures.py", "--dataset", str(DATASET),
                 "--selection", str(selection), "--output", str(output), "--limit", "100",
                 "--device", "cuda"], ROOT / "reports" / f"vietcolornet_{slug}_audit.log")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output / "per_image.csv").open(encoding="utf-8-sig", newline="")))
    ratios = [float(r["pred_chroma"]) / max(float(r["target_chroma"]), 1e-8) for r in rows]
    report["chroma_ratio"] = report["mean"]["pred_chroma"] / report["mean"]["target_chroma"]
    report["under_80_percent"] = sum(r < 0.8 for r in ratios)
    report["over_125_percent"] = sum(r > 1.25 for r in ratios)
    return report, selection


def main():
    update(status="preparing", variants=list(VARIANTS))
    base = json.loads((ROOT / "configs" / "vietcolornet_v7_a_dino.json").read_text(encoding="utf-8"))
    outcomes = []
    try:
        for name, overrides in VARIANTS.items():
            slug = name.lower().replace(".", "_").replace("-", "_")
            config = {**base, **overrides, "ablation": name, "freeze_encoder_epochs": overrides["epochs"]}
            config_path = ROOT / "configs" / f"vietcolornet_{slug}.generated.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            run = ROOT / "runs" / f"vietcolornet_{slug}"
            run.mkdir(parents=True, exist_ok=True)
            if not (run / "history.csv").exists():
                update(status="training", current_variant=name, log=str(run / "train.log"))
                execute([sys.executable, "fine_tune_vietnam.py", "--config", str(config_path),
                         "--protocol", str(PROTOCOL), "--source-checkpoint", str(SOURCE),
                         "--output", str(run)], run / "train.log")
            checkpoint, validation = learned_checkpoint(run)
            update(status="auditing", current_variant=name, checkpoint=str(checkpoint))
            report, selection = audit(name, checkpoint)
            outcomes.append({"name": name, "checkpoint": str(checkpoint), "sha256": sha(checkpoint),
                             "validation": validation, "audit": report, "selection": str(selection)})
            note(f"{name} hoàn tất: learned epoch {validation['epoch']}, audit chroma ratio="
                 f"{report['chroma_ratio']:.4f}, dưới 80%={report['under_80_percent']}/100, "
                 f"trên 125%={report['over_125_percent']}/100.")
        winner = min(outcomes, key=lambda row: float(row["validation"]["val_source_balanced_selection_score"]))
        v6 = json.loads((ROOT / "reports" / "vietcolornet_v6_refined_direct_audit_100" / "report.json").read_text(encoding="utf-8"))["mean"]
        m, a = winner["audit"]["mean"], winner["audit"]
        gates = {
            "delta_e_not_worse_v6": m["delta_e2000"] <= v6["delta_e2000"],
            "ab_mae_not_worse_v6": m["ab_mae"] <= v6["ab_mae"],
            "edge_not_worse_v6": m["edge_ab_mae"] <= v6["edge_ab_mae"],
            "lpips_not_worse_v6": m["lpips"] <= v6["lpips"],
            "chroma_ratio_0_90_to_1_10": 0.90 <= a["chroma_ratio"] <= 1.10,
            "under_80_below_30": a["under_80_percent"] < 30,
            "over_125_at_most_15": a["over_125_percent"] <= 15,
        }
        passed = all(gates.values())
        result = {"winner": winner, "variants": outcomes, "gates": gates,
                  "passed_all_gates": passed,
                  "decision": "eligible_for_app_promotion" if passed else "reject_keep_v6_local"}
        path = ROOT / "reports" / "VIETCOLORNET_V7_1_OUTCOME.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        note(f"V7.1 kết thúc với `{result['decision']}`; ứng viên {winner['name']}.")
        update(status="completed", winner=winner["name"], checkpoint=winner["checkpoint"],
               outcome=str(path), passed_all_gates=passed, decision=result["decision"], completed_at=now())
    except Exception as error:
        update(status="failed", error=repr(error), failed_at=now())
        note(f"Pipeline V7.1 lỗi: `{error!r}`.")
        raise


if __name__ == "__main__": main()
