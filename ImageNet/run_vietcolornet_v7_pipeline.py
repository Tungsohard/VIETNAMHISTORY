"""Run V7 smoke test, ablations, candidate selection and 100-image audit."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "reports" / "vietcolornet_v7_pipeline_status.json"
ROADMAP = ROOT / "VIETCOLORNET_SYSTEM_UPDATES.md"
SOURCE = ROOT / "runs" / "vietcolornet_v6_refined" / "best.pth"
PROTOCOL = ROOT / "data" / "vietnam_historical_still_replay_v4" / "manifest.json"
AUDIT_DATASET = ROOT / "data" / "historical_vietnam_images"
ABLATIONS = [
    ("V7-A", ROOT / "configs" / "vietcolornet_v7_a_dino.json", ROOT / "runs" / "vietcolornet_v7_a_dino"),
    ("V7-B", ROOT / "configs" / "vietcolornet_v7_b_palette.json", ROOT / "runs" / "vietcolornet_v7_b_palette"),
    ("V7-C", ROOT / "configs" / "vietcolornet_v7_c_moe.json", ROOT / "runs" / "vietcolornet_v7_c_moe"),
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update(**fields):
    state = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if fields.get("status") != "failed":
        state.pop("error", None)
        state.pop("failed_at", None)
    state.update(fields, updated_at=now())
    STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def note(text):
    with ROADMAP.open("a", encoding="utf-8") as handle:
        handle.write(f"- {now()}: {text}\n")


def run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def train(name, config, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pth").exists() and not (output / "pending_validation.pth").exists():
        note(f"{name}: tìm thấy checkpoint đã hoàn tất; tái sử dụng.")
        return
    command = [sys.executable, "fine_tune_vietnam.py", "--config", str(config),
               "--protocol", str(PROTOCOL), "--source-checkpoint", str(SOURCE),
               "--output", str(output)]
    latest = output / "latest.pth"
    pending = output / "pending_validation.pth"
    resume = pending if pending.exists() else latest if latest.exists() else None
    if resume:
        command += ["--resume", str(resume)]
    update(status="training", current_ablation=name, log=str(output / "train.log"))
    run(command, output / "train.log")
    note(f"{name}: huấn luyện hoàn tất; checkpoint `{output.relative_to(ROOT) / 'best.pth'}`.")


def validation_score(checkpoint):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = state.get("validation", {})
    score = metrics.get("val_source_balanced_selection_score",
                        metrics.get("val_selection_score", state.get("best_validation_score", float("inf"))))
    return float(score), int(state.get("epoch", 0)), metrics


def main():
    update(status="initializing", source_checkpoint=str(SOURCE), protocol=str(PROTOCOL),
           ablations=[item[0] for item in ABLATIONS])
    try:
        # Short end-to-end smoke test before spending hours on full ablations.
        smoke_dir = ROOT / "runs" / "vietcolornet_v7_smoke"
        if not (smoke_dir / "history.csv").exists():
            smoke_config = json.loads(ABLATIONS[-1][1].read_text(encoding="utf-8"))
            smoke_config.update(epochs=1, scheduler_t_max=1, early_stopping_patience=0,
                                num_workers=0, validation_workers=0)
            smoke_path = ROOT / "configs" / "vietcolornet_v7_smoke.generated.json"
            smoke_path.write_text(json.dumps(smoke_config, indent=2), encoding="utf-8")
            smoke_dir.mkdir(parents=True, exist_ok=True)
            update(status="smoke_testing", current_ablation="V7-smoke")
            run([sys.executable, "fine_tune_vietnam.py", "--config", str(smoke_path),
                 "--protocol", str(PROTOCOL), "--source-checkpoint", str(SOURCE),
                 "--output", str(smoke_dir), "--train-limit", "8", "--val-limit", "4"],
                smoke_dir / "train.log")
            note("Smoke test V7 đạt forward/backward, validation và lưu/nạp checkpoint.")
        for item in ABLATIONS:
            train(*item)
        candidates = []
        for name, config, output in ABLATIONS:
            checkpoint = output / "best.pth"
            score, epoch, metrics = validation_score(checkpoint)
            candidates.append({"name": name, "config": str(config), "checkpoint": str(checkpoint),
                               "sha256": sha256(checkpoint), "epoch": epoch,
                               "validation_score": score, "validation": metrics})
        winner = min(candidates, key=lambda row: row["validation_score"])
        selection = {
            "criterion": "minimum source-balanced chroma-penalized validation score among V7 ablations",
            "winner": {"name": winner["name"], "checkpoint": winner["checkpoint"],
                       "sha256": winner["sha256"], "epoch": winner["epoch"],
                       "model": "vietcolornet_v7", "backbone": "DINOv2-S/14 frozen semantic bridge"},
            "candidates": candidates,
        }
        selection_path = ROOT / "reports" / "vietcolornet_v7_selection.json"
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        audit = ROOT / "reports" / "vietcolornet_v7_direct_audit_100"
        if not (audit / "report.json").exists():
            update(status="auditing", current_ablation=winner["name"], selection=str(selection_path))
            run([sys.executable, "evaluate_image_failures.py", "--dataset", str(AUDIT_DATASET),
                 "--selection", str(selection_path), "--output", str(audit), "--limit", "100",
                 "--device", "cuda"], ROOT / "reports" / "vietcolornet_v7_audit.log")
        report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
        with (audit / "per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ratios = [float(row["pred_chroma"]) / max(float(row["target_chroma"]), 1e-8) for row in rows]
        chroma_ratio = report["mean"]["pred_chroma"] / report["mean"]["target_chroma"]
        under = sum(value < 0.8 for value in ratios)
        over = sum(value > 1.25 for value in ratios)
        v6 = json.loads((ROOT / "reports" / "vietcolornet_v6_refined_direct_audit_100" / "report.json").read_text(encoding="utf-8"))["mean"]
        mean = report["mean"]
        gates = {
            "delta_e2000_not_worse_than_v6": mean["delta_e2000"] <= v6["delta_e2000"],
            "ab_mae_not_worse_than_v6": mean["ab_mae"] <= v6["ab_mae"],
            "edge_ab_mae_not_worse_than_v6": mean["edge_ab_mae"] <= v6["edge_ab_mae"],
            "lpips_not_worse_than_v6": mean["lpips"] <= v6["lpips"],
            "chroma_ratio_0_90_to_1_10": 0.90 <= chroma_ratio <= 1.10,
            "under_80_below_30": under < 30,
            "over_125_at_most_15": over <= 15,
        }
        passed = all(gates.values())
        outcome = {"winner": winner, "audit": report, "chroma_ratio": chroma_ratio,
                   "under_80_percent": under, "over_125_percent": over,
                   "gates": gates, "passed_all_gates": passed,
                   "decision": "eligible_for_app_promotion" if passed else "reject_keep_v6_local"}
        outcome_path = ROOT / "reports" / "VIETCOLORNET_V7_OUTCOME.json"
        outcome_path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
        note(f"Audit V7 hoàn tất: chroma ratio={chroma_ratio:.4f}, dưới 80%={under}/100, "
             f"trên 125%={over}/100; quyết định `{outcome['decision']}`.")
        update(status="completed", winner=winner["name"], checkpoint=winner["checkpoint"],
               audit=str(audit / "report.json"), outcome=str(outcome_path),
               passed_all_gates=passed, decision=outcome["decision"], completed_at=now())
    except Exception as error:
        update(status="failed", error=repr(error), failed_at=now())
        note(f"Pipeline V7 lỗi: `{error!r}`.")
        raise


if __name__ == "__main__":
    main()
