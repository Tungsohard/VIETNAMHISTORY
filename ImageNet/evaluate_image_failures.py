"""Evaluate V3 per still image and render the worst colorization cases."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from colorization.data import sha256
from colorization.image_dataset import infer_collection_id, infer_content_category
from colorization.inference import ColorizationEngine
from colorization.metrics import image_metrics
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_review(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resize_for_evaluation(image, max_side):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS)


def edge_ab_mae(prediction, target):
    target_lab = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2LAB)
    prediction_lab = cv2.cvtColor(prediction.astype(np.float32), cv2.COLOR_RGB2LAB)
    luminance = np.clip(target_lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    edges = cv2.Canny(luminance, 80, 160)
    mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    if not mask.any():
        return float(np.abs(prediction_lab[..., 1:] - target_lab[..., 1:]).mean())
    return float(np.abs(prediction_lab[..., 1:] - target_lab[..., 1:])[mask].mean())


def summarize(rows, field):
    metrics = ["psnr", "ssim", "delta_e2000", "ab_mae", "edge_ab_mae",
               "pred_chroma", "target_chroma"]
    if rows and "lpips" in rows[0]:
        metrics.append("lpips")
    summaries = []
    for value in sorted({row[field] for row in rows}):
        selected = [row for row in rows if row[field] == value]
        summaries.append({field: value, "count": len(selected),
                          **{metric: float(np.mean([row[metric] for row in selected]))
                             for metric in metrics}})
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_failure_sheets(engine, dataset_root, rows, output, max_side, count=24):
    selected = sorted(rows, key=lambda row: row["failure_score"], reverse=True)[:count]
    columns, cell_width, image_height, label_height = 3, 900, 300, 48
    for page, start in enumerate(range(0, len(selected), 6), 1):
        page_rows = selected[start:start + 6]
        sheet = Image.new("RGB", (columns * cell_width, len(page_rows) * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for y_index, row in enumerate(page_rows):
            with Image.open(dataset_root / row["local_path"]) as source:
                target = resize_for_evaluation(source, max_side)
            prediction = engine.colorize_pil(target)
            gray = ImageOps.grayscale(target).convert("RGB")
            panels = [gray, prediction, target]
            y = y_index * (image_height + label_height)
            for x_index, panel in enumerate(panels):
                panel = ImageOps.contain(panel, (cell_width, image_height))
                x = x_index * cell_width + (cell_width - panel.width) // 2
                sheet.paste(panel, (x, y + (image_height - panel.height) // 2))
            draw.text((6, y + image_height + 3),
                      f"#{start + y_index + 1} {row['content_category']} | DeltaE {row['delta_e2000']:.2f} | edge {row['edge_ab_mae']:.2f}",
                      fill="black")
            draw.text((6, y + image_height + 23), row["title"][:120], fill="black")
        sheet.save(output / f"worst_cases_{page:02d}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Collector output containing accepted.jsonl and images/")
    parser.add_argument("--review", type=Path,
                        help="Manual review CSV; defaults to DATASET/manual_review.csv")
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "reports" / "vietnam_v3_selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worst-count", type=int, default=24)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")
    if args.max_side < 256 or args.worst_count < 1:
        parser.error("--max-side must be at least 256 and --worst-count must be positive")
    accepted_path = args.dataset / "accepted.jsonl"
    rows = read_jsonl(accepted_path)
    review_path = args.review or args.dataset / "manual_review.csv"
    if not review_path.is_file():
        raise FileNotFoundError(f"Missing manual review: {review_path}")
    review = read_review(review_path)
    decisions = {row["decision"].strip().lower() for row in review}
    invalid = decisions - {"keep", "reject"}
    if invalid:
        raise ValueError(f"Every review decision must be keep or reject; found: {sorted(invalid)}")
    accepted_by_path = {row["local_path"]: row for row in rows}
    review_by_path = {row["local_path"]: row for row in review}
    if len(review) != len(rows) or set(review_by_path) != set(accepted_by_path):
        raise ValueError("Manual review rows do not match accepted.jsonl")
    rows = [
        {**row, **review_by_path[row["local_path"]]}
        for row in rows
        if review_by_path[row["local_path"]]["decision"].strip().lower() == "keep"
    ]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    args.output.mkdir(parents=True)
    engine = ColorizationEngine.from_selection(args.selection, device=args.device)
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(engine.device).eval()

    results = []
    for row in tqdm(rows, desc="Evaluate still images"):
        path = args.dataset / row["local_path"]
        if sha256(path) != row["local_sha256"]:
            raise ValueError(f"Image fingerprint mismatch: {path}")
        with Image.open(path) as source:
            target_pil = resize_for_evaluation(source, args.max_side)
        prediction_pil = engine.colorize_pil(target_pil)
        target = np.asarray(target_pil, dtype=np.float32) / 255.0
        prediction = np.asarray(prediction_pil, dtype=np.float32) / 255.0
        metrics = image_metrics(prediction, target)
        metrics["edge_ab_mae"] = edge_ab_mae(prediction, target)
        if lpips_model is not None:
            pred_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(engine.device)
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(engine.device)
            with torch.inference_mode():
                metrics["lpips"] = float(lpips_model(pred_tensor * 2 - 1,
                                                     target_tensor * 2 - 1).item())
        failure_score = metrics["delta_e2000"] + 0.35 * metrics["edge_ab_mae"]
        if "lpips" in metrics:
            failure_score += 10.0 * metrics["lpips"]
        results.append({
            "local_path": row["local_path"],
            "title": row["title"],
            "page_url": row.get("page_url", ""),
            "content_category": row.get("content_category") or infer_content_category(row),
            "collection_id": row.get("collection_id") or infer_collection_id(row),
            "failure_score": failure_score,
            **metrics,
        })
    results.sort(key=lambda row: row["failure_score"], reverse=True)
    by_category = summarize(results, "content_category")
    by_collection = summarize(results, "collection_id")
    write_csv(args.output / "per_image.csv", results)
    write_csv(args.output / "by_category.csv", by_category)
    write_csv(args.output / "by_collection.csv", by_collection)
    render_failure_sheets(engine, args.dataset, results, args.output, args.max_side, args.worst_count)
    metric_keys = [key for key in results[0] if key in {
        "psnr", "ssim", "delta_e2000", "ab_mae", "edge_ab_mae", "pred_chroma",
        "target_chroma", "lpips"
    }]
    report = {
        "role": "V3 development failure analysis; not an untouched final test",
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "checkpoint_sha256": engine.checkpoint_sha256,
        "dataset": str(args.dataset.resolve()),
        "accepted_journal_sha256": sha256(accepted_path),
        "manual_review": str(review_path.resolve()),
        "manual_review_sha256": sha256(review_path),
        "count": len(results),
        "partial": bool(args.limit),
        "mean": {key: float(np.mean([row[key] for row in results])) for key in metric_keys},
        "content_counts": dict(Counter(row["content_category"] for row in results)),
        "collection_count": len({row["collection_id"] for row in results}),
        "ranking": "delta_e2000 + 0.35*edge_ab_mae + 10*lpips (when LPIPS enabled)",
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
