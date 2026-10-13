"""Measure flow-aligned chroma flicker before and after temporal stabilization."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from colorization.inference import ColorizationEngine, _normalized_luminance
from colorization.runtime import write_json
from colorization.temporal import TemporalABStabilizer, warp_error


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "reports" / "vietnam_v3_selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temporal-strength", type=float, default=0.55)
    parser.add_argument("--scene-cut-threshold", type=float, default=32.0)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.frames < 2 or args.batch_size < 1:
        parser.error("--frames must be at least 2 and --batch-size must be positive")

    engine = ColorizationEngine.from_selection(args.selection, args.device)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    stabilizer = TemporalABStabilizer(args.temporal_strength, args.scene_cut_threshold)
    raw_errors, stable_errors = [], []
    previous_luminance = previous_raw = previous_stable = None
    decoded = scene_cuts = 0

    def evaluate_batch(frames):
        nonlocal previous_luminance, previous_raw, previous_stable, scene_cuts
        luminances, predictions = engine.predict_ab_batch(frames)
        for index, luminance in enumerate(luminances):
            model_l = cv2.resize(luminance, (engine.image_size, engine.image_size),
                                 interpolation=cv2.INTER_AREA)
            raw_ab = predictions[index].transpose(1, 2, 0)
            stable_ab = stabilizer.apply(model_l, raw_ab)
            if previous_luminance is not None:
                frame_delta = np.mean(np.abs(model_l * 2.55 - previous_luminance * 2.55))
                if frame_delta >= args.scene_cut_threshold:
                    scene_cuts += 1
                else:
                    raw_errors.append(warp_error(previous_luminance, model_l, previous_raw, raw_ab))
                    stable_errors.append(warp_error(previous_luminance, model_l,
                                                    previous_stable, stable_ab))
            previous_luminance, previous_raw, previous_stable = model_l, raw_ab, stable_ab

    pending = []
    try:
        while decoded < args.frames:
            ok, bgr = capture.read()
            if not ok:
                break
            pending.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            decoded += 1
            if len(pending) == args.batch_size:
                evaluate_batch(pending)
                pending.clear()
        if pending:
            evaluate_batch(pending)
    finally:
        capture.release()

    if not raw_errors:
        raise ValueError("Not enough non-cut frame pairs to measure temporal stability")
    raw_mean = float(np.mean(raw_errors))
    stable_mean = float(np.mean(stable_errors))
    report = {
        "video": str(args.video.resolve()),
        "checkpoint_sha256": engine.checkpoint_sha256,
        "device": engine.device_name,
        "decoded_frames": decoded,
        "evaluated_frame_pairs": len(raw_errors),
        "scene_cuts_skipped": scene_cuts,
        "fps": fps,
        "temporal_strength": args.temporal_strength,
        "metric": "optical-flow-aligned normalized-ab mean absolute error; lower is better",
        "raw_temporal_ab_mae": raw_mean,
        "stabilized_temporal_ab_mae": stable_mean,
        "relative_reduction_percent": (raw_mean - stable_mean) / raw_mean * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
