"""Production-oriented image and video inference for a trained colorizer."""
import json
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .data import sha256
from .enhance import MODES, modernize_ab, refine_ab, restore_luminance
from .model import build_model
from .runtime import load_checkpoint, load_config
from .temporal import TemporalABStabilizer


OFFICIAL_DDCOLOR_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ddcolor_tiny.json"


def _normalized_luminance(rgb, image_size, restore=0.0):
    rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    luminance = restore_luminance(lab[..., 0], restore)
    resized = cv2.resize(luminance, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized / 50.0 - 1.0).unsqueeze(0)
    return luminance, tensor


def _compose_rgb(luminance, ab):
    lab = np.concatenate((luminance[..., None], ab), axis=2).astype(np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def _lab_prediction_to_rgb(luminance, normalized_ab, strength=1.0, mode=None, objects=None):
    """Combine full-resolution L with the network ab; ``mode=None`` is the raw model output.

    ``objects`` is an optional object-id map (see ``segmentation.ObjectSegmenter``) that
    stops colour from bleeding across object borders.
    """
    height, width = luminance.shape
    ab = normalized_ab.detach().float().cpu().permute(1, 2, 0).numpy() * (128.0 * strength)
    if mode is None:
        return _compose_rgb(luminance, cv2.resize(ab, (width, height), interpolation=cv2.INTER_CUBIC))
    _, region_weight, vividness = MODES[mode]
    ab = refine_ab(luminance, ab, region_weight, objects=objects)
    ab = modernize_ab(luminance, ab, vividness)
    return _compose_rgb(luminance, ab)


def _lab_array_to_rgb(luminance, normalized_ab, strength=1.0):
    height, width = luminance.shape
    ab = cv2.resize(np.asarray(normalized_ab, np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return _compose_rgb(luminance, ab * (128.0 * strength))


class ColorizationEngine:
    """Thread-safe DDColor inference while preserving the source luminance resolution."""

    def __init__(self, checkpoint_path, device=None, expected_sha256=None, segmenter=None):
        self.segmenter = segmenter
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        if expected_sha256 and self.checkpoint_sha256.lower() != expected_sha256.lower():
            raise ValueError("Checkpoint fingerprint does not match the registered model")

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = torch.device(requested_device)

        checkpoint = load_checkpoint(self.checkpoint_path)
        official = "run_config" not in checkpoint          # weights published by the DDColor authors
        self.config = (load_config(OFFICIAL_DDCOLOR_CONFIG) if official else dict(checkpoint["run_config"]))
        self.image_size = int(self.config["image_size"])
        build_config = {**self.config, "pretrained_encoder": False}
        self.model = build_model(build_config)
        if official:
            self.model.load_official(self.checkpoint_path)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device).eval()
        self._lock = threading.Lock()
        del checkpoint

    @classmethod
    def from_selection(cls, selection_path, device=None):
        selection_path = Path(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        winner = selection["winner"]
        checkpoint_path = Path(winner["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = selection_path.parent / checkpoint_path
        return cls(checkpoint_path, device=device, expected_sha256=winner["sha256"])

    @property
    def device_name(self):
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"

    @torch.inference_mode()
    def predict_ab_batch(self, images, restore=0.0):
        if not images:
            return [], np.empty((0, 2, self.image_size, self.image_size), dtype=np.float32)
        prepared = [_normalized_luminance(image, self.image_size, restore) for image in images]
        batch = torch.stack([item[1] for item in prepared]).to(self.device)
        with self._lock, torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            prediction = self.model(batch)
        return [item[0] for item in prepared], prediction.detach().float().cpu().numpy()

    def colorize_rgb_batch(self, images, strength=1.0, mode=None):
        """``mode`` is one of ``enhance.MODES``; ``None`` returns the raw network colours."""
        if not 0.5 <= strength <= 1.5:
            raise ValueError("Color strength must be between 0.5 and 1.5")
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown colorization mode: {mode}")
        restore = MODES[mode][0] if mode else 0.0
        luminances, predictions = self.predict_ab_batch(images, restore)
        objects = [None] * len(images)
        if mode is not None and self.segmenter is not None:
            objects = [self.segmenter.labels(np.asarray(image)) for image in images]
        return [_lab_prediction_to_rgb(luminance, torch.from_numpy(predictions[index]), strength, mode,
                                       objects[index])
                for index, luminance in enumerate(luminances)]

    def colorize_pil(self, image, strength=1.0, mode=None):
        if image is None:
            raise ValueError("No image was provided")
        image = ImageOps.exif_transpose(image).convert("RGB")
        result = self.colorize_rgb_batch([np.asarray(image)], strength, mode)[0]
        return Image.fromarray(result)

    def colorize_video(self, input_path, output_path, strength=1.0, batch_size=2,
                       max_minutes=10.0, temporal_strength=0.55, progress_callback=None):
        input_path, output_path = Path(input_path), Path(output_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Cannot open the uploaded video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Video has an invalid frame size")
        if total > 0 and total / fps > max_minutes * 60:
            capture.release()
            raise ValueError(f"Video is longer than the {max_minutes:g}-minute demo limit")

        silent_path = output_path.with_name(f"{output_path.stem}_silent.mp4")
        writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Cannot create the output video")
        processed = 0
        frames = []
        stabilizer = TemporalABStabilizer(temporal_strength)

        def write_batch(batch_frames):
            luminances, predictions = self.predict_ab_batch(batch_frames)
            for index, luminance in enumerate(luminances):
                model_luminance = cv2.resize(luminance, (self.image_size, self.image_size),
                                             interpolation=cv2.INTER_AREA)
                model_ab = predictions[index].transpose(1, 2, 0)
                stable_ab = stabilizer.apply(model_luminance, model_ab)
                result = _lab_array_to_rgb(luminance, stable_ab, strength)
                writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) < batch_size:
                    continue
                write_batch(frames)
                processed += len(frames)
                frames.clear()
                if progress_callback:
                    progress_callback(processed, total)
            if frames:
                write_batch(frames)
                processed += len(frames)
                if progress_callback:
                    progress_callback(processed, total)
        finally:
            capture.release()
            writer.release()
        if processed == 0:
            silent_path.unlink(missing_ok=True)
            raise ValueError("The uploaded video contains no readable frames")

        self._mux_original_audio(silent_path, input_path, output_path)
        return output_path

    @staticmethod
    def _mux_original_audio(silent_path, input_path, output_path):
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
            command = [executable, "-y", "-loglevel", "error", "-i", str(silent_path),
                       "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?",
                       "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path)]
            subprocess.run(command, check=True, capture_output=True)
            silent_path.unlink(missing_ok=True)
        except (ImportError, OSError, subprocess.CalledProcessError):
            shutil.move(str(silent_path), str(output_path))
