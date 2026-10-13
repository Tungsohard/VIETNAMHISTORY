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
from .model import build_model
from .runtime import load_checkpoint
from .temporal import TemporalABStabilizer


def _normalized_luminance(rgb, image_size):
    rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    luminance = lab[..., 0]
    resized = cv2.resize(luminance, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized / 50.0 - 1.0).unsqueeze(0)
    return luminance, tensor


def enhance_historical_vibrancy(luminance, scaled_ab, boost=0.35):
    """Bộ hiệu chỉnh sắc độ & cân bằng quang quyển cho ảnh tư liệu lịch sử Việt Nam.
    Đã được tinh chỉnh bảo vệ với các cơ chế:
    1. Ceiling Guard: Giới hạn trần sáng (L <= 82) loại trừ hoàn toàn tóc bạc, áo trắng, đèn flash khỏi vùng trời.
    2. Cool Chroma Prior: Chỉ kích hoạt vùng trời khi màu dự đoán của mô hình đã là tông lạnh hoặc trung tính (b <= 1.5, a <= 0.5); tuyệt đối không nhuộm xanh tường trong nhà, vách gỗ, da mặt.
    3. Person & Hair Shield: Tự động phát hiện khuôn mặt/da người và mở rộng bán kính bảo vệ đầu, tóc, râu, cổ áo.
    4. Top Border Connectivity: Kiểm tra độ mở thông thoáng của vòm trời ở mép trên bức ảnh, tự động triệt tiêu bầu trời nếu là ảnh nội cảnh kín (indoor).
    5. Soft Adaptive Blending: Pha trộn sắc lam tự nhiên có chặn biên, chống viền lem (fringe bleeding).
    """
    if boost <= 0.0:
        return scaled_ab
    h, w = luminance.shape
    a = scaled_ab[..., 0].copy()
    b = scaled_ab[..., 1].copy()

    gy, gx = np.gradient(luminance)
    grad = np.sqrt(gx**2 + gy**2)

    # A. Tấm khiên bảo vệ người & tóc (Person & Hair Shield)
    skin_core = (a >= -1.0) & (a <= 16.0) & (b >= 2.5) & (b <= 24.0) & (luminance >= 24.0) & (luminance <= 82.0)
    if np.any(skin_core):
        k_h = max(11, int(h * 0.05))
        k_w = max(11, int(w * 0.04))
        kernel_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
        person_shield = cv2.dilate(skin_core.astype(np.uint8), kernel_base, iterations=1) > 0

        # Mở rộng bảo vệ lên phía trên đỉnh đầu (bảo vệ tóc, nón, mũ cối, râu)
        kernel_up = np.zeros((k_h * 2 + 1, k_w), dtype=np.uint8)
        kernel_up[:k_h + 1, :] = 1
        hair_shield = cv2.dilate(skin_core.astype(np.uint8), kernel_up, iterations=1) > 0
        person_shield = person_shield | hair_shield
    else:
        person_shield = np.zeros((h, w), dtype=bool)

    # B. Nhận diện bầu trời ngoài trời có kiểm soát nghiêm ngặt
    # 1. Trần độ sáng: Vùng trời xanh trong ảnh tư liệu chỉ có L trong dải [48, 82].
    # Vùng L > 84 là cháy sáng, tóc bạc hoặc áo trắng -> dập tắt hoàn toàn bright_prior.
    lower_ramp = np.clip((luminance - 48.0) / 18.0, 0.0, 1.0)
    upper_cutoff = np.clip(1.0 - (luminance - 82.0) / 5.0, 0.0, 1.0)
    bright_prior = lower_ramp * upper_cutoff

    # 2. Vị trí không gian: 40% phần trên bức ảnh
    y_coords = np.linspace(0, 1, h)[:, None]
    upper_prior = np.clip(1.0 - (y_coords / 0.40), 0.0, 1.0) ** 1.3

    # 3. Độ dốc biên: Bầu trời phải phẳng mịn
    smooth_prior = np.clip(1.0 - (grad / 10.0), 0.0, 1.0)

    # 4. Tiền đề sắc độ (Chroma Prior): Bầu trời TUYỆT ĐỐI không thể là màu ấm (tường vàng, gỗ, giấy ố)
    cool_prior = np.clip(1.0 - (b - 0.5) / 2.0, 0.0, 1.0) * np.clip(1.0 - (a - (-1.0)) / 1.5, 0.0, 1.0)

    # 5. Kiểm tra kết nối mép trên cùng (Top Border Connectivity / Indoor Detector)
    top_slice_h = max(2, int(h * 0.05))
    top_sky_candidate = (luminance[:top_slice_h, :] >= 48.0) & (luminance[:top_slice_h, :] <= 85.0) & (b[:top_slice_h, :] <= 2.0) & (grad[:top_slice_h, :] <= 12.0)
    top_sky_ratio = float(np.mean(top_sky_candidate))

    if top_sky_ratio < 0.05:
        # Nếu mép trên cùng không có mảng trời thông thoáng (như ảnh trong nhà, góc hẹp), triệt tiêu hoàn toàn sky boost
        sky_mask = np.zeros_like(luminance)
    else:
        sky_connectivity = np.clip((top_sky_ratio - 0.04) / 0.12, 0.0, 1.0)
        sky_raw = upper_prior * bright_prior * smooth_prior * cool_prior * sky_connectivity
        sky_mask = np.where(person_shield, 0.0, sky_raw)
        sky_mask = np.clip(sky_mask * 1.5, 0.0, 1.0)

    # Hòa trộn sắc xanh da trời tự nhiên (Cerulean Blue) có chặn biên mềm
    sky_blend = np.clip(sky_mask * (boost * 1.6), 0.0, 0.60)
    b = b * (1.0 - sky_blend) + (-15.0) * sky_blend
    a = a * (1.0 - sky_blend) + (-3.0) * sky_blend

    # C. Bảo vệ và hồi phục sắc da Á Đông tự nhiên (Skin Tone Warmth)
    skin_shadow_mask = (a >= -2.0) & (a <= 8.0) & (b >= 2.5) & (b <= 22.0) & (luminance >= 26.0) & (luminance <= 80.0) & (sky_mask < 0.02)
    a = np.where(skin_shadow_mask, a + (3.2 * boost), a)
    b = np.where(skin_shadow_mask, b + (1.6 * boost), b)

    # D. Tăng cường sắc lục thảm thực vật nhiệt đới (CHỈ tác động lên cây cối thực sự a < -3.5, b > 4.0)
    foliage_mask = (a < -3.5) & (b > 4.0) & (sky_mask < 0.02) & (luminance < 78.0) & (~person_shield)
    a = np.where(foliage_mask, a * (1.0 + 0.45 * boost) - (2.6 * boost), a)
    b = np.where(foliage_mask, b * (1.0 + 0.35 * boost) + (1.8 * boost), b)

    # E. Tôn vinh quân phục ô-liu / khaki (quân phục Tô Châu)
    uniform_mask = (a >= -4.0) & (a <= 2.0) & (b >= 8.0) & (b <= 25.0) & (luminance >= 22.0) & (luminance <= 65.0) & (sky_mask < 0.02) & (~person_shield)
    b = np.where(uniform_mask, b * (1.0 + 0.22 * boost), b)

    # F. Tăng cường sắc đỏ quốc kỳ / huy hiệu / huân chương (a > 5.0)
    red_mask = (a > 5.0) & (sky_mask < 0.02)
    a = np.where(red_mask, a * (1.0 + 0.30 * boost) + (1.2 * boost), a)

    return np.stack([a, b], axis=-1)


def _lab_prediction_to_rgb(luminance, normalized_ab, strength=1.0, vibrancy_boost=0.0):
    height, width = luminance.shape
    ab = normalized_ab.detach().float().cpu().permute(1, 2, 0).numpy()
    ab = cv2.resize(ab, (width, height), interpolation=cv2.INTER_CUBIC)
    scaled_ab = ab * (128.0 * strength)
    if vibrancy_boost > 0.0:
        scaled_ab = enhance_historical_vibrancy(luminance, scaled_ab, boost=vibrancy_boost)
    lab = np.concatenate((luminance[..., None], scaled_ab), axis=2).astype(np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def _lab_array_to_rgb(luminance, normalized_ab, strength=1.0, vibrancy_boost=0.0):
    height, width = luminance.shape
    ab = cv2.resize(np.asarray(normalized_ab, np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    scaled_ab = ab * (128.0 * strength)
    if vibrancy_boost > 0.0:
        scaled_ab = enhance_historical_vibrancy(luminance, scaled_ab, boost=vibrancy_boost)
    lab = np.concatenate((luminance[..., None], scaled_ab), axis=2).astype(np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


class ColorizationEngine:
    """Thread-safe DDColor inference while preserving the source luminance resolution."""

    def __init__(self, checkpoint_path, device=None, expected_sha256=None):
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
        self.config = dict(checkpoint["run_config"])
        self.image_size = int(self.config["image_size"])
        build_config = {**self.config, "pretrained_encoder": False}
        self.model = build_model(build_config)
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
    def predict_ab_batch(self, images):
        if not images:
            return [], np.empty((0, 2, self.image_size, self.image_size), dtype=np.float32)
        prepared = [_normalized_luminance(image, self.image_size) for image in images]
        batch = torch.stack([item[1] for item in prepared]).to(self.device)
        with self._lock, torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            prediction = self.model(batch)
        return [item[0] for item in prepared], prediction.detach().float().cpu().numpy()

    def colorize_rgb_batch(self, images, strength=1.0, vibrancy_boost=0.0):
        if not 0.5 <= strength <= 1.5:
            raise ValueError("Color strength must be between 0.5 and 1.5")
        luminances, predictions = self.predict_ab_batch(images)
        return [_lab_prediction_to_rgb(luminance, torch.from_numpy(predictions[index]), strength, vibrancy_boost)
                for index, luminance in enumerate(luminances)]

    def colorize_pil(self, image, strength=1.0, vibrancy_boost=0.0):
        if image is None:
            raise ValueError("No image was provided")
        image = ImageOps.exif_transpose(image).convert("RGB")
        result = self.colorize_rgb_batch([np.asarray(image)], strength, vibrancy_boost)[0]
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
