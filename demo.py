"""Sắc Việt: Ứng dụng phục dựng và tô màu ảnh tư liệu lịch sử Việt Nam.

Được phát triển trong khuôn khổ Khóa luận tốt nghiệp với kiến trúc đề xuất VietColorNet
kết hợp đối chuẩn cùng mô hình DDColor gốc.
"""
import argparse
import json
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from colorization.inference import ColorizationEngine
from colorization.segmentation import ObjectSegmenter


ROOT = Path(__file__).resolve().parent
MAX_IMAGE_PIXELS = 40_000_000


def ensure_pil_image(image):
    """Chuyển đổi an toàn mọi định dạng đầu vào (PIL, path, numpy, dict) sang ảnh RGB PIL."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        p = Path(image)
        if p.is_file():
            return Image.open(p).convert("RGB")
        return None
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, dict):
        p = image.get("path") or image.get("url")
        if p and Path(p).is_file():
            return Image.open(p).convert("RGB")
    try:
        return Image.fromarray(np.asarray(image)).convert("RGB")
    except Exception:
        return None

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&display=swap');

:root {
  --paper-bg: #faf9f5;
  --paper-card: rgba(255, 255, 255, 0.95);
  --ink-900: #121816;
  --ink-700: #33413d;
  --ink-500: #5c6e69;
  --line: #e3ebe8;
  --line-strong: #cdd9d4;
  --son-red: #b91c1c;
  --son-red-light: #fef2f2;
  --son-red-border: #fecaca;
  --gold-amber: #d97706;
  --gold-light: #fffbeb;
  --ngoc-green: #047857;
  --ngoc-green-dark: #065f46;
  --ngoc-green-light: #ecfdf5;
  --shadow-sm: 0 2px 8px -1px rgba(18, 24, 22, 0.05);
  --shadow-md: 0 10px 25px -4px rgba(18, 24, 22, 0.07), 0 4px 10px -2px rgba(18, 24, 22, 0.03);
}

.gradio-container {
  max-width: 1540px !important;
  margin: 0 auto !important;
  background-color: var(--paper-bg) !important;
  background-image: radial-gradient(#e5e2d6 0.75px, transparent 0.75px) !important;
  background-size: 24px 24px !important;
  color: var(--ink-900) !important;
  font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.app-shell {
  padding: 16px 20px 48px !important;
}

/* Header Banner - Phong cách Hoàng gia & Hiện đại */
.brand-hero {
  position: relative;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.97) 0%, rgba(252, 249, 242, 0.92) 100%);
  border: 1px solid var(--line-strong);
  border-top: 4px solid;
  border-image: linear-gradient(90deg, var(--son-red) 0%, var(--gold-amber) 50%, var(--ngoc-green) 100%) 1;
  border-radius: 12px;
  padding: 22px 28px 18px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-md);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
}

.brand-title-wrap {
  display: flex;
  flex-direction: column;
}

.national-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--son-red);
  background: var(--son-red-light);
  border: 1px solid var(--son-red-border);
  padding: 3px 12px;
  border-radius: 9999px;
  width: fit-content;
  margin-bottom: 8px;
  letter-spacing: 0.6px;
  text-transform: uppercase;
}

.brand-main-title {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 38px !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
  color: var(--ink-900) !important;
  letter-spacing: -0.5px;
  margin: 0;
}

.brand-main-title span {
  color: var(--son-red);
}

.brand-sub-title {
  margin-top: 6px;
  font-size: 14.5px;
  color: var(--ink-500);
  font-weight: 400;
}

.system-status-box {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
}

.status-chip {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #ffffff;
  border: 1px solid var(--line);
  padding: 6px 14px;
  border-radius: 8px;
  box-shadow: var(--shadow-sm);
  font-size: 13px;
  font-weight: 500;
  color: var(--ink-700);
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--ngoc-green);
  box-shadow: 0 0 0 2px rgba(4, 120, 87, 0.2);
  animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(4, 120, 87, 0.5); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(4, 120, 87, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(4, 120, 87, 0); }
}

.card-container {
  background: var(--paper-card) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow-sm) !important;
  overflow: hidden !important;
}

.control-panel-bar {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 18px;
  margin-top: 14px;
  box-shadow: var(--shadow-sm);
}

#primary-run-button {
  background: linear-gradient(135deg, var(--ngoc-green) 0%, var(--ngoc-green-dark) 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 15.5px !important;
  border: none !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 14px rgba(4, 120, 87, 0.28) !important;
  transition: all 0.2s ease-in-out !important;
}

#primary-run-button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(4, 120, 87, 0.38) !important;
}

#compare-run-button {
  background: linear-gradient(135deg, var(--son-red) 0%, #991b1b 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 15.5px !important;
  border: none !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 14px rgba(185, 28, 28, 0.28) !important;
  transition: all 0.2s ease-in-out !important;
}

#compare-run-button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(185, 28, 28, 0.38) !important;
}

.methodology-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 24px;
  box-shadow: var(--shadow-sm);
}

.metric-table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 14px;
}

.metric-table th {
  background: var(--paper-bg);
  color: var(--ink-900);
  font-weight: 600;
  text-align: left;
  padding: 10px 14px;
  border-bottom: 2px solid var(--line-strong);
}

.metric-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  color: var(--ink-700);
}

.metric-table tr.highlight-row {
  background: var(--ngoc-green-light);
  font-weight: 600;
}

.pill-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
}

.pill-tag.proposed {
  background: #dcfce7;
  color: #15803d;
}

.pill-tag.baseline {
  background: #f1f5f9;
  color: #475569;
}

.footer-note {
  color: var(--ink-500);
  font-size: 12.5px;
  margin-top: 24px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  text-align: center;
}

@media (max-width: 840px) {
  .brand-hero { flex-direction: column; align-items: flex-start; }
  .system-status-box { align-items: flex-start; }
  .brand-main-title { font-size: 30px !important; }
}
"""


def safe_stem(path):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")
    return stem[:60] or "image"


MODE_CHOICES = [
    ("🌟 Hiện đại (phục hồi ảnh + màu tươi tự nhiên, không lem)", "modern"),
    ("🔥 Rực rỡ (như Hiện đại, màu đậm hơn)", "vivid"),
    ("🌤️ Nguyên bản (màu của mô hình, chỉ chống lem biên)", "natural"),
]


def load_engines(args):
    """Nạp các mô hình khả dụng để phục vụ chế độ đơn và chế độ đối so sánh."""
    engines = {}

    # 1. Nạp DDColor gốc (baseline thuần, không tinh chỉnh)
    try:
        engine_dd = ColorizationEngine.from_selection(args.selection, device=args.device)
        engines["ddcolor"] = {
            "engine": engine_dd,
            "label": "DDColor gốc (Baseline thuần)",
            "desc": "Trọng số DDColor chính thức huấn luyện trên ImageNet, không tinh chỉnh trên dữ liệu Việt Nam",
        }
    except Exception as exc:
        print(f"Warning: Could not load DDColor baseline: {exc}")

    # 2. Hai mô hình VietColorNet được giữ lại, trong checkpoints/models/:
    #  - historical (mặc định): train thêm trên phim tư liệu ViCoW; ít ám vàng nâu, chống lem tốt hơn.
    #  - vivid: không dùng phim tư liệu; màu tươi hơn nhưng ám vàng nâu nhiều hơn.
    model_dir = ROOT / "checkpoints" / "models"
    historical = model_dir / "vietcolornet_historical.pth"
    vivid = model_dir / "vietcolornet_vivid.pth"
    primary = Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).is_file() else historical

    if primary.is_file():
        try:
            engines["vietcolornet"] = {
                "engine": ColorizationEngine(primary, device=args.device),
                "label": "VietColorNet · Lịch sử (khuyên dùng)",
                "desc": "Huấn luyện trên ảnh màu hiện đại Việt Nam, ảnh quân đội và phim tư liệu; phục hồi ảnh và chống lem màu bằng SAM",
            }
        except Exception as exc:
            print(f"Warning: Could not load VietColorNet {primary}: {exc}")

    if vivid.is_file() and vivid.resolve() != primary.resolve():
        try:
            engines["vietcolornet_vivid"] = {
                "engine": ColorizationEngine(vivid, device=args.device),
                "label": "VietColorNet · Tươi (màu đậm hơn, dễ ám vàng)",
                "desc": "Không dùng phim tư liệu; màu tươi hơn nhưng ám vàng nâu nhiều hơn",
            }
        except Exception as exc:
            print(f"Warning: Could not load VietColorNet vivid: {exc}")

    # Dự phòng nếu cả hai đều chưa có checkpoint chỉ định
    if not engines and args.checkpoint:
        engines["custom"] = {
            "engine": ColorizationEngine(args.checkpoint, device=args.device),
            "label": "Mô hình tùy chọn",
            "desc": str(args.checkpoint),
        }

    # SAM chỉ cung cấp ranh giới vật thể để màu không lem sang nhau; một bản dùng chung
    # cho các VietColorNet. Thiếu gói/trọng số thì vẫn chạy với chống lem theo superpixel.
    if not getattr(args, "no_sam", False):
        segmenter = ObjectSegmenter.try_create(device=args.device)
        if segmenter is None:
            print("Warning: SAM không khả dụng (thiếu segment-anything hoặc checkpoints/pretrained/sam); "
                  "dùng chống lem theo superpixel.")
        for key, item in engines.items():
            if key.startswith("vietcolornet") or key == "custom":
                item["engine"].segmenter = segmenter

    return engines


def create_demo(engines, output_dir, max_video_minutes=10.0, enable_video=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_keys = list(engines.keys())
    default_key = "vietcolornet" if "vietcolornet" in engines else model_keys[0]

    # Danh sách ảnh mẫu tư liệu
    example_images = []
    example_dir = ROOT / "examples"
    if example_dir.exists():
        for path in sorted(example_dir.glob("*.jpg")):
            example_images.append([str(path)])

    def colorize_single(image, strength, model_key, vibrancy_mode="modern", progress=gr.Progress()):
        img = ensure_pil_image(image)
        if img is None:
            return None, None, None, "⚠️ Vui lòng chọn hoặc tải lên một ảnh tư liệu trước khi bắt đầu phục chế màu."
        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            return None, None, None, "⚠️ Ảnh vượt quá giới hạn 40 megapixel. Vui lòng giảm độ phân giải."

        progress(0.1, desc="Đang chuẩn bị ảnh & không gian màu")
        started = time.perf_counter()

        # Chuẩn bị ảnh xám đối chiếu
        gray_image = ImageOps.grayscale(img).convert("RGB")

        engine_item = engines.get(model_key) or engines[default_key]
        active_engine = engine_item["engine"]
        progress(0.4, desc=f"Đang suy luận màu với {engine_item['label']}")

        result = active_engine.colorize_pil(img, strength, mode=vibrancy_mode)
        progress(0.85, desc="Đang hoàn tất và xuất kết quả")

        output_path = output_dir / f"sac_viet_{uuid.uuid4().hex[:10]}.png"
        result.save(output_path, format="PNG", optimize=True)
        gray_path = output_dir / f"sac_viet_gray_{uuid.uuid4().hex[:10]}.png"
        gray_image.save(gray_path, format="PNG", optimize=True)
        elapsed = time.perf_counter() - started

        status = (
            f"✨ Hoàn tất trong **{elapsed:.2f} giây** · Kích thước: **{width:,} × {height:,} px** · "
            f"Mô hình: **{engine_item['label']}** · Cường độ: **{strength:.2f}x** · Chế độ: **{vibrancy_mode}**"
        )
        # Trả về đường dẫn file cho ImageSlider và ảnh PIL cho Xem trực tiếp
        return (str(gray_path), str(output_path)), result, str(output_path), status

    def compare_models(image, fallback_image, strength, vibrancy_mode="modern", progress=gr.Progress()):
        img = ensure_pil_image(image) or ensure_pil_image(fallback_image)
        if img is None:
            return None, None, None, "⚠️ Vui lòng tải lên ảnh hoặc chọn ảnh mẫu bên dưới để đối so sánh."

        progress(0.1, desc="Chuẩn bị ảnh xám")
        gray_image = ImageOps.grayscale(img).convert("RGB")

        dd_result = None
        vnet_result = None
        info_lines = []

        if "ddcolor" in engines:
            progress(0.3, desc="Đang chạy DDColor gốc")
            t0 = time.perf_counter()
            dd_result = engines["ddcolor"]["engine"].colorize_pil(img, strength)
            t_dd = time.perf_counter() - t0
            info_lines.append(f"⏱️ **DDColor gốc**: {t_dd:.2f}s")
        else:
            dd_result = gray_image
            info_lines.append("⚠️ DDColor gốc: Chưa nạp được checkpoint")

        if "vietcolornet" in engines:
            progress(0.65, desc="Đang chạy VietColorNet")
            t0 = time.perf_counter()
            vnet_result = engines["vietcolornet"]["engine"].colorize_pil(img, strength, mode=vibrancy_mode)
            t_vnet = time.perf_counter() - t0
            info_lines.append(f"⚡ **VietColorNet (Đề xuất)**: {t_vnet:.2f}s")
        else:
            vnet_result = gray_image
            info_lines.append("⚠️ VietColorNet: Chưa nạp được checkpoint")

        progress(0.95, desc="Hoàn tất so sánh")
        compare_status = " · ".join(info_lines)
        return gray_image, dd_result, vnet_result, compare_status


    def colorize_video(video_path, strength, temporal_strength, progress=gr.Progress()):
        if not video_path:
            raise gr.Error("Chưa chọn file video")
        output = output_dir / f"{safe_stem(video_path)}_{uuid.uuid4().hex[:8]}_colorized.mp4"

        engine_item = engines.get(default_key)
        active_engine = engine_item["engine"]

        def update(done, total):
            if total > 0:
                progress(min(done / total, 1.0), desc=f"{done}/{total} frames")
            else:
                progress(0, desc=f"{done} frames")

        try:
            return str(active_engine.colorize_video(video_path, output, strength=strength,
                                                   batch_size=2, max_minutes=max_video_minutes,
                                                   temporal_strength=temporal_strength,
                                                   progress_callback=update))
        except Exception as exc:
            output.unlink(missing_ok=True)
            raise gr.Error(str(exc)) from exc

    # Xác định thông tin thiết bị hiển thị
    primary_engine = engines[default_key]["engine"]
    device_label = primary_engine.device_name
    active_model_name = engines[default_key]["label"]

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=["Be Vietnam Pro", "Arial", "sans-serif"],
    )

    with gr.Blocks(title="Sắc Việt · Phục Chế Ảnh Tư Liệu", theme=theme, css=CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            # Top Banner Hero
            gr.HTML(f"""
            <div class="brand-hero">
              <div class="brand-title-wrap">
                <div class="national-badge">🇻🇳 Đề tài Khóa luận Tốt nghiệp · AI phục dựng lịch sử</div>
                <h1 class="brand-main-title">SẮC <span>VIỆT</span></h1>
                <div class="brand-sub-title">Hệ thống Phục chế và Tô màu Ảnh tư liệu Lịch sử Việt Nam bằng Mạng nơ-ron sâu</div>
              </div>
              <div class="system-status-box">
                <div class="status-chip">
                  <span class="status-indicator"></span>
                  <span>{active_model_name}</span>
                </div>
                <div style="font-size: 12px; color: #718096; margin-top: 2px;">
                  Phần cứng: <strong>{device_label}</strong> · SHA: <code>{primary_engine.checkpoint_sha256[:10]}</code>
                </div>
              </div>
            </div>
            """)

            with gr.Tabs():
                # TAB 1: Phục chế ảnh đơn
                with gr.Tab("🎨 Phục Chế Ảnh Tư Liệu"):
                    with gr.Row(equal_height=True):
                        # Khung tải ảnh
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="pil", image_mode="RGB", sources=["upload", "clipboard"],
                                label="Ảnh tư liệu gốc (Tải ảnh hoặc dán Ctrl+V)", height=450,
                                elem_classes=["card-container"],
                            )
                        # Khung hiển thị kết quả trực tiếp độ nét cao
                        with gr.Column(scale=1):
                            single_output = gr.Image(
                                type="pil", format="png", label="Ảnh Phục Chế Màu Sắc Nét (VietColorNet)",
                                height=450, interactive=False, elem_classes=["card-container"],
                            )

                    with gr.Accordion("↔️ Kéo Trượt So Sánh Trực Quan (Trước / Sau)", open=True):
                        slider_output = gr.ImageSlider(
                            label="Kéo thanh trượt để so sánh Trước / Sau",
                            type="filepath", show_download_button=True,
                        )

                    # Băng điều khiển tham số
                    with gr.Column(elem_classes=["control-panel-bar"]):
                        with gr.Row(equal_height=True):
                            model_selector = gr.Dropdown(
                                choices=[(v["label"], k) for k, v in engines.items()],
                                value=default_key,
                                label="Lựa chọn Mô hình",
                                scale=2,
                            )
                            image_strength = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="Cường độ màu sắc (Chroma Boost)",
                                info="1.00: Chuẩn xác suất mô hình · >1.10: Tăng độ rực màu cho ảnh quá cũ",
                                scale=2,
                            )
                            vibrancy_mode = gr.Dropdown(
                                choices=MODE_CHOICES,
                                value="modern",
                                label="Chế độ phục chế",
                                info="Phục hồi ảnh mờ/nhiễu trước khi tô, màu bám biên vật thể (không lem), khử ám sepia/olive",
                                scale=2,
                            )
                        with gr.Row():
                            preset_natural = gr.Button("🎨 Tự nhiên (1.00x)", size="sm", variant="secondary")
                            preset_vibrant = gr.Button("🔥 Rực rỡ (1.20x)", size="sm", variant="secondary")
                            preset_vintage = gr.Button("🍂 Hoài niệm (0.85x)", size="sm", variant="secondary")

                        preset_natural.click(lambda: 1.0, outputs=image_strength)
                        preset_vibrant.click(lambda: 1.2, outputs=image_strength)
                        preset_vintage.click(lambda: 0.85, outputs=image_strength)

                    status_msg = gr.Markdown("🟢 Hệ thống sẵn sàng phục chế.", elem_classes=["status-line"])

                    with gr.Row():
                        run_btn = gr.Button("🚀 Bắt đầu Phục Chế Màu", variant="primary", elem_id="primary-run-button", scale=3)
                        dl_btn = gr.DownloadButton("💾 Tải ảnh PNG chất lượng cao", variant="secondary", scale=1)
                        clear_btn = gr.ClearButton(
                            [image_input, slider_output, single_output, dl_btn, status_msg],
                            value="🔄 Làm mới", scale=1,
                        )

                    # Sự kiện click tô màu ảnh
                    run_btn.click(
                        colorize_single,
                        inputs=[image_input, image_strength, model_selector, vibrancy_mode],
                        outputs=[slider_output, single_output, dl_btn, status_msg],
                        show_progress="full",
                    )
                    # Thư viện ảnh mẫu Việt Nam
                    if example_images:
                        gr.Markdown("### 📚 Ảnh tư liệu mẫu Việt Nam (Bấm để thử nghiệm ngay):")
                        gr.Examples(
                            examples=example_images,
                            inputs=image_input,
                            label="Kho tư liệu lịch sử mẫu",
                            examples_per_page=4,
                        )

                # TAB 2: Đối so sánh mô hình A/B Benchmark
                with gr.Tab("⚖️ Đối So Sánh Mô Hình (A/B Benchmark)"):
                    gr.Markdown("""
                    **Chế độ Đối So Sánh Độc Lập:** Cho phép so sánh trực quan cùng lúc giữa **Ảnh xám gốc**,
                    **DDColor gốc** (baseline thuần) và **VietColorNet** (mô hình đề xuất với Edge-Guidance, CPCA và Decoupled Chroma).
                    """)
                    with gr.Row():
                        compare_input = gr.Image(
                            type="pil", image_mode="RGB", sources=["upload", "clipboard"],
                            label="Ảnh xám thử nghiệm", height=320,
                        )
                    with gr.Row(equal_height=True):
                        compare_strength = gr.Slider(
                            0.5, 1.5, value=1.0, step=0.05, label="Cường độ màu sắc (Chroma Boost)",
                            scale=2,
                        )
                        compare_vibrancy = gr.Dropdown(
                            choices=MODE_CHOICES,
                            value="modern",
                            label="Chế độ phục chế (VietColorNet)",
                            info="Phục hồi ảnh mờ/nhiễu trước khi tô, màu bám biên vật thể (không lem), khử ám sepia/olive",
                            scale=2,
                        )
                    compare_btn = gr.Button("⚡ Chạy Đối So Sánh Cả 2 Mô Hình", variant="primary", elem_id="compare-run-button")
                    compare_status_msg = gr.Markdown("")

                    with gr.Row(equal_height=True):
                        comp_gray = gr.Image(type="pil", label="1. Ảnh xám đầu vào", interactive=False)
                        comp_dd = gr.Image(type="pil", label="2. DDColor gốc (Baseline)", interactive=False)
                        comp_vnet = gr.Image(type="pil", label="3. VietColorNet (Mô hình Đề Xuất)", interactive=False)

                    compare_btn.click(
                        compare_models,
                        inputs=[compare_input, image_input, compare_strength, compare_vibrancy],
                        outputs=[comp_gray, comp_dd, comp_vnet, compare_status_msg],
                        show_progress="full",
                    )

                    if example_images:
                        gr.Examples(
                            examples=example_images,
                            inputs=compare_input,
                            label="Chọn ảnh mẫu thử nghiệm đối chuẩn:",
                            examples_per_page=4,
                        )

                # TAB 3: Giới thiệu đề tài & Phương pháp nghiên cứu
                with gr.Tab("📜 Về Đề Tài & Phương Pháp"):
                    with gr.Column(elem_classes=["methodology-card"]):
                        gr.HTML("""
                        <h2 style="font-family: 'Playfair Display', serif; color: #121816; margin-top: 0;">
                          Đề Tài Khóa Luận Tốt Nghiệp: Phục Chế Ảnh Tư Liệu Lịch Sử Việt Nam
                        </h2>
                        <p style="color: #4b5563; font-size: 15px; line-height: 1.6;">
                          Dự án tập trung giải quyết bài toán phục chế và hồi sinh màu sắc cho các tư liệu ảnh lịch sử,
                          chiến tranh và văn hóa Việt Nam thế kỷ 20. Khác với các mô hình chung chung trên thế giới thường
                          gặp lỗi <strong>bạc màu (desaturation)</strong> và <strong>lem màu qua biên giới vật thể (color bleeding)</strong>,
                          chúng tôi đề xuất kiến trúc <strong>VietColorNet</strong> cùng quy trình thích nghi miền (Domain Adaptation) nghiêm ngặt.
                        </p>

                        <h3 style="color: #047857; margin-top: 24px;">1. Đóng góp kỹ thuật chính (VietColorNet)</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 12px;">
                          <div style="background: #fafaf9; border: 1px solid #e5e5e4; border-radius: 8px; padding: 16px;">
                            <h4 style="margin: 0 0 8px; color: #b91c1c;">🎯 Edge-Guidance Stream</h4>
                            <p style="font-size: 13.5px; color: #525252; margin: 0;">
                              Sử dụng bộ lọc đạo hàm vi sai Scharr khả vi kết hợp cổng kiểm soát biên (Edge Gate),
                              ngăn chặn màu da, quân phục và bối cảnh tràn qua ranh giới vật thể.
                            </p>
                          </div>
                          <div style="background: #fafaf9; border: 1px solid #e5e5e4; border-radius: 8px; padding: 16px;">
                            <h4 style="margin: 0 0 8px; color: #d97706;">🌐 Strip Pooling Context Bridge</h4>
                            <p style="font-size: 13.5px; color: #525252; margin: 0;">
                              Nắm bắt bối cảnh dải ngang và dọc (bầu trời, sông nước, rừng rậm) với chi phí bộ nhớ cực thấp,
                              vận hành mượt mà trên phần cứng phổ thông RTX 3050 6GB VRAM.
                            </p>
                          </div>
                          <div style="background: #fafaf9; border: 1px solid #e5e5e4; border-radius: 8px; padding: 16px;">
                            <h4 style="margin: 0 0 8px; color: #047857;">🌈 Decoupled Hue-Chroma Heads</h4>
                            <p style="font-size: 13.5px; color: #525252; margin: 0;">
                              Phân tách bài toán thành hướng sắc thái (Hue vector) và độ rực màu (Chroma magnitude),
                              triệt tiêu triệt để hiện tượng ảnh bị ám xám do hàm mất mát L1/L2 trung bình.
                            </p>
                          </div>
                        </div>

                        <h3 style="color: #047857; margin-top: 28px;">2. Bảng đối so sánh chỉ số khoa học</h3>
                        <table class="metric-table">
                          <thead>
                            <tr>
                              <th>Mô hình</th>
                              <th>Kiến trúc</th>
                              <th>Tham số</th>
                              <th>Bảo toàn biên</th>
                              <th>Độ đậm màu (Chroma)</th>
                              <th>Trạng thái</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td><strong>Semantic ResNet34</strong></td>
                              <td>U-Net + CBAM</td>
                              <td>~21.8M</td>
                              <td>Trung bình</td>
                              <td>Thấp (dễ bạc màu)</td>
                              <td><span class="pill-tag baseline">Baseline sơ khai</span></td>
                            </tr>
                            <tr>
                              <td><strong>DDColor gốc</strong></td>
                              <td>ConvNeXt-T + Dual Decoder</td>
                              <td>~36.7M</td>
                              <td>Khá</td>
                              <td>Khá</td>
                              <td><span class="pill-tag baseline">Baseline thuần</span></td>
                            </tr>
                            <tr class="highlight-row">
                              <td><strong>VietColorNet (Đề xuất)</strong></td>
                              <td>Dual-Stream + Edge-Gated</td>
                              <td>~28.4M</td>
                              <td><strong>Rất cao (Edge Stream)</strong></td>
                              <td><strong>Tự nhiên & Tươi sáng</strong></td>
                              <td><span class="pill-tag proposed">Mô hình Khóa luận</span></td>
                            </tr>
                          </tbody>
                        </table>

                        <h3 style="color: #047857; margin-top: 24px;">3. Nguồn dữ liệu tư liệu lịch sử Việt Nam</h3>
                        <p style="font-size: 14px; color: #525252;">
                          Toàn bộ dữ liệu huấn luyện và đánh giá được xây dựng theo giao thức <em>collection-disjoint</em>
                          và <em>video-disjoint</em> nghiêm ngặt từ 4 nguồn: Frame phim truyện chiến tranh Việt Nam,
                          phim màu Super 8 lịch sử (1966-1968), bộ ảnh tư liệu Wikimedia Việt Nam và 3.000 ảnh màu lịch sử thập niên 1960-1970
                          được thẩm định thủ công.
                        </p>
                        """)

                # TAB 4: Video nếu bật cờ enable-video
                if enable_video:
                    with gr.Tab("🎬 Video Tư Liệu"):
                        with gr.Row(equal_height=True):
                            video_input = gr.Video(label="Video tư liệu gốc")
                            video_output = gr.Video(label="Video đã phục chế màu")
                        with gr.Row():
                            video_strength = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Cường độ màu")
                            temporal_strength = gr.Slider(0.0, 0.9, value=0.55, step=0.05,
                                                          label="Ổn định màu giữa các frame (Optical Flow)")
                        with gr.Row():
                            video_run = gr.Button("Tô màu video", variant="primary")
                            gr.ClearButton([video_input, video_output, video_strength, temporal_strength], value="Xóa")
                        video_run.click(colorize_video,
                                        [video_input, video_strength, temporal_strength], video_output)

            gr.HTML(
                "<div class='footer-note'>🇻🇳 Ứng dụng Sắc Việt · Phục chế ảnh cục bộ an toàn trên thiết bị · "
                "Kết quả lưu trữ tại thư mục <code>outputs/demo_outputs/</code>.</div>"
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "outputs" / "reports" / "baseline_selection.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional explicit project checkpoint (e.g. VietColorNet)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo_outputs")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-sam", action="store_true",
                        help="Không dùng SAM để chống lem màu theo ranh giới vật thể")
    parser.add_argument("--max-video-minutes", type=float, default=10.0)
    parser.add_argument("--enable-video", action="store_true",
                        help="Kích hoạt tab xử lý video trong demo")
    args = parser.parse_args()

    engines = load_engines(args)
    if not engines:
        raise RuntimeError("Không tìm thấy mô hình nào hợp lệ để khởi chạy ứng dụng.")

    print("=" * 60)
    print("SẮC VIỆT · ỨNG DỤNG PHỤC DỰNG ẢNH LỊCH SỬ VIỆT NAM")
    print("=" * 60)
    for k, v in engines.items():
        print(f"[{k.upper()}] {v['label']} -> {v['engine'].device_name} ({v['engine'].checkpoint_path.name})")
    print(f"Địa chỉ web: http://{args.host}:{args.port}")

    demo = create_demo(engines, args.output, args.max_video_minutes, args.enable_video)
    allowed_dirs = [
        str(args.output.resolve()),
        str((ROOT / "examples").resolve()),
        str(ROOT.resolve()),
        tempfile.gettempdir(),
    ]
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        show_error=True,
        show_api=False,
        allowed_paths=allowed_dirs,
    )


if __name__ == "__main__":
    main()
