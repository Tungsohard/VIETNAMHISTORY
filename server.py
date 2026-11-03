"""Sắc Việt web server: custom website + JSON API on top of the VietColorNet engine.

Run:  python server.py --open-browser
The Gradio demo (demo.py) stays available as a fallback and for model comparison.
"""
import argparse
import io
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

import demo as legacy_demo

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
EXAMPLES = ROOT / "examples"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2400
MODES = {"natural", "modern", "vivid"}
RESULT_TTL_SECONDS = 3600

# Only greyscale archive photos are listed; captions describe what is visible (no names or dates
# are asserted because the sample files carry no verified provenance).
EXAMPLE_INFO = {
    "pho_phuong_mien_bac_1973": "Phố phường miền Bắc",
    "pho_moi_nam_dinh_1963": "Dãy phố ven sông",
    "dai_tuong_vo_nguyen_giap_1968": "Đại hội “3 sẵn sàng”",
    "quan_nhan_lich_su_1970": "Cảnh sinh hoạt",
    "doan_quan_lich_su_1975": "Khung hình phim tư liệu",
}


def create_app(engines, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    default_key = "vietcolornet" if "vietcolornet" in engines else next(iter(engines))
    lock = threading.Lock()
    app = FastAPI(title="Sắc Việt", docs_url=None, redoc_url=None)

    def cleanup():
        cutoff = time.time() - RESULT_TTL_SECONDS
        for path in output_dir.glob("web_*.png"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def run(image, mode, strength, model_key):
        item = engines.get(model_key) or engines[default_key]
        gray = ImageOps.grayscale(image).convert("RGB")
        started = time.perf_counter()
        with lock:
            result = item["engine"].colorize_pil(image, strength, mode=mode)
        elapsed = time.perf_counter() - started
        token = uuid.uuid4().hex[:12]
        gray.save(output_dir / f"web_{token}_gray.png", optimize=True)
        result.save(output_dir / f"web_{token}_color.png", optimize=True)
        return {"gray": f"/results/web_{token}_gray.png", "color": f"/results/web_{token}_color.png",
                "seconds": round(elapsed, 2), "width": image.width, "height": image.height,
                "model": item["label"], "mode": mode, "strength": strength}

    def prepare(raw):
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        except Exception:
            raise HTTPException(400, "Không đọc được ảnh. Hãy dùng JPG, PNG hoặc WebP.")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        return image

    @app.get("/api/status")
    def status():
        primary = engines[default_key]
        examples = []
        for path in sorted(EXAMPLES.glob("*.jpg")):
            if path.stem in EXAMPLE_INFO:
                examples.append({"id": path.stem, "title": EXAMPLE_INFO[path.stem], "url": f"/examples/{path.name}"})
        return {"model": primary["label"], "device": primary["engine"].device_name,
                "sam": getattr(primary["engine"], "segmenter", None) is not None,
                "models": [{"key": k, "label": v["label"]} for k, v in engines.items()],
                "default_model": default_key, "examples": examples,
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}

    @app.post("/api/colorize")
    def colorize(file: UploadFile = File(...), mode: str = Form("modern"),
                 strength: float = Form(1.0), model: str = Form("")):
        if mode not in MODES:
            raise HTTPException(400, "Chế độ không hợp lệ.")
        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Ảnh lớn hơn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        cleanup()
        return run(prepare(raw), mode, min(max(strength, 0.5), 1.5), model)

    @app.get("/api/example/{name}")
    def example(name: str, mode: str = "modern"):
        """Colourised example, cached so the landing page does not recompute it."""
        path = EXAMPLES / f"{name}.jpg"
        if not path.is_file() or mode not in MODES:
            raise HTTPException(404, "Không có ảnh mẫu này.")
        gray_cache = output_dir / f"cache_{name}_{mode}_gray.png"
        color_cache = output_dir / f"cache_{name}_{mode}_color.png"
        if not color_cache.is_file():
            image = prepare(path.read_bytes())
            with lock:
                result = engines[default_key]["engine"].colorize_pil(image, 1.0, mode=mode)
            ImageOps.grayscale(image).convert("RGB").save(gray_cache, optimize=True)
            result.save(color_cache, optimize=True)
        return {"gray": f"/results/{gray_cache.name}", "color": f"/results/{color_cache.name}"}

    @app.get("/results/{filename}")
    def results(filename: str):
        path = (output_dir / filename).resolve()
        if path.parent != output_dir.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs" / "reports" / "baseline_selection.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo_outputs")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-sam", action="store_true")
    args = parser.parse_args()

    engines = legacy_demo.load_engines(args)
    if not engines:
        raise RuntimeError("Không tìm thấy mô hình nào hợp lệ để khởi chạy ứng dụng.")
    for key, item in engines.items():
        print(f"[{key}] {item['label']} -> {item['engine'].device_name}")
    url = f"http://{args.host}:{args.port}"
    print(f"Sắc Việt web: {url}")
    if args.open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(engines, args.output), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
