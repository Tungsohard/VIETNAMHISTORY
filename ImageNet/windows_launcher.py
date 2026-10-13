"""Windows entry point for the packaged Sắc Việt application."""
import os
import builtins
import io
import socket
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path


def _show_error(message):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "Sắc Việt - Lỗi khởi động", 0x10)
    except Exception:
        pass


def _existing_app_url(port=7860):
    """Return the URL when an existing Sắc Việt web UI is already available."""
    url = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            if response.status == 200:
                return url
    except Exception:
        return None
    return None


def _available_port(preferred=7860):
    """Use the preferred port when free, otherwise ask Windows for a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_app(url, server_state, timeout=180):
    """Wait until Gradio is ready, while surfacing background startup errors."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_state.get("error"):
            raise RuntimeError("Máy chủ Sắc Việt không thể khởi động") from server_state["error"]
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Sắc Việt khởi động quá thời gian cho phép.")


def _open_desktop_window(url):
    import webview

    webview.create_window(
        "Sắc Việt · Phục chế và Tô màu Ảnh Tư liệu",
        url=url,
        width=1440,
        height=920,
        min_size=(1100, 720),
        resizable=True,
        text_select=True,
    )
    webview.start(gui="edgechromium", debug=False)


def main():
    app_dir = Path(sys.executable).resolve().parent
    # Gradio chỉ phục vụ file đầu ra từ các vị trí tin cậy. Đặt kết quả ngay
    # trong thư mục ứng dụng (cũng là current working directory) để bản cài và
    # bản portable đều có thể hiển thị ảnh mà không vướng kiểm tra cache.
    output_dir = app_dir / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(app_dir)
    # Bản windowed không có console nên Uvicorn cần một stream thật có isatty().
    runtime_log = open(app_dir / "SacViet.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = runtime_log
    if sys.stderr is None:
        sys.stderr = runtime_log

    # Nhấp đúp nhiều lần không được nạp model/GPU lặp lại. Phiên thứ hai chỉ
    # tạo thêm một cửa sổ desktop nối tới máy chủ cục bộ đang chạy.
    existing_url = _existing_app_url()
    if existing_url:
        _open_desktop_window(existing_url)
        return
    port = _available_port()

    # Model và selection nằm ngoài exe. Người dùng chỉ cần thay các file này để
    # cập nhật model; ứng dụng sẽ đọc lại ở lần khởi động kế tiếp.
    sys.argv = [
        "SacViet.exe",
        "--selection", str(app_dir / "reports" / "production_selection.json"),
        "--checkpoint", str(app_dir / "runs" / "vietcolornet_v6_refined" / "best.pth"),
        "--output", str(output_dir),
        "--port", str(port),
    ]
    original_open = builtins.open

    def frozen_dependency_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            normalized = os.fsdecode(os.fspath(file)).replace("\\", "/").lower()
            if normalized.endswith("/safehttpx/version.txt"):
                return io.StringIO("0.1.7")
        return original_open(file, *args, **kwargs)

    builtins.open = frozen_dependency_open
    try:
        from demo import main as demo_main
        server_state = {}

        def run_server():
            try:
                demo_main()
            except BaseException as exc:
                server_state["error"] = exc

        server_thread = threading.Thread(target=run_server, name="SacVietServer", daemon=True)
        server_thread.start()
        app_url = f"http://127.0.0.1:{port}"
        _wait_for_app(app_url, server_state)
        _open_desktop_window(app_url)
    finally:
        builtins.open = original_open


if __name__ == "__main__":
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        log_dir = Path(sys.executable).resolve().parent
        log_path = log_dir / "startup-error.log"
        log_path.write_text(details, encoding="utf-8")
        _show_error(
            "Không thể khởi động Sắc Việt.\n\n"
            "Chi tiết đã được lưu tại:\n"
            f"{log_path}"
        )
        raise
