"""
Extract image frames (slides/scenes) and audio (tracks and transcription) from video files.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Add torch lib to DLL directory for CUDA support in CTranslate2
    torch_lib = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
import subprocess
import imageio_ffmpeg


def format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def format_srt_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "path": str(video_path.resolve()),
        "name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "duration_formatted": format_timestamp(duration),
    }


def extract_audio(video_path: Path, output_dir: Path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp3_path = output_dir / f"{video_path.stem}_audio.mp3"
    wav_path = output_dir / f"{video_path.stem}_audio_16k.wav"

    print(f"\n[1/4] Extracting audio from {video_path.name}...")
    # Extract MP3 (high quality compressed)
    cmd_mp3 = [
        ffmpeg_exe, "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
        str(mp3_path)
    ]
    subprocess.run(cmd_mp3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f" -> Saved MP3: {mp3_path.name} ({mp3_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Extract WAV (16kHz mono for speech recognition)
    cmd_wav = [
        ffmpeg_exe, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f" -> Saved WAV 16kHz: {wav_path.name} ({wav_path.stat().st_size / 1024 / 1024:.2f} MB)")

    return mp3_path, wav_path


def extract_slides(video_path: Path, output_dir: Path, diff_threshold: float = 16.0, min_interval_sec: int = 5):
    """
    Extracts key presentation slides by sampling at 1 frame per second
    and detecting significant, settled scene changes.
    """
    slides_dir = output_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/4] Detecting and extracting key visual slides/frames...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames_per_sec = max(1, int(round(fps)))
    prev_thumb = None
    last_saved_sec = -999
    slides = []

    sec = 0
    t0 = time.time()
    
    # Store candidate changes to allow screen settling (avoiding blur/scrolling in progress)
    candidates = []

    while True:
        # Fast grab remaining frames in this 1-second interval
        for _ in range(frames_per_sec - 1):
            if not cap.grab():
                break
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        # Presentation region (roughly left 75% where slide / screen share is)
        slide_roi = frame[int(h * 0.12):int(h * 0.92), 0:int(w * 0.75)]
        gray = cv2.cvtColor(slide_roi, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (80, 45))

        if prev_thumb is None:
            # First frame of the video
            candidates.append((sec, frame.copy(), 100.0))
            prev_thumb = thumb
            last_saved_sec = sec
        else:
            diff = float(np.mean(np.abs(thumb.astype(float) - prev_thumb.astype(float))))
            if diff > diff_threshold:
                candidates.append((sec, frame.copy(), diff))
                prev_thumb = thumb

        sec += 1
        if sec % 300 == 0:
            print(f" -> Processed {sec}/{int(duration)}s ({sec / duration * 100:.1f}%), {len(candidates)} changes found...")

    cap.release()

    # Post-process candidates: debounce closely spaced changes (keep the settled final frame)
    filtered = []
    i = 0
    while i < len(candidates):
        # Group candidates that are within 3 seconds of each other
        group = [candidates[i]]
        j = i + 1
        while j < len(candidates) and (candidates[j][0] - group[-1][0]) <= 3:
            group.append(candidates[j])
            j += 1

        # Pick the last frame of the transition (the settled slide)
        chosen = group[-1]
        if not filtered or (chosen[0] - filtered[-1][0] >= min_interval_sec):
            filtered.append(chosen)
        i = j

    print(f" -> Filtered to {len(filtered)} distinct slide keyframes. Saving images...")
    for idx, (t_sec, frame_bgr, diff_val) in enumerate(filtered, 1):
        ts_str = format_timestamp(t_sec).replace(":", "m") + "s"
        filename = f"slide_{idx:02d}_{ts_str}.jpg"
        filepath = slides_dir / filename
        cv2.imwrite(str(filepath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        slides.append({
            "index": idx,
            "filename": filename,
            "relative_path": f"slides/{filename}",
            "timestamp_seconds": t_sec,
            "timestamp_formatted": format_timestamp(t_sec),
            "diff_score": round(diff_val, 2)
        })

    with open(output_dir / "slides_manifest.json", "w", encoding="utf-8") as f:
        json.dump(slides, f, ensure_ascii=False, indent=2)

    print(f" -> Saved {len(slides)} slides in {slides_dir.name}/ (took {time.time()-t0:.1f}s)")
    return slides


def transcribe_audio(audio_path: Path, output_dir: Path, model_size: str = "base", language: str = "vi"):
    print(f"\n[3/4] Transcribing audio with faster-whisper ({model_size}, lang={language})...")
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f" -> Using device: {device} ({compute_type})")

    t0 = time.time()
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"CUDA initialization failed ({e}), falling back to CPU int8...")
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    results = []
    srt_lines = []
    txt_lines = []

    print(f" -> Audio duration: {info.duration:.1f}s. Transcribing speech...")
    for i, seg in enumerate(segments, 1):
        clean_text = seg.text.strip()
        if not clean_text:
            continue

        results.append({
            "id": i,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "start_formatted": format_timestamp(seg.start),
            "end_formatted": format_timestamp(seg.end),
            "text": clean_text
        })

        # SRT format
        srt_lines.append(f"{i}\n{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n{clean_text}\n")

        # Plain text with timestamps
        txt_lines.append(f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {clean_text}")

        if i % 50 == 0:
            print(f"    ... Transcribed {i} segments ({format_timestamp(seg.end)})")

    # Save transcripts
    transcript_json = output_dir / "transcript.json"
    transcript_txt = output_dir / "transcript.txt"
    transcript_srt = output_dir / "transcript.srt"

    with open(transcript_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(transcript_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    with open(transcript_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f" -> Transcription completed in {time.time()-t0:.1f}s ({len(results)} segments)")
    print(f" -> Saved: {transcript_txt.name}, {transcript_srt.name}, {transcript_json.name}")
    return results


def generate_synchronized_report(video_info: dict, slides: list, transcript_segments: list, output_dir: Path):
    print("\n[4/4] Aligning slides with spoken transcript & generating reports...")

    # For each slide, find transcript segments that occurred while this slide was visible
    for i, slide in enumerate(slides):
        t_start = slide["timestamp_seconds"]
        t_end = slides[i + 1]["timestamp_seconds"] if i + 1 < len(slides) else video_info["duration_seconds"]
        slide["end_seconds"] = t_end
        slide["end_formatted"] = format_timestamp(t_end)

        slide_speech = [
            seg for seg in transcript_segments
            if (seg["start"] < t_end and seg["end"] > t_start)
        ]
        slide["spoken_text"] = " ".join(seg["text"] for seg in slide_speech)
        slide["segments"] = slide_speech

    # Generate Markdown Report
    md_content = [
        f"# Báo Cáo Trích Xuất Video: {video_info['name']}\n",
        f"- **Thời lượng video**: {video_info['duration_formatted']} ({video_info['duration_seconds']:.1f} giây)",
        f"- **Độ phân giải**: {video_info['width']}x{video_info['height']} @ {video_info['fps']} fps",
        f"- **Tổng số slide/khung hình chính trích xuất**: {len(slides)}",
        f"- **Tổng số đoạn lời thoại trích xuất**: {len(transcript_segments)}\n",
        "## Tệp tin kết quả\n",
        f"- **Âm thanh MP3**: [`{video_info['name'].replace('.mp4', '_audio.mp3')}`]({video_info['name'].replace('.mp4', '_audio.mp3')})",
        f"- **Âm thanh WAV (16kHz)**: [`{video_info['name'].replace('.mp4', '_audio_16k.wav')}`]({video_info['name'].replace('.mp4', '_audio_16k.wav')})",
        "- **Phụ đề SRT**: [`transcript.srt`](transcript.srt)",
        "- **Bản ghi thoại Text**: [`transcript.txt`](transcript.txt)",
        "- **Dữ liệu cấu trúc JSON**: [`transcript.json`](transcript.json), [`slides_manifest.json`](slides_manifest.json)\n",
        "## Chi Tiết Slide & Nội Dung Thuyết Trình Tương Ứng\n",
    ]

    for slide in slides:
        md_content.append(f"### Slide {slide['index']:02d} [{slide['timestamp_formatted']} - {slide['end_formatted']}]")
        md_content.append(f"![Slide {slide['index']}]({slide['relative_path']})\n")
        if slide["spoken_text"]:
            md_content.append(f"> **Nội dung trao đổi / thuyết trình:**\n> {slide['spoken_text']}\n")
        else:
            md_content.append("*(Không có lời thoại trong khoảng thời gian này)*\n")

    readme_path = output_dir / "SUMMARY_REPORT.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    # Generate Interactive HTML viewer
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trích xuất Video & Thuyết trình - {video_info['name']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1200px;
            margin: 0 auto 30px auto;
            background: #1e293b;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
        .meta-badges {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
        .badge {{ background: #334155; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; }}
        .audio-player {{ width: 100%; margin-top: 15px; border-radius: 8px; }}
        .container {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }}
        .slide-card {{
            background: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            overflow: hidden;
            display: grid;
            grid-template-columns: 550px 1fr;
            transition: border-color 0.2s;
        }}
        @media (max-width: 900px) {{
            .slide-card {{ grid-template-columns: 1fr; }}
        }}
        .slide-card:hover {{ border-color: #38bdf8; }}
        .img-wrap {{ background: #000; display: flex; align-items: center; justify-content: center; }}
        .img-wrap img {{ width: 100%; height: auto; display: block; }}
        .slide-content {{ padding: 20px; display: flex; flex-direction: column; justify-content: space-between; }}
        .slide-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .slide-title {{ font-size: 18px; font-weight: 600; color: #f1f5f9; }}
        .timestamp {{ background: #0369a1; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
        .speech-box {{
            background: #0f172a;
            border-left: 4px solid #38bdf8;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.6;
            color: #cbd5e1;
            max-height: 250px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Báo Cáo Trích Xuất Video: {video_info['name']}</h1>
        <div class="meta-badges">
            <span class="badge">⏱ Thời lượng: {video_info['duration_formatted']}</span>
            <span class="badge">📺 Độ phân giải: {video_info['width']}x{video_info['height']}</span>
            <span class="badge">🖼 {len(slides)} Slide/Khung hình chính</span>
            <span class="badge">💬 {len(transcript_segments)} Đoạn hội thoại</span>
        </div>
        <audio class="audio-player" controls src="{video_info['name'].replace('.mp4', '_audio.mp3')}"></audio>
    </div>
    <div class="container">
"""

    for slide in slides:
        text_html = slide["spoken_text"] if slide["spoken_text"] else "<em>(Không có phát biểu trong slide này)</em>"
        html_content += f"""
        <div class="slide-card">
            <div class="img-wrap">
                <a href="{slide['relative_path']}" target="_blank">
                    <img src="{slide['relative_path']}" alt="Slide {slide['index']}" loading="lazy" />
                </a>
            </div>
            <div class="slide-content">
                <div>
                    <div class="slide-header">
                        <span class="slide-title">Slide #{slide['index']:02d}</span>
                        <span class="timestamp">{slide['timestamp_formatted']} &rarr; {slide['end_formatted']}</span>
                    </div>
                    <div class="speech-box">
                        {text_html}
                    </div>
                </div>
            </div>
        </div>
        """

    html_content += """
    </div>
</body>
</html>
"""

    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" -> Generated report: {readme_path.name}")
    print(f" -> Generated interactive viewer: {html_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract image content and audio transcription from video")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    parser.add_argument("--output-dir", type=str, default="extracted_content", help="Output directory")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size (tiny, base, small, medium)")
    parser.add_argument("--language", type=str, default="vi", help="Audio language")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio extraction")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip speech transcription")
    parser.add_argument("--skip-slides", action="store_true", help="Skip slide extraction")
    args = parser.parse_args()

    # Find video if not specified
    if args.video:
        video_path = Path(args.video)
    else:
        root_dir = Path.cwd()
        videos = [p for p in root_dir.glob("*.mp4")] + [p for p in root_dir.glob("*.avi")] + [p for p in root_dir.glob("*.mkv")]
        if not videos:
            print("No video files found in the current directory.")
            sys.exit(1)
        video_path = videos[0]
        print(f"Auto-selected video: {video_path.name}")

    if not video_path.exists():
        print(f"Video file does not exist: {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(video_path)
    print("=" * 60)
    print(f"VIDEO: {video_info['name']}")
    print(f"Duration: {video_info['duration_formatted']} ({video_info['duration_seconds']:.1f}s)")
    print(f"Resolution: {video_info['width']}x{video_info['height']} @ {video_info['fps']} fps")
    print(f"Output Directory: {out_dir.resolve()}")
    print("=" * 60)

    # 1. Audio extraction
    mp3_path, wav_path = None, None
    if not args.skip_audio:
        mp3_path, wav_path = extract_audio(video_path, out_dir)

    # 2. Keyframe / Slide extraction
    slides = []
    if not args.skip_slides:
        slides = extract_slides(video_path, out_dir)

    # 3. Speech Transcription
    transcript = []
    if not args.skip_transcribe:
        target_audio = mp3_path if mp3_path and mp3_path.exists() else (out_dir / f"{video_path.stem}_audio.mp3")
        if target_audio.exists():
            transcript = transcribe_audio(target_audio, out_dir, model_size=args.whisper_model, language=args.language)
        else:
            print(f"Audio file not found for transcription: {target_audio}")

    # 4. Alignment & Summary Report
    if slides and transcript:
        generate_synchronized_report(video_info, slides, transcript, out_dir)

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f"Extracted content is stored in: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
