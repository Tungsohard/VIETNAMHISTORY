"""Download a licensed, checksummed set of historical Vietnam videos from Commons."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests

from collect_wikimedia_vietnam import API, allowed_license, api_get, metadata_value
from colorization.runtime import write_json


ROOT = Path(__file__).resolve().parent
FIELDS = ["id", "title", "preferred_split", "source_group", "page_url", "original_url", "author", "license", "license_url",
          "credit", "description", "date", "mime", "width", "height", "duration",
          "commons_sha1", "remote_size", "local_path", "local_sha1", "local_sha256"]


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_stream(session, url, destination, expected_sha1, retries=6):
    if destination.exists() and file_digest(destination, "sha1") == expected_sha1:
        return
    part = destination.with_suffix(destination.suffix + ".part")
    last_error = "unknown error"
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 120))
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(120, 10 * 2 ** attempt)
                last_error = f"HTTP {response.status_code}"
                time.sleep(wait)
                continue
            response.raise_for_status()
            mode = "ab" if offset and response.status_code == 206 else "wb"
            with part.open(mode) as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
            if file_digest(part, "sha1") != expected_sha1:
                last_error = "SHA-1 mismatch after download"
                part.unlink(missing_ok=True)
                time.sleep(min(60, 2 ** attempt))
                continue
            part.replace(destination)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "historical_vietnam_videos.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "historical_vietnam_videos")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--api-delay", type=float, default=7.0)
    parser.add_argument("--user-agent", default="KhoaLuanColorization/1.0 (educational research; Wikimedia Commons API)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    configured = {row["title"]: row for row in config["videos"]}
    if len(configured) != len(config["videos"]):
        raise ValueError("Configured video titles must be unique")
    configured_ids = [row["id"] for row in config["videos"]]
    if len(set(configured_ids)) != len(configured_ids):
        raise ValueError("Configured video IDs must be unique")
    exclusions = set(config.get("frozen_test_exclusions", []))
    overlap = exclusions & configured.keys()
    if overlap:
        raise ValueError(f"Frozen final-test videos cannot be reused: {sorted(overlap)}")
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    session.api_delay = args.api_delay
    session.last_api_request = 0.0
    payload = api_get(session, {"action": "query", "prop": "imageinfo", "titles": "|".join(configured),
                                "iiprop": "url|size|mime|sha1|extmetadata"}, method="post")
    rows = []
    for page in payload.get("query", {}).get("pages", []):
        info = page.get("imageinfo", [{}])[0]
        metadata = info.get("extmetadata", {})
        title = page["title"]
        configured_row = configured[title]
        preferred_split = configured_row.get("preferred_split", "")
        if preferred_split not in ("", "train", "val"):
            raise ValueError(f"Invalid preferred_split for {title}: {preferred_split}")
        row = {"id": configured_row["id"], "title": title,
               "preferred_split": preferred_split,
               "source_group": configured_row.get("source_group", configured_row["id"]),
               "page_url": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_")),
               "original_url": info.get("url", ""), "author": metadata_value(metadata, "Artist"),
               "license": metadata_value(metadata, "LicenseShortName"),
               "license_url": metadata_value(metadata, "LicenseUrl"),
               "credit": metadata_value(metadata, "Credit"),
               "description": metadata_value(metadata, "ImageDescription"),
               "date": metadata_value(metadata, "DateTimeOriginal") or metadata_value(metadata, "DateTime"),
               "mime": info.get("mime", ""), "width": info.get("width", 0), "height": info.get("height", 0),
               "duration": info.get("duration", metadata_value(metadata, "Duration")),
               "commons_sha1": info.get("sha1", ""), "remote_size": info.get("size", 0),
               "local_path": "", "local_sha1": "", "local_sha256": ""}
        if not allowed_license(row["license"]):
            raise ValueError(f"License is not allowed for {title}: {row['license']}")
        if row["mime"] not in ("video/webm", "application/ogg", "video/ogg"):
            raise ValueError(f"Unsupported video MIME for {title}: {row['mime']}")
        rows.append(row)
    if len(rows) != len(configured):
        raise ValueError("Commons did not return every configured video")
    rows.sort(key=lambda row: row["id"])

    args.output.mkdir(parents=True, exist_ok=True)
    video_dir = args.output / "videos"
    video_dir.mkdir(exist_ok=True)
    if not args.metadata_only:
        for index, row in enumerate(rows, 1):
            suffix = Path(row["title"].removeprefix("File:")).suffix.lower()
            destination = video_dir / f"{row['id']}{suffix}"
            print(f"[{index}/{len(rows)}] {row['title']} ({int(row['remote_size']) / 1024**2:.1f} MiB)", flush=True)
            download_stream(session, row["original_url"], destination, row["commons_sha1"])
            row.update(local_path=destination.relative_to(args.output).as_posix(),
                       local_sha1=file_digest(destination, "sha1"),
                       local_sha256=file_digest(destination, "sha256"))

    attribution = args.output / "attribution.csv"
    with attribution.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"source": "Wikimedia Commons", "api": API, "purpose": config["purpose"],
                "frozen_test_exclusions": sorted(exclusions),
                "metadata_only": args.metadata_only, "video_count": len(rows),
                "downloaded_count": sum(bool(row["local_path"]) for row in rows),
                "total_remote_bytes": sum(int(row["remote_size"]) for row in rows),
                "preferred_split_counts": {
                    split: sum(row["preferred_split"] == split for row in rows)
                    for split in ("train", "val")
                },
                "attribution_sha256": file_digest(attribution, "sha256"), "videos": rows}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"video_count": manifest["video_count"],
                      "downloaded_count": manifest["downloaded_count"],
                      "total_remote_mib": round(manifest["total_remote_bytes"] / 1024**2, 1)}, indent=2))


if __name__ == "__main__":
    main()
