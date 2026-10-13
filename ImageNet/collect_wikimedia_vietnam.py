"""Collect licensed Vietnam-domain color photographs from Wikimedia Commons."""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

from colorization.runtime import write_json
from colorization.image_dataset import (
    infer_collection_id,
    infer_content_category,
    has_near_duplicate,
    manipulation_reason,
)


ROOT = Path(__file__).resolve().parent
API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_CATEGORIES = [
    "Category:People of Vietnam",
    "Category:Streets in Vietnam",
    "Category:Markets in Vietnam",
    "Category:Buildings in Vietnam",
    "Category:Clothing of Vietnam",
    "Category:Food of Vietnam",
    "Category:Transport in Vietnam",
]
HISTORICAL_CATEGORIES = [
    "Category:Vietnam in the 1960s",
    "Category:Vietnam in the 1970s",
    "Category:South Vietnam in the 1960s",
    "Category:South Vietnam in the 1970s",
    "Category:Saigon in the 1960s",
    "Category:Photograph collections of the Vietnam War",
]
ATTRIBUTION_FIELDS = ["local_path", "title", "page_url", "original_url", "download_url",
                      "author", "license", "license_url", "credit", "description",
                      "source_category", "commons_sha1", "local_sha256", "width", "height",
                      "colorfulness", "dhash", "content_category", "collection_id"]


def clean_html(value):
    value = html.unescape(value or "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def metadata_value(metadata, name):
    return clean_html(metadata.get(name, {}).get("value", ""))


def allowed_license(name):
    normalized = name.strip().lower()
    if "-nc" in normalized or "noncommercial" in normalized or "-nd" in normalized:
        return False
    return (normalized.startswith("cc by") or normalized.startswith("cc0")
            or normalized.startswith("public domain") or normalized.startswith("pdm")
            or normalized == "kogl type 1")


def image_colorfulness(rgb):
    pixels = np.asarray(rgb, dtype=np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    deviation = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(deviation + 0.3 * mean)


def difference_hash(rgb):
    gray = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray)
    bits = values[:, 1:] > values[:, :-1]
    return f"{sum(int(bit) << index for index, bit in enumerate(bits.flat)):016x}"


def api_get(session, params, retries=5, method="get"):
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    last_error = "unknown error"
    for attempt in range(retries):
        elapsed = time.monotonic() - getattr(session, "last_api_request", 0.0)
        if elapsed < session.api_delay:
            time.sleep(session.api_delay - elapsed)
        try:
            if method == "post":
                response = session.post(API, data=params, timeout=60)
            else:
                response = session.get(API, params=params, timeout=60)
            session.last_api_request = time.monotonic()
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else max(5, 2 ** attempt)
            time.sleep(wait)
            continue
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload
    raise RuntimeError(f"Wikimedia API remained unavailable after retries: {last_error}")


def download_bytes(session, url, retries=2, timeout=25):
    last_error = "unknown error"
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=(10, timeout))
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(min(60, 2 ** attempt))
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * 2 ** attempt)
            if attempt + 1 < retries:
                time.sleep(wait)
            continue
        response.raise_for_status()
        return response.content
    raise requests.RequestException(f"Wikimedia image download failed after retries: {last_error}")


def metadata_rejection(row, min_side, historical=False):
    if row["mime"] not in ("image/jpeg", "image/png"):
        return "unsupported_mime"
    if min(int(row["width"]), int(row["height"])) < min_side:
        return "too_small"
    if not allowed_license(row["license"]):
        return "license_not_allowlisted"
    if not row["download_url"]:
        return "missing_download_url"
    if historical and manipulation_reason(row):
        return "suspected_manipulated_color"
    return None


def iter_downloads(rows, user_agent, workers, timeout, retries, min_side, historical):
    """Bound prefetch memory and preserve row order for deterministic deduplication."""
    def fetch(row):
        if metadata_rejection(row, min_side, historical):
            return None, None
        try:
            with requests.Session() as session:
                session.headers.update({"User-Agent": user_agent})
                return download_bytes(session, row["download_url"], retries, timeout), None
        except requests.RequestException as exc:
            return None, exc

    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    iterator = iter(rows)
    try:
        for _ in range(workers):
            row = next(iterator, None)
            if row is not None:
                pending.append((row, executor.submit(fetch, row)))
        while pending:
            row, future = pending.popleft()
            content, error = future.result()
            yield row, content, error
            following = next(iterator, None)
            if following is not None:
                pending.append((following, executor.submit(fetch, following)))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def category_files(session, roots, max_depth, candidate_limit, max_files_per_category):
    queue = [(category, 0, category) for category in roots]
    visited = set()
    files = {}
    progress = tqdm(desc="Discover categories", unit="category")
    try:
        while queue and len(files) < candidate_limit:
            category, depth, root = queue.pop(0)
            if category in visited:
                continue
            visited.add(category)
            continuation = {}
            category_file_count = 0
            while True:
                payload = api_get(session, {"action": "query", "list": "categorymembers",
                                           "cmtitle": category, "cmtype": "file|subcat", "cmlimit": "max",
                                           **continuation})
                for member in payload.get("query", {}).get("categorymembers", []):
                    title = member["title"]
                    if member["ns"] == 6 and category_file_count < max_files_per_category:
                        files.setdefault(title, category)
                        category_file_count += 1
                    elif member["ns"] == 14 and depth < max_depth:
                        lowered = title.lower()
                        if "black and white" not in lowered and "monochrome" not in lowered:
                            queue.append((title, depth + 1, root))
                    if len(files) >= candidate_limit:
                        break
                if ("continue" not in payload or len(files) >= candidate_limit
                        or category_file_count >= max_files_per_category):
                    break
                continuation = {"cmcontinue": payload["continue"]["cmcontinue"]}
            progress.update(1)
            progress.set_postfix(files=len(files), queued=len(queue))
    finally:
        progress.close()
    return files, sorted(visited)


def round_robin(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    balanced = []
    while groups:
        for name in list(groups):
            balanced.append(groups[name].pop(0))
            if not groups[name]:
                del groups[name]
    return balanced


def metadata_batches(session, title_to_category, thumbnail_size):
    titles = list(title_to_category)
    offsets = range(0, len(titles), 50)
    for offset in tqdm(offsets, total=(len(titles) + 49) // 50, desc="Fetch metadata", unit="batch"):
        selected = titles[offset:offset + 50]
        payload = api_get(session, {"action": "query", "prop": "imageinfo", "titles": "|".join(selected),
                                   "iiprop": "url|size|mime|sha1|extmetadata",
                                   "iiurlwidth": thumbnail_size}, method="post")
        for page in payload.get("query", {}).get("pages", []):
            info = page.get("imageinfo", [{}])[0]
            if info:
                yield page["title"], title_to_category[page["title"]], info


def load_journal(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def has_complete_metadata_cache(output, metadata_rows):
    """The discovered CSV is written only after every metadata batch completes."""
    return bool(metadata_rows) and (Path(output) / "discovered_metadata.csv").is_file()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--historical", action="store_true",
                        help="Use historical Vietnam categories and reject manipulated/colorized works")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "wikimedia_vietnam")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--max-files-per-category", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--thumbnail-size", type=int, default=768)
    parser.add_argument("--min-original-side", type=int, default=512)
    parser.add_argument("--min-colorfulness", type=float, default=7.0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-workers", type=int, default=3,
                        help="Concurrent image transfers (1-4); metadata API remains sequential")
    parser.add_argument("--download-timeout", type=float, default=25)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry previously failed downloads; permanent filter rejections remain skipped")
    parser.add_argument("--api-delay", type=float, default=7.0,
                        help="Seconds between API requests; 7 stays below the unidentified 10/min limit")
    parser.add_argument("--user-agent", default="KhoaLuanColorization/1.0 (educational research; Wikimedia Commons API)")
    args = parser.parse_args(argv)
    if (args.max_images < 1 or args.thumbnail_size < 256 or args.max_depth < 0
            or args.api_delay < 0 or args.max_files_per_category < 1
            or not 1 <= args.download_workers <= 4 or args.download_timeout <= 0
            or args.download_retries < 1):
        parser.error("Invalid collection limits")

    categories = args.categories or (HISTORICAL_CATEGORIES if args.historical else DEFAULT_CATEGORIES)
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    journal_path = args.output / "accepted.jsonl"
    rejection_path = args.output / "rejected.jsonl"
    metadata_journal_path = args.output / "metadata.jsonl"
    accepted = load_journal(journal_path)
    rejected = load_journal(rejection_path)
    metadata_rows = load_journal(metadata_journal_path)
    seen_titles = {row["title"] for row in accepted}
    seen_titles.update(row["title"] for row in rejected
                       if not (args.retry_errors and row["reason"] == "download_or_decode_error"))
    cached_metadata_titles = {row["title"] for row in metadata_rows}
    seen_sha1 = {row["commons_sha1"] for row in accepted}
    seen_dhash = {row["dhash"] for row in accepted}

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    session.api_delay = args.api_delay
    session.last_api_request = 0.0
    candidate_limit = args.max_images * args.candidate_multiplier
    discovery_cached = has_complete_metadata_cache(args.output, metadata_rows)
    if discovery_cached:
        title_to_category = {row["title"]: row["source_category"] for row in metadata_rows}
        visited = []
        print(f"Reuse completed metadata cache: {len(metadata_rows):,} candidates")
    else:
        title_to_category, visited = category_files(session, categories, args.max_depth, candidate_limit,
                                                    args.max_files_per_category)
    uncached = {title: category for title, category in title_to_category.items()
                if title not in seen_titles and title not in cached_metadata_titles}
    with metadata_journal_path.open("a", encoding="utf-8") as metadata_handle:
        for title, category, info in metadata_batches(session, uncached, args.thumbnail_size):
            metadata = info.get("extmetadata", {})
            row = {"title": title, "source_category": category,
                   "page_url": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_")),
                   "original_url": info.get("url", ""), "download_url": info.get("thumburl", info.get("url", "")),
                   "commons_sha1": info.get("sha1", ""), "mime": info.get("mime", ""),
                   "width": info.get("width", 0), "height": info.get("height", 0),
                   "author": metadata_value(metadata, "Artist"),
                   "license": metadata_value(metadata, "LicenseShortName"),
                   "license_url": metadata_value(metadata, "LicenseUrl"),
                   "credit": metadata_value(metadata, "Credit"),
                   "description": metadata_value(metadata, "ImageDescription")}
            metadata_rows.append(row)
            cached_metadata_titles.add(title)
            metadata_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            metadata_handle.flush()
    metadata_rows = round_robin(metadata_rows, "source_category")
    write_csv(args.output / "discovered_metadata.csv",
              ["title", "source_category", "page_url", "original_url", "download_url", "commons_sha1",
               "mime", "width", "height", "author", "license", "license_url", "credit", "description"],
              metadata_rows)

    if not args.metadata_only:
        with journal_path.open("a", encoding="utf-8") as accepted_handle, \
                rejection_path.open("a", encoding="utf-8") as rejected_handle:
            pending_rows = [row for row in metadata_rows if row["title"] not in seen_titles]
            print(f"Resume: accepted={len(accepted):,}; pending={len(pending_rows):,}; workers={args.download_workers}")
            downloads = iter_downloads(pending_rows, args.user_agent, args.download_workers,
                                       args.download_timeout, args.download_retries,
                                       args.min_original_side, args.historical)
            progress = tqdm(downloads, total=len(pending_rows), desc="Wikimedia Vietnam")
            progress.set_postfix(accepted=len(accepted))
            for row, content, download_error in progress:
                if len(accepted) >= args.max_images:
                    break
                reason = metadata_rejection(row, args.min_original_side, args.historical)
                if not reason and row["commons_sha1"] in seen_sha1:
                    reason = "duplicate_commons_sha1"
                if reason:
                    rejected_row = {**row, "reason": reason}
                    rejected_handle.write(json.dumps(rejected_row, ensure_ascii=True) + "\n")
                    rejected_handle.flush()
                    rejected.append(rejected_row)
                    continue
                try:
                    if download_error is not None:
                        raise download_error
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
                    colorfulness = image_colorfulness(image)
                    dhash = difference_hash(image)
                    if colorfulness < args.min_colorfulness:
                        reason = "low_colorfulness"
                    elif dhash in seen_dhash:
                        reason = "duplicate_dhash"
                    elif args.historical and has_near_duplicate(dhash, seen_dhash):
                        reason = "duplicate_near_dhash"
                    if reason:
                        rejected_row = {**row, "reason": reason, "colorfulness": colorfulness, "dhash": dhash}
                        rejected_handle.write(json.dumps(rejected_row, ensure_ascii=True) + "\n")
                        rejected_handle.flush()
                        rejected.append(rejected_row)
                        continue
                    filename = f"commons_{row['commons_sha1']}.jpg"
                    destination = image_dir / filename
                    image.save(destination, "JPEG", quality=92, optimize=True)
                    local_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
                    accepted_row = {**row, "local_path": destination.relative_to(args.output).as_posix(),
                                    "local_sha256": local_sha256, "colorfulness": colorfulness, "dhash": dhash,
                                    "content_category": infer_content_category(row),
                                    "collection_id": infer_collection_id(row)}
                    accepted_handle.write(json.dumps(accepted_row, ensure_ascii=True) + "\n")
                    accepted_handle.flush()
                    accepted.append(accepted_row)
                    seen_sha1.add(row["commons_sha1"])
                    seen_dhash.add(dhash)
                    progress.set_postfix(accepted=len(accepted))
                except (requests.RequestException, OSError, ValueError) as exc:
                    rejected_row = {**row, "reason": "download_or_decode_error", "error": str(exc)[:300]}
                    rejected_handle.write(json.dumps(rejected_row, ensure_ascii=True) + "\n")
                    rejected_handle.flush()
                    rejected.append(rejected_row)

            downloads.close()

    write_csv(args.output / "attribution.csv", ATTRIBUTION_FIELDS, accepted)
    accepted_titles = {row["title"] for row in accepted}
    latest_rejections = {row["title"]: row for row in rejected if row["title"] not in accepted_titles}
    rejection_reasons = {}
    for row in latest_rejections.values():
        rejection_reasons[row["reason"]] = rejection_reasons.get(row["reason"], 0) + 1
    manifest = {"source": "Wikimedia Commons", "api": API, "root_categories": categories,
                "visited_category_count": len(visited) if not discovery_cached else None,
                "discovery_reused_from_cache": discovery_cached,
                "download_workers": args.download_workers,
                "download_timeout": args.download_timeout,
                "download_retries": args.download_retries,
                "max_files_per_category": args.max_files_per_category,
                "thumbnail_size": args.thumbnail_size, "min_original_side": args.min_original_side,
                "min_colorfulness": args.min_colorfulness, "metadata_only": args.metadata_only,
                "historical_mode": args.historical,
                "manipulated_color_policy": "metadata matches are rejected" if args.historical else "disabled",
                "near_duplicate_policy": "dHash Hamming distance <= 2" if args.historical else "exact dHash",
                "accepted_count": len(accepted), "rejected_count": len(latest_rejections),
                "rejection_event_count": len(rejected),
                "rejection_reasons": rejection_reasons,
                "license_policy": "CC BY*, CC0, Public Domain/PDM; attribution retained per file",
                "attribution_sha256": hashlib.sha256((args.output / "attribution.csv").read_bytes()).hexdigest()}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
