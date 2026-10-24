"""Download reusable Vietnam images and metadata from Wikimedia Commons."""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED = ("public domain", "cc0", "cc by", "cc-by")
START_CATEGORIES = (
    "Category:Historical images of Vietnam",
    "Category:20th-century photographs of Vietnam",
    "Category:Culture of Vietnam",
    "Category:Architecture of Vietnam",
)


def api(params):
    params.update({"format": "json", "formatversion": "2"})
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VietnamColorizationDataset/1.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 6:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))


def category_files(category, depth, seen_categories):
    if category in seen_categories:
        return
    seen_categories.add(category)
    token = None
    while True:
        params = {
            "action": "query", "list": "categorymembers", "cmtitle": category,
            "cmnamespace": "6|14", "cmlimit": "500",
        }
        if token:
            params["cmcontinue"] = token
        data = api(params)
        for member in data["query"]["categorymembers"]:
            if member["ns"] == 6:
                yield member["title"]
            elif depth > 0:
                yield from category_files(member["title"], depth - 1, seen_categories)
        token = data.get("continue", {}).get("cmcontinue")
        if not token:
            break


def image_info(title):
    data = api({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1920",
    })
    page = data["query"]["pages"][0]
    return page.get("imageinfo", [None])[0]


def clean(value):
    return re.sub(r"<[^>]+>", "", value or "").strip()


def download_image(url, path):
    for attempt in range(7):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "VietnamColorizationDataset/1.0 (research)"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 6:
                raise
            time.sleep(min(90, 8 * (attempt + 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="wikimedia-vietnam")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.output)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"
    existing = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if rows and rows[0] and rows[0][0] != "title":
            repaired = manifest_path.with_suffix(".repair.csv")
            with repaired.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["title", "filename", "url", "description_url", "license", "artist", "credit"])
                writer.writerows(rows)
            repaired.replace(manifest_path)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {row["title"] for row in csv.DictReader(handle)}
    fields = ["title", "filename", "url", "description_url", "license", "artist", "credit"]
    write_header = not manifest_path.exists()
    count = len(existing)
    seen_titles = set(existing)
    with manifest_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        seen_categories = set()
        for category in START_CATEGORIES:
            for title in category_files(category, args.depth, seen_categories):
                if title in seen_titles or count >= args.max_images:
                    continue
                seen_titles.add(title)
                try:
                    info = image_info(title)
                    if not info or not info.get("mime", "").startswith("image/"):
                        continue
                    meta = info.get("extmetadata", {})
                    license_name = clean(meta.get("LicenseShortName", {}).get("value", ""))
                    usage = clean(meta.get("UsageTerms", {}).get("value", ""))
                    if not any(key in (license_name + " " + usage).lower() for key in ALLOWED):
                        continue
                    url = info.get("thumburl") or info["url"]
                    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
                    filename = f"{count:06d}{suffix}"
                    download_image(url, images / filename)
                    writer.writerow({
                        "title": title, "filename": filename, "url": info["url"],
                        "description_url": info.get("descriptionurl", ""),
                        "license": license_name or usage,
                        "artist": clean(meta.get("Artist", {}).get("value", "")),
                        "credit": clean(meta.get("Credit", {}).get("value", "")),
                    })
                    handle.flush()
                    count += 1
                    print(f"[{count}/{args.max_images}] {title}", flush=True)
                    time.sleep(1)
                except Exception as exc:
                    print(f"SKIP {title}: {exc}", flush=True)
                    time.sleep(1)
                if count >= args.max_images:
                    return


if __name__ == "__main__":
    main()
