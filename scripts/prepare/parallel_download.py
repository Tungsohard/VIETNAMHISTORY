"""Resumable HTTP range downloader for large public dataset files."""

import argparse
import concurrent.futures
import os
import time
import urllib.request
from pathlib import Path


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "dataset-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_part(url, path, start, end):
    expected = end - start + 1
    current = path.stat().st_size if path.exists() else 0
    if current >= expected:
        return
    for attempt in range(8):
        try:
            position = start + current
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={position}-{end}", "User-Agent": "dataset-downloader/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, path.open("ab") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    current += len(block)
            if current >= expected:
                return
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(args.url)
    chunk = (size + args.workers - 1) // args.workers
    parts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for index in range(args.workers):
            start = index * chunk
            if start >= size:
                break
            end = min(size - 1, start + chunk - 1)
            part = target.with_name(target.name + f".part{index:02d}")
            parts.append(part)
            jobs.append(pool.submit(fetch_part, args.url, part, start, end))
        for job in concurrent.futures.as_completed(jobs):
            job.result()
    assembled = target.with_name(target.name + ".assembling")
    with assembled.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    if assembled.stat().st_size != size:
        raise RuntimeError("assembled file size mismatch")
    os.replace(assembled, target)
    for part in parts:
        part.unlink()
    print(f"Downloaded {target} ({size} bytes)")


if __name__ == "__main__":
    main()
