#!/usr/bin/env python3
"""Download a small, fixed environmental-noise subset of ESC-50.

The full MUSAN archive is ~11 GB.  v0 only needs controlled non-speech noise,
so this downloads 24 publicly available ESC-50 clips (four clips from each of
six environmental categories), plus the metadata used to select them.
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.request
from pathlib import Path


META_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
WAV_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/{name}"
CATEGORIES = ("rain", "thunderstorm", "wind", "sea_waves", "water_drops", "engine")


def download(url: str, destination: Path, retries: int = 6) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AuK-dataset-builder/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temporary.stat().st_size < 1000:
                raise RuntimeError(f"downloaded file is unexpectedly small: {temporary}")
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 20))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/noise_esc50"))
    ap.add_argument("--per-category", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(META_URL, timeout=60) as r:
        rows = list(csv.DictReader(io.TextIOWrapper(r, encoding="utf-8")))
    selected = []
    for category in CATEGORIES:
        category_rows = [r for r in rows if r["category"] == category]
        if args.per_category < 1 or args.per_category > len(category_rows):
            raise ValueError(
                f"--per-category must be between 1 and {len(category_rows)}, got {args.per_category}"
            )
        selected.extend(category_rows[: args.per_category])
    selected.sort(key=lambda r: (r["category"], r["filename"]))
    for row in selected:
        dest = args.out / row["filename"]
        if not dest.exists() or dest.stat().st_size < 1000:
            print("downloading", row["filename"])
            download(WAV_URL.format(name=row["filename"]), dest)
    (args.out / "esc50_subset.csv").write_text(
        "filename,category,fold,target,source_url\n" +
        "".join(f"{r['filename']},{r['category']},{r['fold']},{r['target']},{WAV_URL.format(name=r['filename'])}\n" for r in selected),
        encoding="utf-8",
    )
    print(f"downloaded {len(selected)} clips into {args.out}")


if __name__ == "__main__":
    main()
