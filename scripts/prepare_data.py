"""Extract the selected Bluesky feed files into data/feed_posts/.

Usage:
    python scripts/prepare_data.py data/source/feed_posts.tar.gz

The script extracts only the 11 feed files used by the study and writes
SHA-256 checksums plus parsed row counts to data/MANIFEST.csv.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "feed_posts"
MANIFEST = ROOT / "data" / "MANIFEST.csv"

EXPECTED = [
    "Science.jsonl.gz", "What's History.jsonl.gz", "News.jsonl.gz",
    "Political Science.jsonl.gz", "Blacksky.jsonl.gz", "BookSky.jsonl.gz",
    "Game Dev.jsonl.gz", "GreenSky.jsonl.gz", "#Disability.jsonl.gz",
    "#UkrainianView.jsonl.gz", "AcademicSky.jsonl.gz",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parsed_rows(path):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def safe_extract_selected(archive):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(EXPECTED)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in wanted or not member.isfile():
                continue
            target = DATA_DIR / name
            with tar.extractfile(member) as src, target.open("wb") as dst:
                if src is None:
                    raise RuntimeError(f"Could not read {member.name}")
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/prepare_data.py data/source/feed_posts.tar.gz")
        return 2
    archive = Path(sys.argv[1]).expanduser().resolve()
    if not archive.exists():
        print(f"Archive not found: {archive}")
        return 1

    safe_extract_selected(archive)
    missing = [x for x in EXPECTED if not (DATA_DIR / x).exists()]
    if missing:
        print("Missing expected files:")
        print("\n".join(f"  - {x}" for x in missing))
        return 1

    rows = []
    for name in EXPECTED:
        path = DATA_DIR / name
        print(f"Checking {name} ...")
        rows.append({
            "filename": name,
            "sha256": sha256(path),
            "rows_parsed": parsed_rows(path),
            "notes": "selected thematic feed archive",
        })

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "sha256", "rows_parsed", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} files in {DATA_DIR}")
    print(f"Manifest written to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
