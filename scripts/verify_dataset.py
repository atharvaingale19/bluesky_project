"""Verify the selected feed files against data/MANIFEST.csv."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "feed_posts"
MANIFEST = ROOT / "data" / "MANIFEST.csv"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not MANIFEST.exists():
        print("MANIFEST.csv does not exist. Run scripts/prepare_data.py first.")
        return 1
    ok = True
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = DATA_DIR / row["filename"]
            if not path.exists():
                print(f"MISSING  {row['filename']}")
                ok = False
                continue
            actual = sha256(path)
            if actual != row["sha256"]:
                print(f"MISMATCH {row['filename']}")
                ok = False
            else:
                print(f"OK       {row['filename']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
