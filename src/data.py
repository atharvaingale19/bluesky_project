import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _first(d, names, default=None):
    if not isinstance(d, dict):
        return default
    for name in names:
        if name in d and d[name] is not None:
            return d[name]
    return default


def _parse_language(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _parse_created_at(value):
    """Handle the actual Bluesky Social Dataset timestamp format and common variants."""
    if value is None:
        return pd.NaT

    # Actual selected feed files use YYYYMMDDHHMM as an integer/string.
    text = str(value).strip()
    if text.isdigit() and len(text) in (12, 14):
        fmt = "%Y%m%d%H%M" if len(text) == 12 else "%Y%m%d%H%M%S"
        return pd.to_datetime(text, format=fmt, utc=True, errors="coerce")

    return pd.to_datetime(value, utc=True, errors="coerce")


def normalize_record(raw):
    """Normalize records from the selected Bluesky Social Dataset feed exports.

    The actual feed files use fields such as post_id, user_id, date,
    text, langs, and repost_count. Common nested/export variants are also
    accepted so the loader remains useful if a feed file is reorganized.
    """
    candidates = list(_walk_candidates(raw))

    post_id = author_id = created_at = text = repost_count = language = None

    for d in candidates:
        post_id = post_id or _first(d, ["post_id", "postId", "uri", "id"])
        author_id = author_id or _first(
            d, ["user_id", "author_id", "authorDid", "did", "author_did"]
        )
        created_at = created_at or _first(
            d, ["date", "created_at", "createdAt", "indexedAt", "indexed_at"]
        )
        text = text if text is not None else _first(d, ["text"])
        repost_count = repost_count if repost_count is not None else _first(
            d, ["repost_count", "repostCount", "reposts"]
        )
        language = language or _first(d, ["langs", "language", "lang"])

        author = d.get("author")
        if isinstance(author, dict):
            author_id = author_id or _first(author, ["did", "id", "handle"])

    if post_id is None or author_id is None or created_at is None:
        return None

    try:
        repost_count = int(repost_count or 0)
    except (TypeError, ValueError):
        repost_count = 0

    return {
        "post_id": str(post_id),
        "author_id": str(author_id),
        "created_at": _parse_created_at(created_at),
        "text": "" if text is None else str(text),
        "repost_count": max(0, repost_count),
        "language": _parse_language(language),
    }


def _walk_candidates(obj):
    """Yield dictionaries found in common nested Bluesky export structures."""
    if isinstance(obj, dict):
        yield obj
        for key in ("post", "record", "author", "value", "data"):
            if key in obj:
                yield from _walk_candidates(obj[key])
    elif isinstance(obj, list):
        for x in obj:
            yield from _walk_candidates(x)


def discover_files(data_dir):
    """Find JSONL.GZ feed files recursively, including nested feed_posts folders."""
    root = Path(data_dir)
    return sorted(root.rglob("*.jsonl.gz"))


def load_jsonl_gz(path):
    path = Path(path)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = normalize_record(json.loads(line))
            if row is not None:
                row["source_file"] = path.name
                rows.append(row)
    return rows


def load_all(data_dir):
    paths = discover_files(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"No .jsonl.gz files found under {data_dir}. "
            "Put the selected feed_posts dataset there."
        )

    rows = []
    for path in paths:
        rows.extend(load_jsonl_gz(path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No usable records were parsed from the dataset.")

    df = df.drop_duplicates("post_id", keep="first").copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["repost_count"] = pd.to_numeric(
        df["repost_count"], errors="coerce"
    ).fillna(0).clip(lower=0).astype(int)

    df = df.dropna(subset=["created_at", "author_id", "post_id"]).copy()
    return df.sort_values(
        ["author_id", "created_at", "post_id"]
    ).reset_index(drop=True)


def add_relative_target(df, k=5):
    """Create the retrospective author-relative target.

    Only strictly earlier timestamps are eligible. Posts sharing the same
    timestamp are not treated as earlier than one another.
    """
    df = df.sort_values(["author_id", "created_at", "post_id"]).reset_index(drop=True).copy()
    baselines = np.full(len(df), np.nan, dtype=float)

    for _, group in df.groupby("author_id", sort=False):
        positions = group.index.to_numpy()
        times = group["created_at"].to_numpy()
        counts = group["repost_count"].to_numpy(dtype=float)

        history = []
        j = 0
        while j < len(positions):
            group_end = j
            while (
                group_end + 1 < len(positions)
                and times[group_end + 1] == times[j]
            ):
                group_end += 1

            baseline = float(np.median(history[-k:])) if len(history) >= k else np.nan
            baselines[positions[j : group_end + 1]] = baseline

            history.extend(counts[j : group_end + 1])
            j = group_end + 1

    df["author_baseline"] = baselines
    df["y_abs"] = np.log1p(df["repost_count"].astype(float))
    df["y_rel"] = np.log1p(df["repost_count"].astype(float)) - np.log1p(
        df["author_baseline"]
    )
    return df
