import pandas as pd

from src.data import add_relative_target, normalize_record


def test_actual_feed_schema_is_normalized():
    row = {
        "post_id": 123,
        "user_id": 456,
        "date": 202403182356,
        "text": "hello",
        "langs": ["eng"],
        "repost_count": 7,
    }
    out = normalize_record(row)
    assert out["post_id"] == "123"
    assert out["author_id"] == "456"
    assert out["language"] == "eng"
    assert out["repost_count"] == 7
    assert out["created_at"].year == 2024
    assert out["created_at"].month == 3


def test_same_timestamp_posts_are_not_history():
    df = pd.DataFrame([
        {"post_id": "a", "author_id": "u", "created_at": "2024-01-01T00:00:00Z", "text": "a", "repost_count": 1, "language": "eng"},
        {"post_id": "b", "author_id": "u", "created_at": "2024-01-01T00:00:00Z", "text": "b", "repost_count": 10, "language": "eng"},
        {"post_id": "c", "author_id": "u", "created_at": "2024-01-02T00:00:00Z", "text": "c", "repost_count": 5, "language": "eng"},
        {"post_id": "d", "author_id": "u", "created_at": "2024-01-03T00:00:00Z", "text": "d", "repost_count": 8, "language": "eng"},
        {"post_id": "e", "author_id": "u", "created_at": "2024-01-04T00:00:00Z", "text": "e", "repost_count": 4, "language": "eng"},
        {"post_id": "f", "author_id": "u", "created_at": "2024-01-05T00:00:00Z", "text": "f", "repost_count": 9, "language": "eng"},
    ])
    out = add_relative_target(df, k=5)
    assert pd.isna(out.loc[out.post_id.isin(["a", "b", "c", "d", "e"]), "author_baseline"]).all()
    assert out.loc[out.post_id == "f", "author_baseline"].iloc[0] == 5.0
