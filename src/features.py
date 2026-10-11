import re

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from .config import N_FEATURES


TEXT_VECTOR = HashingVectorizer(
    n_features=N_FEATURES,
    analyzer="word",
    ngram_range=(1, 2),
    alternate_sign=False,
    norm="l2",
    lowercase=True,
)


def metadata_dict(text, created_at, language):
    text = "" if text is None else str(text)

    return {
        "text_length": float(len(text)),
        "word_count": float(len(text.split())),
        "hashtag_count": float(len(re.findall(r"(?<!\w)#\w+", text))),
        "mention_count": float(len(re.findall(r"(?<!\w)@\w+", text))),
        "url_count": float(
            len(re.findall(r"https?://\S+|www\.\S+", text, flags=re.I))
        ),
        "question_count": float(text.count("?")),
        "exclamation_count": float(text.count("!")),
        "newline_count": float(text.count("\n")),
        "hour": float(created_at.hour),
        "weekday": float(created_at.weekday()),
        "month": float(created_at.month),
        "language": str(language or ""),
    }


def fit_metadata(rows):
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(
        [
            metadata_dict(r.text, r.created_at, r.language)
            for r in rows.itertuples()
        ]
    )
    return vec, X


def transform_metadata(vec, rows):
    return vec.transform(
        [
            metadata_dict(r.text, r.created_at, r.language)
            for r in rows.itertuples()
        ]
    )


def transform_text(rows):
    return TEXT_VECTOR.transform(
        rows["text"].fillna("").astype(str).tolist()
    )


def build_features(train_df, test_df, mode):
    X_text_train = transform_text(train_df)
    X_text_test = transform_text(test_df)

    meta_vec, X_meta_train = fit_metadata(train_df)
    X_meta_test = transform_metadata(meta_vec, test_df)

    if mode == "text_only":
        return X_text_train, X_text_test, meta_vec

    if mode == "metadata_only":
        return X_meta_train, X_meta_test, meta_vec

    if mode == "text_plus_metadata":
        return (
            hstack([X_text_train, X_meta_train], format="csr"),
            hstack([X_text_test, X_meta_test], format="csr"),
            meta_vec,
        )

    raise ValueError(f"Unknown feature mode: {mode}")
