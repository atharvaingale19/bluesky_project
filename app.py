from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DATA_DIR, EXPECTED_FEEDS, LEGACY_RAW_DIR, RESULTS_DIR
from src.data import add_relative_target, discover_files, load_all
from src.features import fit_metadata, transform_metadata
from src.models import make_regressor

st.set_page_config(
    page_title="Author-Relative Engagement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(14,165,233,.08));
        border: 1px solid rgba(148,163,184,.18);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 .25rem 0; }
    .muted { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


def data_signature(data_dir: Path):
    files = discover_files(data_dir) if data_dir.exists() else []
    return tuple((str(p.relative_to(data_dir)), p.stat().st_size, p.stat().st_mtime_ns) for p in files)


def active_data_dir():
    if DATA_DIR.exists() and discover_files(DATA_DIR):
        return DATA_DIR
    if LEGACY_RAW_DIR.exists() and discover_files(LEGACY_RAW_DIR):
        return LEGACY_RAW_DIR
    return DATA_DIR


DATA_ROOT = active_data_dir()
FILES = discover_files(DATA_ROOT) if DATA_ROOT.exists() else []


@st.cache_data
def load_dataset_cache(data_root, signature):
    df = load_all(data_root)
    return df

@st.cache_data(show_spinner=False)
def add_targets_cached(df, k=5):
    return add_relative_target(df, k=k)


@st.cache_resource(show_spinner=False)
@st.cache_resource
def train_demo_model(df: pd.DataFrame, data_root: str):
    df = add_relative_target(df, k=5)

    vec, X = fit_metadata(df)

    X = X.tocsr()
    X.indices = X.indices.astype(np.int32, copy=False)
    X.indptr = X.indptr.astype(np.int32, copy=False)

    model = make_regressor()
    model.fit(X, df["y_abs"].to_numpy(dtype=float))

    return vec, model


st.markdown(
    '<div class="hero"><h1>Author-Relative Engagement</h1>'
    '<div class="muted">Beyond absolute popularity: an author-disjoint study of retrospective repost engagement.</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset status")
    st.metric("Feed files detected", len(FILES))
    st.caption(f"Expected: {len(EXPECTED_FEEDS)} selected thematic feeds")

    if len(FILES) == len(EXPECTED_FEEDS):
        st.success("All selected feed files detected")
    elif FILES:
        st.warning(f"Found {len(FILES)} feed files. Expected {len(EXPECTED_FEEDS)}.")
    else:
        st.error("No .jsonl.gz files detected")

    st.caption(f"Data path: `{DATA_ROOT.relative_to(DATA_ROOT.parents[1]) if DATA_ROOT.exists() else DATA_ROOT}`")
    st.divider()
    st.markdown(
        "**Important:** paper metrics come from the author-disjoint experiment. "
        "The interactive predictor is a demonstration, not a validation result."
    )

if not FILES:
    st.info(
        "Place the extracted thematic-feed `.jsonl.gz` files under `data/feed_posts/`. "
        "The app recursively discovers them, so an extra nested folder is okay."
    )
    st.stop()

with st.spinner("Reading the dataset for the dashboard..."):
    sig = data_signature(DATA_ROOT)
    df_raw = load_dataset_cache(str(DATA_ROOT), sig)
    df = add_targets_cached(df_raw, 5)

# ---- Overview ----
tab_overview, tab_explorer, tab_results, tab_demo = st.tabs(
    ["Overview", "Dataset Explorer", "Research Results", "Interactive Demo"]
)

with tab_overview:
    st.subheader("Study at a glance")
    eligible = int(df["y_rel"].notna().sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique posts", f"{len(df):,}")
    c2.metric("Authors", f"{df['author_id'].nunique():,}")
    c3.metric("Relative eligible", f"{eligible:,}")
    c4.metric("Median reposts", f"{df['repost_count'].median():.0f}")
    c5.metric("Feed files", len(FILES))

    st.markdown("### What this project tests")
    st.write(
        "Absolute repost counts are strongly tied to an author's existing reach. "
        "The study therefore compares absolute engagement with a retrospective "
        "author-relative target: current log(1 + reposts) minus the log(1 + median) "
        "of the author's five most recent strictly earlier posts."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Dataset coverage")
        st.write(f"**Earliest:** {df['created_at'].min():%Y-%m-%d %H:%M UTC}")
        st.write(f"**Latest:** {df['created_at'].max():%Y-%m-%d %H:%M UTC}")
        st.write("**Evaluation:** 3-fold GroupKFold by author")
    with right:
        st.markdown("#### Target definitions")
        st.code("y_abs = log1p(current_repost_count)")
        st.code("y_rel = log1p(current_reposts) - log1p(author_baseline)")

with tab_explorer:
    st.subheader("Dataset Explorer")
    st.caption("These are descriptive statistics from the local feed files, not model evaluation metrics.")

    feed_stats = (
        df.groupby("source_file")
        .agg(
            posts=("post_id", "count"),
            authors=("author_id", "nunique"),
            median_reposts=("repost_count", "median"),
            mean_reposts=("repost_count", "mean"),
            relative_eligible=("y_rel", lambda x: int(x.notna().sum())),
        )
        .sort_values("posts", ascending=False)
        .reset_index()
    )
    st.dataframe(feed_stats, use_container_width=True, hide_index=True)

    st.markdown("#### Repost distribution")
    hist = df["repost_count"].clip(upper=int(df["repost_count"].quantile(.99)))
    st.bar_chart(hist.value_counts().sort_index().head(50), height=300)

    st.markdown("#### Sample records")
    cols = ["source_file", "created_at", "language", "repost_count", "author_baseline", "y_abs", "y_rel", "text"]
    st.dataframe(df[cols].head(100), use_container_width=True, hide_index=True)

with tab_results:
    st.subheader("Author-disjoint evaluation")
    summary_path = RESULTS_DIR / "summary.csv"
    fold_path = RESULTS_DIR / "fold_results.csv"

    if not summary_path.exists():
        st.info(
            "No computed experiment results are present yet. Run "
            "`python -m src.run_experiment` from the repository root. "
            "The dashboard will automatically display the resulting summary here."
        )
    else:
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if {"target", "model", "MAE_mean"}.issubset(summary.columns):
            plot = summary.pivot(index="model", columns="target", values="MAE_mean")
            st.markdown("#### Mean absolute error")
            st.bar_chart(plot, height=350)

        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            st.markdown("#### Fold-level results")
            st.dataframe(folds, use_container_width=True, hide_index=True)

with tab_demo:
    st.subheader("Interactive engagement estimate")
    st.warning(
        "Demonstration only. This model is trained on the full local dataset and "
        "does not reproduce the paper's author-disjoint evaluation."
    )

    text = st.text_area(
        "Post text",
        "A surprisingly useful observation about machine learning and social media.",
        height=130,
    )
    c1, c2, c3 = st.columns(3)
    language = c1.text_input("Language", "eng")
    hour = c2.slider("Posting hour", 0, 23, 18)
    weekday = c3.slider("Weekday", 0, 6, 5)

    st.markdown("#### Five observed prior repost counts")
    prior_cols = st.columns(5)
    prior = [
        prior_cols[i].number_input(f"Post {i + 1}", min_value=0, value=0, step=1)
        for i in range(5)
    ]

    if st.button("Estimate engagement", type="primary"):
        with st.spinner("Training the demonstration model…"):
            vec, model = train_demo_model(df_raw, str(DATA_ROOT))
            
        user_time = pd.Timestamp.now(tz="UTC").replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        user_row = pd.DataFrame([{
            "text": text,
            "created_at": user_time,
            "language": language,
        }])
        X_user = transform_metadata(vec, user_row)
        pred_log = float(model.predict(X_user)[0])
        pred_reposts = max(0.0, float(np.expm1(pred_log)))
        historical_baseline = float(np.median(prior))
        relative_score = pred_log - np.log1p(historical_baseline)

        a, b, c = st.columns(3)
        a.metric("Estimated log(1 + reposts)", f"{pred_log:.3f}")
        b.metric("Estimated reposts", f"{pred_reposts:.1f}")
        c.metric("Author-relative score", f"{relative_score:.3f}")

        st.caption(
            "Relative score = estimated log(1 + reposts) − log(1 + median of the five prior repost counts)."
        )
