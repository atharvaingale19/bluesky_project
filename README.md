# Author-Relative Engagement

Research repository for the exploratory study **Beyond Absolute Popularity: Author-Relative Engagement Estimation under Author-Disjoint Evaluation**.

## Dataset location

The app and experiment now expect the selected thematic-feed files under:

```text
data/feed_posts/
```

The loader searches recursively, so this also works:

```text
data/feed_posts/feed_posts/*.jsonl.gz
```

The selected files are the 11 thematic feed outputs used in the study.

### If you have `feed_posts.tar.gz`

Place it at:

```text
data/source/feed_posts.tar.gz
```

and run:

```powershell
python scripts/prepare_data.py data/source/feed_posts.tar.gz
```

This extracts the 11 selected files into `data/feed_posts/` and creates `data/MANIFEST.csv` with SHA-256 checksums and parsed row counts.

If you already copied the complete `feed_posts` folder into `data/feed_posts/`, you do **not** need to extract anything again.

## Important schema detail

The supplied Bluesky Social Dataset feed files use fields including:

- `post_id`
- `user_id`
- `date` in `YYYYMMDDHHMM` format
- `text`
- `langs`
- `repost_count`

`src/data.py` explicitly normalizes this schema. It is not assuming generic `author_id` and ISO timestamps.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verify the dataset

```powershell
python scripts/verify_dataset.py
```

## Run tests

```powershell
pytest -q
```

## Run the research experiment

```powershell
python -m src.run_experiment
```

Results are written to `results/`.

## Run the Streamlit dashboard

```powershell
streamlit run app.py
```

The dashboard contains:

1. **Overview**: study definition and dataset-level KPIs.
2. **Dataset Explorer**: feed-level counts, repost distribution, and sample records.
3. **Research Results**: displays computed author-disjoint results when `results/summary.csv` exists.
4. **Interactive Demo**: full-data metadata-only demonstration. It is explicitly separated from the paper evaluation.

## Research caution

The target is retrospective repost engagement. Engagement counters are dataset snapshots rather than a fixed-horizon future-virality label. The dataset is also a selected thematic-feed sample rather than a representative sample of all Bluesky posts.
