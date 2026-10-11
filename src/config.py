from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The user-facing repo stores the selected thematic-feed dataset here.
# A legacy data/raw/ fallback is handled by get_data_dirs().
DATA_DIR = ROOT / "data" / "feed_posts"
LEGACY_RAW_DIR = ROOT / "data" / "raw"
RAW_DIR = DATA_DIR
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
N_SPLITS = 3
N_FEATURES = 2**18
K_PRIMARY = 5

EXPECTED_FEEDS = [
    "Science.jsonl.gz",
    "What's History.jsonl.gz",
    "News.jsonl.gz",
    "Political Science.jsonl.gz",
    "Blacksky.jsonl.gz",
    "BookSky.jsonl.gz",
    "Game Dev.jsonl.gz",
    "GreenSky.jsonl.gz",
    "#Disability.jsonl.gz",
    "#UkrainianView.jsonl.gz",
    "AcademicSky.jsonl.gz",
]


def get_data_dirs():
    """Return dataset directories that exist, preferring data/feed_posts."""
    dirs = []
    if DATA_DIR.exists():
        dirs.append(DATA_DIR)
    if LEGACY_RAW_DIR.exists() and LEGACY_RAW_DIR not in dirs:
        dirs.append(LEGACY_RAW_DIR)
    return dirs
