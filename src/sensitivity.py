import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import RAW_DIR, RESULTS_DIR
from .data import add_relative_target, load_all
from .features import build_features
from .metrics import calculate_metrics
from .models import make_regressor


def run_k(df, k):
    work = add_relative_target(df.copy(), k=k).dropna(subset=["y_rel"])
    splitter = GroupKFold(n_splits=3)
    rows = []

    for train_idx, test_idx in splitter.split(
        work, work["y_rel"], groups=work["author_id"]
    ):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]

        baseline = np.full(
            len(test), np.median(train["y_rel"].to_numpy(dtype=float))
        )
        rows.append({
            "model": "median_baseline",
            **calculate_metrics(test["y_rel"], baseline),
        })

        X_train, X_test, _ = build_features(
            train, test, "metadata_only"
        )
        model = make_regressor()
        model.fit(X_train, train["y_rel"])
        pred = model.predict(X_test)

        rows.append({
            "model": "metadata_only",
            **calculate_metrics(test["y_rel"], pred),
        })

    return (
        pd.DataFrame(rows)
        .groupby("model")
        .mean(numeric_only=True)
        .reset_index()
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all(RAW_DIR)

    for k in (3, 5, 10):
        out = run_k(df, k)
        out.insert(0, "K", k)
        out.to_csv(
            RESULTS_DIR / f"relative_baseline_sensitivity_k{k}.csv",
            index=False,
        )
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
