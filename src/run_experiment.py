import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import K_PRIMARY, N_SPLITS, RAW_DIR, RESULTS_DIR
from .data import add_relative_target, load_all
from .features import build_features
from .metrics import calculate_metrics
from .models import make_regressor


MODES = ["text_only", "metadata_only", "text_plus_metadata"]


def run_target(df, target_name):
    work = df.dropna(subset=[target_name]).copy()
    groups = work["author_id"].to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)

    rows = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(work, work[target_name], groups=groups), start=1
    ):
        train = work.iloc[train_idx].copy()
        test = work.iloc[test_idx].copy()

        overlap = set(train["author_id"]) & set(test["author_id"])
        assert not overlap, f"Author leakage in fold {fold}"

        baseline_pred = np.full(
            len(test), np.median(train[target_name].to_numpy(dtype=float))
        )
        rows.append({
            "target": target_name,
            "model": "median_baseline",
            "fold": fold,
            "n_test": len(test),
            "author_overlap": len(overlap),
            **calculate_metrics(test[target_name], baseline_pred),
        })

        for mode in MODES:
            X_train, X_test, _ = build_features(train, test, mode)
            model = make_regressor()
            model.fit(X_train, train[target_name].to_numpy(dtype=float))
            pred = model.predict(X_test)

            rows.append({
                "target": target_name,
                "model": mode,
                "fold": fold,
                "n_test": len(test),
                "author_overlap": len(overlap),
                **calculate_metrics(test[target_name], pred),
            })

            oof_rows.extend(
                {
                    "target": target_name,
                    "model": mode,
                    "fold": fold,
                    "post_id": post_id,
                    "author_id": author_id,
                    "y_true": truth,
                    "y_pred": prediction,
                }
                for post_id, author_id, truth, prediction in zip(
                    test["post_id"],
                    test["author_id"],
                    test[target_name],
                    pred,
                )
            )

        oof_rows.extend(
            {
                "target": target_name,
                "model": "median_baseline",
                "fold": fold,
                "post_id": post_id,
                "author_id": author_id,
                "y_true": truth,
                "y_pred": prediction,
            }
            for post_id, author_id, truth, prediction in zip(
                test["post_id"],
                test["author_id"],
                test[target_name],
                baseline_pred,
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def summarize(fold_results):
    return (
        fold_results.groupby(["target", "model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            Spearman_mean=("Spearman", "mean"),
            Spearman_sd=("Spearman", "std"),
            author_overlap_max=("author_overlap", "max"),
            n_test_min=("n_test", "min"),
            n_test_max=("n_test", "max"),
        )
        .reset_index()
    )


def cluster_bootstrap(oof, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for target in oof["target"].unique():
        sub = oof[oof["target"] == target]
        base = sub[sub["model"] == "median_baseline"].set_index("post_id")

        for model_name in MODES:
            pred = sub[sub["model"] == model_name].set_index("post_id")
            common = base.index.intersection(pred.index)

            joined = pd.DataFrame({
                "author_id": pred.loc[common, "author_id"].values,
                "base_mae": np.abs(
                    base.loc[common, "y_true"] - base.loc[common, "y_pred"]
                ).values,
                "model_mae": np.abs(
                    pred.loc[common, "y_true"] - pred.loc[common, "y_pred"]
                ).values,
            })

            authors = joined["author_id"].unique()
            diffs = []

            for _ in range(n_boot):
                sampled = rng.choice(
                    authors, size=len(authors), replace=True
                )
                boot = pd.concat(
                    [joined[joined["author_id"] == a] for a in sampled],
                    ignore_index=True,
                )
                diffs.append(
                    boot["model_mae"].mean() - boot["base_mae"].mean()
                )

            rows.append({
                "target": target,
                "model": model_name,
                "MAE_difference": float(
                    joined["model_mae"].mean() - joined["base_mae"].mean()
                ),
                "CI95_low": float(np.quantile(diffs, 0.025)),
                "CI95_high": float(np.quantile(diffs, 0.975)),
            })

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = add_relative_target(
        load_all(RAW_DIR), k=K_PRIMARY
    )

    print(f"Rows: {len(df):,}")
    print(f"Authors: {df['author_id'].nunique():,}")
    print(f"Relative eligible: {df['y_rel'].notna().sum():,}")

    fold_frames = []
    oof_frames = []

    for target in ("y_abs", "y_rel"):
        fold_df, oof_df = run_target(df, target)
        fold_frames.append(fold_df)
        oof_frames.append(oof_df)

    fold_results = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)

    fold_results.to_csv(RESULTS_DIR / "fold_results.csv", index=False)
    summarize(fold_results).to_csv(
        RESULTS_DIR / "summary.csv", index=False
    )
    oof.to_csv(
        RESULTS_DIR / "oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    cluster_bootstrap(oof).to_csv(
        RESULTS_DIR / "author_cluster_bootstrap_mae.csv",
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(summarize(fold_results).to_string(index=False))


if __name__ == "__main__":
    main()
