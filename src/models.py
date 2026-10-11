from sklearn.linear_model import SGDRegressor


def make_regressor():
    return SGDRegressor(
        loss="huber",
        alpha=1e-4,
        max_iter=15,
        random_state=42,
        tol=1e-3,
    )
