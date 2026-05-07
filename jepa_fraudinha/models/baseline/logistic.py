from sklearn.linear_model import LogisticRegression


def logistic() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

