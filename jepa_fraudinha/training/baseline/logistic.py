import argparse
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from jepa_fraudinha.models.baseline.logistic import logistic


def load_split(data_folder: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_folder / f"{split}.npz")

    vectors = data["arr_0"]
    missing_mask = data["arr_1"]
    labels = data["arr_2"]
    idx = data["arr_3"]

    x = np.concatenate(
        [vectors[idx], missing_mask[idx]],
        axis=1,
    )
    y = labels[idx].astype(int)

    return x, y


def compute_metrics(
      y_true: np.ndarray,
      scores: np.ndarray,
      threshold: float = 0.5,
  ) -> dict:
    preds = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "auc_pr": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def train_logistic(args: argparse.Namespace) -> None:
    data_folder = Path(args.data_folder)
    metrics_folder = Path(args.metrics_folder)
    metrics_folder.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(data_folder, "train")
    x_val, y_val = load_split(data_folder, "val")
    x_test, y_test = load_split(data_folder, "test")

    model = logistic()
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    metrics = {
        "model": "logistic_regression",
        "features": "vector_plus_missing_mask",
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "val": compute_metrics(y_val, val_scores),
        "test": compute_metrics(y_test, test_scores),
    }

    metrics_path = metrics_folder / "baseline_logreg.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {metrics_path}")
