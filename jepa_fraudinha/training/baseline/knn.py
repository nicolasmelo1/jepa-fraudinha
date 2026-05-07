import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors


def load_split(data_folder: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_folder / f"{split}.npz")

    vectors = data["arr_0"]
    labels = data["arr_2"]
    idx = data["arr_3"]

    x = vectors[idx].astype(np.float32)
    y = labels[idx].astype(int)

    return x, y


def sample_rows(
    x: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0 or sample_size >= len(y):
        return x, y

    sampled_idx = rng.choice(len(y), size=sample_size, replace=False)
    return x[sampled_idx], y[sampled_idx]


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


def score_knn_all(
    neighbors: NearestNeighbors,
    y_reference: np.ndarray,
    x_eval: np.ndarray,
    ks: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    scores_by_k: dict[int, list[np.ndarray]] = {k: [] for k in ks}
    max_k = max(ks)

    for start in range(0, len(x_eval), batch_size):
        batch_x = x_eval[start:start + batch_size]
        neighbor_idx = neighbors.kneighbors(
            batch_x,
            n_neighbors=max_k,
            return_distance=False,
        )
        neighbor_labels = y_reference[neighbor_idx]

        for k in ks:
            scores_by_k[k].append(neighbor_labels[:, :k].mean(axis=1))

    return {
        k: np.concatenate(scores, axis=0)
        for k, scores in scores_by_k.items()
    }


def train_knn(args: argparse.Namespace) -> None:
    data_folder = Path(args.data_folder)
    metrics_folder = Path(args.metrics_folder)
    metrics_folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ks = sorted(set(args.ks))
    max_k = max(ks)

    x_train, y_train = load_split(data_folder, "train")
    x_val, y_val = load_split(data_folder, "val")
    x_test, y_test = load_split(data_folder, "test")

    x_train, y_train = sample_rows(x_train, y_train, args.train_sample, rng)
    x_val, y_val = sample_rows(x_val, y_val, args.eval_sample, rng)
    x_test, y_test = sample_rows(x_test, y_test, args.eval_sample, rng)

    if len(y_train) < max_k:
        raise ValueError(
            f"train sample has {len(y_train)} rows, but max k is {max_k}"
        )

    neighbors = NearestNeighbors(
        n_neighbors=max_k,
        metric=args.metric,
        n_jobs=-1,
    )
    neighbors.fit(x_train)

    metrics = {
        "model": "knn",
        "features": "vector_only",
        "metric": args.metric,
        "ks": ks,
        "train_reference_size": int(len(y_train)),
        "val_eval_size": int(len(y_val)),
        "test_eval_size": int(len(y_test)),
        "val": {},
        "test": {},
    }

    val_scores_by_k = score_knn_all(
        neighbors,
        y_train,
        x_val,
        ks,
        args.batch_size,
    )
    test_scores_by_k = score_knn_all(
        neighbors,
        y_train,
        x_test,
        ks,
        args.batch_size,
    )

    for k in ks:
        val_scores = val_scores_by_k[k]
        test_scores = test_scores_by_k[k]

        metrics["val"][f"k_{k}"] = compute_metrics(y_val, val_scores)
        metrics["test"][f"k_{k}"] = compute_metrics(y_test, test_scores)

        print(
            f"k={k} "
            f"val_auc_pr={metrics['val'][f'k_{k}']['auc_pr']:.6f} "
            f"test_auc_pr={metrics['test'][f'k_{k}']['auc_pr']:.6f}"
        )

    metrics_path = metrics_folder / "baseline_knn.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {metrics_path}")
