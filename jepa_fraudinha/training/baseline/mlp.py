import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.metrics import (
  average_precision_score,
  confusion_matrix,
  precision_recall_fscore_support,
  roc_auc_score,
)

from jepa_fraudinha.models.baseline.mlp import MLP


def load_split(data_folder: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_folder / f"{split}.npz")

    vectors = data["arr_0"]
    missing_mask = data["arr_1"]
    labels = data["arr_2"]
    idx = data["arr_3"]

    x = np.concatenate(
        [vectors[idx], missing_mask[idx]],
        axis=1,
    ).astype(np.float32)

    y = labels[idx].astype(np.float32)

    return x, y


def compute_metrics(
  y_true: np.ndarray,
  scores: np.ndarray,
  threshold: float = 0.5,
) -> dict:
    preds = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.astype(int),
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
        "confusion_matrix": confusion_matrix(y_true.astype(int), preds).tolist(),
    }


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    indices = rng.permutation(len(y))

    for start in range(0, len(y), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def predict_scores(
      model: MLP,
      x: np.ndarray,
      batch_size: int,
  ) -> np.ndarray:
    scores = []

    model.eval()

    for start in range(0, len(x), batch_size):
        batch_x = mx.array(x[start:start + batch_size])
        logits = model(batch_x)
        batch_scores = mx.sigmoid(logits)
        mx.eval(batch_scores)
        scores.append(np.array(batch_scores))

    return np.concatenate(scores, axis=0)


def train_mlp(args: argparse.Namespace) -> None:
    data_folder = Path(args.data_folder)
    metrics_folder = Path(args.metrics_folder)
    metrics_folder.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(data_folder, "train")
    x_val, y_val = load_split(data_folder, "val")
    x_test, y_test = load_split(data_folder, "test")

    model = MLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
    )

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    positive_weight = negative_count / positive_count

    def loss_fn(model: MLP, batch_x: mx.array, batch_y: mx.array) -> mx.array:
        logits = model(batch_x)

        weights = mx.where(
            batch_y == 1,
            positive_weight,
            1.0,
        )

        return nn.losses.binary_cross_entropy(
            logits,
            batch_y,
            weights=weights,
            with_logits=True,
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(args.epochs):
        model.train()
        losses = []

    for batch_x_np, batch_y_np in iter_batches(
        x_train,
        y_train,
        args.batch_size,
        rng,
    ):
        batch_x = mx.array(batch_x_np)
        batch_y = mx.array(batch_y_np)

        loss, grads = loss_and_grad(model, batch_x, batch_y)
        optimizer.update(model, grads)

        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss))

    val_scores = predict_scores(model, x_val, args.batch_size)
    val_metrics = compute_metrics(y_val, val_scores)

    print(
        f"epoch={epoch + 1} "
        f"loss={np.mean(losses):.6f} "
        f"val_auc_pr={val_metrics['auc_pr']:.6f} "
        f"val_f1={val_metrics['f1']:.6f}"
    )

    val_scores = predict_scores(model, x_val, args.batch_size)
    test_scores = predict_scores(model, x_test, args.batch_size)

    metrics = {
      "model": "baseline_mlp",
      "features": "vector_plus_missing_mask",
      "input_dim": int(x_train.shape[1]),
      "hidden_dim": int(args.hidden_dim),
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "learning_rate": float(args.learning_rate),
      "train_size": int(len(y_train)),
      "val_size": int(len(y_val)),
      "test_size": int(len(y_test)),
      "val": compute_metrics(y_val, val_scores),
      "test": compute_metrics(y_test, test_scores),
    }

    metrics_path = metrics_folder / "baseline_mlp.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {metrics_path}")

