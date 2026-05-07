# Step 03 - Baselines

Before training JEPA, prove that the 14D vectors contain useful fraud signal.

## Baseline 1 - Logistic Regression

Train directly on the original model input:

```txt
vector + missing_mask -> Logistic Regression -> fraud probability
```

This is a strong sanity check. If this performs poorly, JEPA may not rescue the
problem.

## Baseline 2 - Small MLP in MLX

Train a small supervised MLP:

```txt
vector + missing_mask -> MLP -> fraud probability
```

Suggested shape:

```txt
input_dim: 28
hidden_dim: 64
layers: 2
dropout: optional
loss: binary cross entropy
```

## Baseline 3 - KNN original vector

Compare against the existing vector-search intuition:

```txt
original 14D vector -> KNN -> fraud score
```

Try:

```txt
k = 5
k = 11
k = 31
```

## Metrics

Report at least:

```txt
auc_pr
roc_auc
precision
recall
f1
confusion_matrix
```

For fraud detection, AUC-PR and recall are more important than accuracy.

## Expected output

Write:

```txt
artifacts/metrics/baseline_logreg.json
artifacts/metrics/baseline_mlp.json
artifacts/metrics/baseline_knn_original.json
```

