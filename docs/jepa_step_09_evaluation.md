# Step 09 - Evaluation

Evaluation decides whether JEPA helped.

## Required comparisons

Compare these models:

```txt
1. Logistic Regression on original input
2. Small MLP on original input
3. KNN on original 14D vector
4. Random frozen encoder + FraudHead
5. JEPA frozen encoder + linear FraudHead
6. JEPA frozen encoder + MLP FraudHead
7. JEPA encoder + FraudHead fine-tuned
8. Joint JEPA + FraudHead
9. KNN on JEPA embeddings
```

## Main metrics

Use:

```txt
auc_pr
roc_auc
precision
recall
f1
confusion_matrix
```

Accuracy is secondary.

## Threshold selection

For reporting probabilities, evaluate multiple thresholds:

```txt
0.3
0.4
0.5
0.6
0.7
```

The challenge docs use `0.6` for KNN-style fraud score. For model probability,
we should still inspect the precision/recall tradeoff.

## KNN in latent space

Export embeddings:

```txt
artifacts/embeddings/train_embeddings.npz
artifacts/embeddings/val_embeddings.npz
artifacts/embeddings/test_embeddings.npz
```

Then compare:

```txt
KNN original 14D
KNN JEPA embedding
```

If KNN in JEPA space improves, the representation is useful for similarity
search too.

## Collapse checks

A good model should show non-trivial variance:

```txt
embedding_std_mean > near-zero
embedding_norm_mean stable
```

Also inspect whether fraud and legit examples are not mapped to the same latent
region.

