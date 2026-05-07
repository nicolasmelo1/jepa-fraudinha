# Step 08 - Joint training

Joint training uses both JEPA and fraud classification losses.

This is no longer pure JEPA, but it can create a latent space that is useful for
both structure learning and fraud separation.

## Model flow

```txt
vector
  -> encoder
    -> JEPA predictor
    -> FraudHead
```

## Loss

Use:

```txt
loss_total = loss_jepa + lambda_fraud * loss_fraud
```

Start with:

```txt
lambda_fraud = 0.1
```

Then compare:

```txt
lambda_fraud = 0.05
lambda_fraud = 0.1
lambda_fraud = 0.25
lambda_fraud = 0.5
```

## When to use this

Run joint training only after:

- baselines exist
- pure JEPA pretraining runs
- linear probe works
- fine-tuning pipeline works

Otherwise it becomes hard to know what caused the result.

## Metrics

Log both objectives:

```txt
train_loss_jepa
train_loss_fraud
train_loss_total
val_loss_jepa
val_loss_fraud
auc_pr
precision
recall
embedding_std_mean
```

