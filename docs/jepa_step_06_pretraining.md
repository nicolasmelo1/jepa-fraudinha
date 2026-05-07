# Step 06 - JEPA pretraining

Pretraining learns structure from the vectors without using labels.

## Objective

Use masked latent prediction:

```txt
context input -> context_encoder -> predictor -> predicted target embedding
target input  -> target_encoder  -> target embedding
```

Loss:

```txt
loss_jepa = mse(predicted_target_embedding, stop_gradient(target_embedding))
```

Optional debugging auxiliary:

```txt
loss_total = loss_jepa + 0.1 * loss_reconstruction
```

Do not let reconstruction dominate the JEPA objective.

## Training phases

Run four phases:

```txt
phase 1: random light
phase 2: semantic moderate
phase 3: hard masks
phase 4: mixed curriculum
```

Each phase should save:

```txt
artifacts/checkpoints/jepa_phase_<n>.npz
artifacts/metrics/jepa_phase_<n>.json
```

## Metrics per epoch

Log:

```txt
train_loss_jepa
val_loss_jepa
embedding_std_mean
embedding_norm_mean
mask_strategy_counts
learning_rate
```

## Validation rule

After each phase, run a frozen linear probe. The JEPA loss alone is not enough
to decide whether the representation improved.

