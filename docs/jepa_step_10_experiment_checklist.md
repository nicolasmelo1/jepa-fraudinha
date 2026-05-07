# Step 10 - Experiment checklist

Use this checklist while implementing.

## Data

- [ ] Load `data/references.json`.
- [ ] Validate every vector has 14 dimensions.
- [ ] Encode labels as `legit=0`, `fraud=1`.
- [ ] Create real missing mask from `-1`.
- [ ] Create stratified train/val/test split.
- [ ] Save prepared `.npz` files.

## Baselines

- [ ] Train Logistic Regression.
- [ ] Train supervised MLX MLP.
- [ ] Evaluate KNN on original 14D vector.
- [ ] Save metrics JSON for each baseline.

## JEPA

- [ ] Implement MLP encoder.
- [ ] Implement target encoder with EMA.
- [ ] Implement predictor.
- [ ] Implement mask sampler.
- [ ] Implement collapse metrics.
- [ ] Train phase 1.
- [ ] Train phase 2.
- [ ] Train phase 3.
- [ ] Train phase 4.

## FraudHead

- [ ] Train linear probe after each JEPA phase.
- [ ] Train MLP probe if useful.
- [ ] Fine-tune encoder + FraudHead.
- [ ] Compare against random frozen encoder.

## Joint training

- [ ] Implement combined JEPA + BCE loss.
- [ ] Run `lambda_fraud=0.1`.
- [ ] Compare with pure pretrain + fine-tune.

## Evaluation

- [ ] Export embeddings.
- [ ] Run KNN on JEPA embeddings.
- [ ] Compare all models on the same test split.
- [ ] Save final metrics table.
- [ ] Plot loss and AUC-PR curves.

