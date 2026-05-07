# Step 07 - FraudHead training

The FraudHead maps latent embeddings to fraud probability.

## Linear probe

First probe:

```txt
freeze encoder
train only linear FraudHead
```

Architecture:

```txt
embedding_dim -> 1 logit
```

Loss:

```txt
binary cross entropy with logits
```

This tells us whether the encoder learned a linearly useful latent space.

## MLP probe

If linear probe is weak but not hopeless, try:

```txt
embedding_dim -> hidden_dim -> 1 logit
```

Keep this small. A huge FraudHead can hide a weak encoder.

## Fine-tuning

After probing:

```txt
load pretrained encoder
attach FraudHead
unfreeze encoder
train supervised
```

Use a smaller learning rate for the encoder than the head if possible.

## Output artifacts

Write:

```txt
artifacts/checkpoints/fraud_head_linear.npz
artifacts/checkpoints/finetuned_encoder_head.npz
artifacts/metrics/probe_linear_phase_<n>.json
artifacts/metrics/finetune.json
```

