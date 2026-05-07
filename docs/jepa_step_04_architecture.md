# Step 04 - JEPA architecture

The first implementation should be simple and debuggable.

## Framework

Use MLX only:

```python
import mlx.core as mx
import mlx.nn as nn
```

## Version A - MLP encoder

Start with a small MLP encoder:

```txt
input:  vector + missing_mask
output: latent embedding
```

Suggested dimensions:

```txt
input_dim: 28
hidden_dim: 128
embedding_dim: 64
```

This version is easier to debug and should be implemented first.

Minimal shape in MLX:

```python
class TabularEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def __call__(self, x):
        z = self.net(x)
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
```

## Version B - Feature tokenizer

After the MLP version works, add feature tokens:

```txt
feature value + missing flag + feature id embedding -> feature token
```

Then aggregate the tokens:

```txt
14 feature tokens -> encoder -> transaction embedding
```

This is closer to JEPA-style thinking because each feature has identity.

## JEPA components

Use these modules:

```txt
context_encoder
target_encoder
predictor
```

The target encoder should be an EMA copy of the context encoder:

```txt
target = tau * target + (1 - tau) * context
```

Suggested:

```txt
tau = 0.99 initially
tau = 0.999 for more stable longer runs
```

## Stop gradient

The target embedding must not receive gradient from the JEPA loss:

```txt
loss = mse(predicted_target_embedding, stop_gradient(target_embedding))
```

## Collapse monitoring

During training, log:

```txt
embedding_std_mean
embedding_std_min
embedding_norm_mean
```

If `embedding_std_mean` goes close to zero, the encoder may have collapsed.
