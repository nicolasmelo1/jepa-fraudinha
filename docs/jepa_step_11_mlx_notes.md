# Step 11 - MLX implementation notes

This project uses MLX. PyTorch examples are useful conceptually, but the repo
implementation should stay MLX-only.

## Imports

Use:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
```

Use NumPy only for loading, splitting, and CPU-side preprocessing:

```python
import numpy as np
```

## PyTorch to MLX translation

| PyTorch concept | MLX equivalent |
|---|---|
| `torch.Tensor` | `mx.array` |
| `nn.Module.forward` | `nn.Module.__call__` |
| `torch.no_grad()` | avoid gradient transform around target path |
| `optimizer.zero_grad()` | not used the same way in MLX |
| `loss.backward()` | `nn.value_and_grad(model, loss_fn)` |
| `optimizer.step()` | `optimizer.update(model, grads)` |
| `F.normalize` | manual norm with `mx.linalg.norm` |
| `F.mse_loss` | `mx.mean(mx.square(a - b))` |
| `binary_cross_entropy_with_logits` | implement with stable logits formula |

## Minimal encoder

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
        norm = mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        return z / norm
```

## Minimal predictor

```python
class Predictor(nn.Module):
    def __init__(self, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def __call__(self, z):
        z = self.net(z)
        norm = mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        return z / norm
```

## JEPA loss

```python
def mse_loss(pred, target):
    return mx.mean(mx.square(pred - target))
```

Target embeddings should be computed without updating the target encoder by
gradient. The target encoder is updated only by EMA.

## EMA target encoder

The target encoder starts as a copy of the context encoder and then tracks it:

```txt
target = tau * target + (1 - tau) * context
```

Suggested values:

```txt
tau = 0.99
tau = 0.996
tau = 0.999
```

Keep the EMA update in one helper function so it is easy to audit.

## Stable BCE with logits

For FraudHead training:

```python
def bce_with_logits(logits, labels):
    return mx.mean(mx.maximum(logits, 0) - logits * labels + mx.log1p(mx.exp(-mx.abs(logits))))
```

Labels should be shaped like:

```txt
batch_size x 1
```

with:

```txt
legit = 0.0
fraud = 1.0
```

## Missing mask input

For the first MLP version, concatenate:

```python
model_input = mx.concatenate([vectors_without_minus_one, missing_mask], axis=-1)
```

Replace real `-1` values with `0` in `vectors_without_minus_one`, but preserve
that information in `missing_mask`.

## Artificial JEPA masks

The JEPA mask is separate from the real missing mask:

```txt
real_missing_mask: comes from the dataset
jepa_context_mask: generated during pretraining
jepa_target_mask: generated during pretraining
```

For the simple MLP JEPA version, masked context features can be zeroed before
concatenating with the missing mask.

## First implementation target

The first working model should be:

```txt
MLP context encoder
MLP target encoder with EMA
MLP predictor
random 30% feature masking
linear FraudHead probe
```

After that works, move to feature tokens and semantic masks.

