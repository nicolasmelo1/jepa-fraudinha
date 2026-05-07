# Step 02 - Project files and CLI plan

This is the planned repository shape for the training code.

## Package layout

Create a Python package:

```txt
jepa_fraudinha/
  __init__.py
  cli.py
  data.py
  metrics.py
  masks.py
  models/
    __init__.py
    encoder.py
    jepa.py
    fraud_head.py
  training/
    __init__.py
    pretrain.py
    probe.py
    finetune.py
    joint.py
  eval/
    __init__.py
    knn.py
    embeddings.py
```

## Config layout

Use YAML configs for reproducible experiments:

```txt
configs/
  data.yaml
  pretrain_phase1.yaml
  pretrain_phase2.yaml
  pretrain_phase3.yaml
  pretrain_phase4.yaml
  probe.yaml
  finetune.yaml
  joint.yaml
```

## Artifact layout

Generated files should go under:

```txt
artifacts/
  data/
  checkpoints/
  metrics/
  embeddings/
  plots/
```

Do not commit large generated artifacts unless we explicitly decide to.

## CLI commands

The CLI should expose these commands:

```bash
uv run python -m jepa_fraudinha.cli prepare-data --input data/references.json
uv run python -m jepa_fraudinha.cli baseline --config configs/data.yaml
uv run python -m jepa_fraudinha.cli pretrain --config configs/pretrain_phase1.yaml
uv run python -m jepa_fraudinha.cli probe --config configs/probe.yaml
uv run python -m jepa_fraudinha.cli finetune --config configs/finetune.yaml
uv run python -m jepa_fraudinha.cli joint --config configs/joint.yaml
uv run python -m jepa_fraudinha.cli eval --checkpoint artifacts/checkpoints/model.safetensors
```

MLX checkpoint format can be `.npz` initially. A custom format is fine as long
as load/save is centralized in one module.

## Implementation rule

Every command should write a JSON metrics file:

```txt
artifacts/metrics/<run_name>.json
```

This keeps experiment comparison simple.

## Framework rule

All model code must be MLX-based:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
```

Do not add PyTorch modules, dataloaders, tensors, or checkpoint conventions.
