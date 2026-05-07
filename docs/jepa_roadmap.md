# JEPA fraud roadmap

This is the implementation roadmap for the MLX-based JEPA-like fraud model.
The goal is to keep the work split into small steps so we can build, test, and
compare each idea without losing track of the experiment.

## Framework decision

This repository uses **MLX**, not PyTorch.

Any PyTorch examples used for learning should be translated to MLX before they
enter the codebase. The implementation should use:

```txt
mlx.core as mx
mlx.nn as nn
mlx.optimizers as optim
```

## Core idea

The dataset has records shaped like:

```json
{ "vector": [14 numbers], "label": "fraud" }
```

The JEPA pretraining stage does not use the label. It learns a latent
representation by predicting target embeddings for masked feature groups.

The fraud classification stage uses the labels through a `FraudHead`:

```txt
14D vector -> encoder -> latent embedding -> FraudHead -> fraud probability
```

## Step order

1. [Data contract](./jepa_step_01_data_contract.md)
2. [Project files and CLIs](./jepa_step_02_files_and_cli.md)
3. [Baselines](./jepa_step_03_baselines.md)
4. [JEPA architecture](./jepa_step_04_architecture.md)
5. [Mask curriculum](./jepa_step_05_mask_curriculum.md)
6. [JEPA pretraining](./jepa_step_06_pretraining.md)
7. [FraudHead training](./jepa_step_07_fraud_head.md)
8. [Joint training](./jepa_step_08_joint_training.md)
9. [Evaluation](./jepa_step_09_evaluation.md)
10. [Experiment checklist](./jepa_step_10_experiment_checklist.md)
11. [MLX implementation notes](./jepa_step_11_mlx_notes.md)

## Minimum success criteria

The JEPA encoder is useful only if it beats simple controls:

- Logistic Regression directly on the original 14D vector.
- Small MLP directly on the original 14D vector.
- Random frozen encoder plus FraudHead.
- KNN on the original 14D vector.

The main target metric should be AUC-PR because fraud is usually imbalanced.
Recall, precision, F1, and confusion matrix should also be reported.
