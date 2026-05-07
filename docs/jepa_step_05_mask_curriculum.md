# Step 05 - Mask curriculum

The model should train in phases, starting easy and ending with mixed masks.

## Feature groups

Use semantic groups based on the 14D vector:

```txt
amount:
  [0, 1, 2, 13]

time:
  [3, 4]

history:
  [5, 6, 8]

location:
  [6, 7]

flags:
  [9, 10, 11]

risk:
  [12]
```

Groups may overlap. That is acceptable because masks are training strategies,
not schema definitions.

## Phase 1 - random light

Goal: stabilize the encoder.

```txt
mask 2 or 3 random features
```

## Phase 2 - semantic moderate

Goal: learn relationships between groups.

Examples:

```txt
mask time + part of history
mask history + location
```

Keep a small probability of phase 1 masks.

## Phase 3 - hard masks

Goal: force inference of high-signal features.

Examples:

```txt
mask amount + flags
mask amount + risk
mask history + flags
```

If the loss becomes unstable or the probe gets worse, reduce mask difficulty.

## Phase 4 - mixed final curriculum

Goal: avoid over-specialization.

Sampler probabilities:

```txt
30% random medium
25% semantic group
20% hard masks
15% random light
10% missing-aware masks
```

## Mask output

The mask sampler should return:

```txt
context_mask: 14 dims, 1 for visible features
target_mask:  14 dims, 1 for target features
strategy_name: string
```

Keep artificial JEPA masks separate from real missing masks.

