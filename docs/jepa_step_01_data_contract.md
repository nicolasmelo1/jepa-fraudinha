# Step 01 - Data contract

## Input file

Use:

```txt
data/references.json
```

Each item must contain:

```json
{
  "vector": [0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416],
  "label": "legit"
}
```

## Feature order

The vector has 14 dimensions, matching [detection_rules.md](./detection_rules.md):

| index | feature |
|---:|---|
| 0 | amount |
| 1 | installments |
| 2 | amount_vs_avg |
| 3 | hour_of_day |
| 4 | day_of_week |
| 5 | minutes_since_last_tx |
| 6 | km_from_last_tx |
| 7 | km_from_home |
| 8 | tx_count_24h |
| 9 | is_online |
| 10 | card_present |
| 11 | unknown_merchant |
| 12 | mcc_risk |
| 13 | merchant_avg_amount |

## Label encoding

Use:

```txt
legit -> 0
fraud -> 1
```

## Missing values

Indices 5 and 6 can be `-1`. In the challenge docs this means no previous
transaction exists.

For model training, do not treat `-1` as only a numeric value. Create an
explicit missing indicator:

```txt
raw vector:      14 dims
missing mask:    14 dims, 1 where value is -1, else 0
model input:     either 28 dims for MLP or feature tokens with value + missing flag
```

Important distinction:

- real missing data: comes from `-1` in the dataset
- JEPA mask: artificial training mask used by the pretext task

These two masks must stay separate.

## Dataset splits

Use stratified splits:

```txt
train: 70%
val:   15%
test:  15%
```

The test split should be touched only for final comparison.

## Output artifacts

The data loading step should produce:

```txt
artifacts/data/train.npz
artifacts/data/val.npz
artifacts/data/test.npz
artifacts/data/stats.json
```

Each `.npz` should contain:

```txt
vectors
missing_mask
labels
```

