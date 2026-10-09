# Development log (task: source batch 1 -> target batch 2; batches 3..10 never used here)
| variant | dev acc (3 seeds) | mean |
|---|---|---|
| SensorJEPA fine-tune (default cfg) | 66.80 78.54 65.92 | 70.42 |
| same network, no pretraining | 58.68 65.76 69.69 | 64.71 |
| batch-z MLP (baseline) | 74.20 74.84 75.48 | 74.84 |
| batch-z logistic regression (baseline) | 81.51 x3 | 81.51 |
| JEPA frozen + logistic probe | 49.92 55.14 55.95 | 53.67 |
| JEPA frozen + raw features + logistic probe | 65.68 73.79 76.13 | 71.86 |
| JEPA pretrained on all 10 batches' features, fine-tune | 57.48 54.34 64.31 | 58.71 |
| JEPA pretrained on all 10 batches' features, frozen + raw + logistic probe | 66.56 75.88 71.38 | 71.28 |
Status 2026-10-06: no JEPA variant reaches the plain batch-z logistic regression (81.51) on the dev task.
