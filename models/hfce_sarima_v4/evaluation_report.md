# HFCE SARIMA v4 Evaluation Report

Twelve category-specific quarterly SARIMA models were trained on observed PSA HFCE current-price levels from `2000-Q1` through `2026-Q2`.

## Aggregate Held-Out Results

| Metric | Unweighted Macro Average |
| --- | ---: |
| MAPE | 5.22% |
| SMAPE | 5.11% |
| MDA | 94.70% |
| MAE | PHP 23,279.09 million |
| RMSE | PHP 29,227.25 million |

The held-out period is `2023-Q4` through `2026-Q2`. See `evaluation.json` for category-level candidates, selected specifications, and metrics.

## Serving Boundary

The models forecast national aggregate HFCE, not individual transactions. The API uses each next-quarter national category forecast only as a multiplier against the requesting user's completed-quarter category-spend baseline.
