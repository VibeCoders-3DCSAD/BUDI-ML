# HFCE SARIMA v4 Evaluation Report

Twelve category-specific quarterly SARIMA models were selected from observed PSA HFCE current-price levels from `2000-Q1` through `2026-Q2`.

## Selection Procedure

The final model for each purpose uses a quarterly seasonal period of `s=4`. COVID observations from `2020-Q1` through `2021-Q4` remain in the training history.

| Stage | Period | Purpose |
| :--- | :--- | :--- |
| Initial training history | `2000-Q1` to `2021-Q4` | Fits the first rolling-validation model and includes COVID observations. |
| Expanding validation | `2022-Q1` to `2023-Q4` | Two sequential four-quarter forecast folds select each category's model. |
| Untouched test | `2024-Q1` to `2026-Q2` | Evaluates the selected model under post-COVID conditions. |

Each category searches all valid combinations of `p,q=0..3`, `d=0..1`, `P,Q=0..2`, and `D=0..1`, with `s=4`. Models with both `d=1` and `D=1` are excluded to avoid double differencing. The search compares raw and `log1p` targets, yielding 864 candidates per category. Candidates that do not converge or produce non-finite forecasts are rejected. The winner has the lowest rolling four-quarter sMAPE; MAE, RMSE, and MDA resolve ties.

## Selected Models

| Category | Selected SARIMA | Transform | Test MAPE |
| :--- | :--- | :--- | ---: |
| Alcohol and tobacco | `(3,0,0)(1,0,2)_4` | Raw | 5.23% |
| Clothing and footwear | `(0,0,2)(1,0,2)_4` | `log1p` | 6.36% |
| Communication | `(0,0,1)(1,1,0)_4` | Raw | 3.69% |
| Education | `(0,0,0)(2,0,1)_4` | `log1p` | 1.49% |
| Food | `(0,0,0)(1,0,1)_4` | `log1p` | 7.48% |
| Furnishings and household equipment | `(0,0,0)(1,0,2)_4` | `log1p` | 1.83% |
| Health | `(0,0,1)(2,0,0)_4` | Raw | 4.81% |
| Housing, water, and utilities | `(0,0,2)(2,0,2)_4` | `log1p` | 7.31% |
| Miscellaneous goods and services | `(3,0,1)(1,0,1)_4` | Raw | 2.18% |
| Recreation and culture | `(3,1,0)(1,0,0)_4` | `log1p` | 7.52% |
| Restaurants and hotels | `(0,1,1)(1,0,0)_4` | `log1p` | 16.11% |
| Transport | `(2,0,1)(0,0,1)_4` | Raw | 5.23% |

## Aggregate Held-Out Results

| Metric | Unweighted Macro Average |
| --- | ---: |
| MAPE | 5.77% |
| SMAPE | 5.52% |
| MDA | 92.50% |
| MAE | PHP 28,032.68 million |
| RMSE | PHP 32,727.09 million |

The held-out period is `2024-Q1` through `2026-Q2`. See `evaluation.json` for the full search contract, top-ten validation leaderboards, selected specifications, and test metrics.

## Serving Boundary

The models forecast national aggregate HFCE, not individual transactions. The API uses each next-quarter national category forecast only as a multiplier against the requesting user's completed-quarter category-spend baseline.
