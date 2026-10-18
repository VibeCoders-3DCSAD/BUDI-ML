# HFCE v4 Evaluation Report

**Generated:** 2026-10-15T06:31:56.586203+00:00
**Target:** `current_price_million_php`
**Data vintage:** 2026-08-07

## Evaluation Contract

- Initial training: `2000Q1-2016Q4` (68 quarters).
- Expanding rolling validation: `2017Q1-2023Q4`, up to four quarters per origin.
- Untouched test: `2024Q1-2026Q2`, one-step expanding refits.
- Candidates: seasonal naive, ETS, and SARIMA; ranking is MAE, SMAPE, RMSE, then MDA.
- Metrics: MAE, SMAPE, MDA, and RMSE only.

## Released Models

| Category | Released family | Test MAE (million PHP) | Test SMAPE | Test MDA | Test RMSE (million PHP) |
| --- | --- | ---: | ---: | ---: | ---: |
| alcohol_tobacco | sarima | 3757.64 | 3.98% | 80.00% | 4328.61 |
| clothing_footwear | sarima | 3366.96 | 5.03% | 100.00% | 4487.20 |
| communication | sarima | 1498.06 | 1.11% | 100.00% | 1793.95 |
| education | sarima | 8027.46 | 3.04% | 90.00% | 10325.86 |
| food | sarima | 40816.13 | 2.09% | 100.00% | 47769.92 |
| furnishings_household | sarima | 2834.28 | 2.13% | 100.00% | 3472.53 |
| health | sarima | 5581.02 | 2.26% | 100.00% | 6487.45 |
| housing_water_utilities | sarima | 12872.22 | 1.98% | 100.00% | 14780.51 |
| miscellaneous_goods_services | sarima | 9061.70 | 1.22% | 100.00% | 10262.03 |
| recreation_culture | sarima | 4123.65 | 5.13% | 100.00% | 4579.29 |
| restaurants_hotels | sarima | 17750.99 | 4.48% | 100.00% | 21265.75 |
| transport | sarima | 21021.49 | 4.46% | 100.00% | 28749.01 |

The full candidate and release-gate records are in `evaluation.json`.
