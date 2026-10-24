# Spending Forecaster V1

```json
{
  "document-type": "model-evaluation",
  "version": "1.1.1",
  "date": "2026.10.18",
  "authors": ["Group 4, III-DCSAD"]
}
```

This first service-document generation describes an unserved research pipeline using selected 2023 FIES households and PSA HFCE temporal allocation. Its monthly targets are synthetic allocations, so internal metrics do not establish real household forecast accuracy. It is not served. `/api/v1/forecast/predict` uses the HFCE quarterly SARIMA artifacts in `models/hfce_sarima_v4/` (see [`forecast/v3/overview.md`](../v3/overview.md)); deploying this research pipeline would require external validation and an explicit artifact-release approval.

The model and training artifacts retain their `forecaster_v3` names; artifact versions are independent of service-document versions.

## Candidate Contract

The v3 Random Forest is a global model: it learns normalized next-month spending behavior across training households, rather than an absolute PHP forecast per household.

For target month `T`, only the three completed months before `T` are used:

```text
user_scale[T] = mean(expenses[T-3], expenses[T-2], expenses[T-1])
target_ratio[T] = expenses[T] / user_scale[T]
```

The input features are the three prior expense lags and their three-month standard deviation, each divided by `user_scale[T]`. The forest predicts `target_ratio`; serving or evaluation converts that result back to PHP:

```text
forecast_php[T] = predicted_ratio[T] * user_scale[T]
```

This requires three completed positive-expense months. Shorter histories use a non-model fallback and are excluded from RF MAE. The naive comparator uses `user_scale[T]` as its PHP prediction on the identical eligible rows.

The research pipeline deliberately excludes calendar month and quarter features from this RF contract. The source allocates each PSA quarter equally over its three months, so calendar features could primarily memorize a synthetic allocation assumption rather than learn user behavior.

## Evaluation Limit

Evaluation is household-disjoint and reports MAE in PHP after rescaling. It tests whether the RF predicts the pipeline's synthetic monthly allocation for held-out households. It does not establish accuracy on observed household transaction history and must not authorize deployment by itself.

## Service Activation

Full-corpus RF training persists `models/forecaster_v3/random_forest.joblib`. After the naive and RF reports are compared, `compare_forecaster_v3_full.py` writes the compatible winner contract to `models/forecaster_v3/evaluation.json`. The service does not load any `models/forecaster_v3/` artifact; activating it would require a code change to the model registry.
