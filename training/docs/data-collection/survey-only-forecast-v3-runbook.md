# Survey-Only Forecast v3 Runbook

Run `python -m training.scripts.pipeline.survey_pipeline_v3 --input <2023-fies.csv>`. It writes only `survey_v3/` and `training/datasets/processed_v3/`.

Then run `python -m training.scripts.pipeline.feature_engineering_forecaster_v3`. It writes RF-eligible rows only: target month `T` needs three strictly prior completed expense months. The output stores normalized lag and rolling features, the three-month `user_scale`, the normalized `target_ratio`, and the original PHP target for evaluation.

Run the final candidate and its same-row naive baseline:

```bash
python -m training.scripts.modeling.train_forecaster_v3_full --candidate naive
python -m training.scripts.modeling.train_forecaster_v3_full --candidate random_forest
python -m training.scripts.modeling.compare_forecaster_v3_full
```

The RF predicts `target_ratio` and the evaluator multiplies each prediction by that row's strictly-prior `user_scale` before calculating PHP MAE. This keeps the global learned behavior separate from each household's actual spending level. The resulting metrics are internal synthetic-target evidence only and do not establish real household transaction accuracy.

The RF command persists `models/forecaster_v3/random_forest.joblib`; the comparison command writes the winner contract consumed by the service at startup. If RF is not the winner, the service retains the v2 artifact.
