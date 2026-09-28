# models/forecaster/

Canonical home for the **new-scope** spending forecaster artifact (`metadata.json` + final
`.joblib`). Current winner: **`tier3_sarima.joblib`** (SARIMA, degrades to plain ARIMA on
pools < 24 months), cutting MAPE ~75% vs naive in 5-fold temporal evaluation (see
`evaluation.json` / `evaluation_report.md`).

## Decision rule (pre-registered)

Winner must beat the naive baseline by **≥ 20% MAPE reduction**.

## Primary metrics

- MAE, SMAPE, MDA, RMSE

## Target artifact

- `tier3_sarima.joblib` — committed here (winner resolved from `metadata.json` at serve time).
- `app/models/registry.py` (`FORECASTER_MODULE = "forecaster"`) loads it at serve time via
  `_resolve_forecaster_artifact` (`.pth` + `_meta.joblib` pairs rebuild `_SequenceForecaster`).

## Committed state

- `metadata.json` — winner contract (`winner`, `winner_artifact`, `winner_params`, `threshold`).
- `evaluation.json` — raw fold metrics (source of truth for the report).
- `evaluation_report.md` — regenerated from `evaluation.json`.

Regenerate reports/metadata from existing `evaluation.json` without retraining:

```bash
python -m training.scripts.modeling.regenerate_artifacts
```
