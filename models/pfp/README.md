# models/pfp/

Canonical home for the **new-scope** PFP classification artifact (`metadata.json` + final
`.joblib`). Current winner: **`tier3_svm.joblib`** (calibrated SVM), selected after 5-fold
temporal evaluation (see `evaluation.json` / `evaluation_report.md`).

## Decision rule (pre-registered)

Winner must beat the Tier-1 rule-based floor by **≥ 0.02 Macro-F1** (5-fold expanding window).

## Primary metric

- Macro-F1 (primary); Accuracy (secondary)

## Target artifact

- `tier3_svm.joblib` — committed here (winner resolved from `metadata.json` at serve time).
- `app/models/registry.py` (`PFP_MODULE = "pfp"`) loads it at serve time via
  `_resolve_pfp_artifact`.

## Committed state

- `metadata.json` — winner contract (`winner`, `winner_artifact`, `winner_params`, threshold).
- `evaluation.json` — raw fold metrics (source of truth for the report).
- `evaluation_report.md` — regenerated from `evaluation.json`.

Regenerate reports/metadata from existing `evaluation.json` without retraining:

```bash
python -m training.scripts.modeling.regenerate_artifacts
```
