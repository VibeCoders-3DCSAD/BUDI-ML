# ML Development Phases (Odin-ML)

Complete model development lifecycle reference for the revised BUDI ML pipeline. Design
documents and phase-by-phase narratives live in `Odin-Paper/docs/ml/`; this directory holds
the Odin-ML-side operational artifacts and runbooks.

Phase numbers match `Odin-Paper/docs/ml/README.md`.

| Phase | Name | Status | Odin-ML artifact |
|-------|------|--------|------------------|
| 1 | Problem Statement | Complete | (design docs in `Odin-Paper/docs/ml/1_problem-statement/`) |
| 2 | Data Collection | Complete | `training/scripts/data/collector.py`, `raw-datasets/` |
| 3 | Data Preprocessing | Complete | `training/scripts/pipeline/preprocessor.py`, `training/datasets/processed/` |
| 4 | EDA | Complete | `training/scripts/analysis/eda.py`, `training/figures/eda_report.md` |
| 4.5 | Dimension & Threshold Discovery | Complete | `training/scripts/analysis/dimension_discovery.py`, `training/docs/dimension-threshold-discovery/` |
| 5 | Feature Engineering | Complete | `training/scripts/pipeline/feature_engineering*.py`, `training/datasets/{engineered,forecaster,anomaly}/` |
| 6 | Model Training | Complete | `training/scripts/modeling/train_*.py` (tier engines) |
| 7 | Model Evaluation | **Complete** | `models/*/evaluation.json` + `evaluation_report.md` (see `phases/07-model-evaluation.md`) |
| 8 | Model Selection & Versioning | **Partial** | `models/` (top-level, committed winners + `metadata.json`), `phases/08-model-selection-versioning.md` |
| 9 | Deployment | **Partial** | `app/` FastAPI serving; `phases/09-deployment.md` |
| 10 | Model Monitoring | **Pending** | `phases/10-model-monitoring.md` |

## Phase documents

- [`07-model-evaluation.md`](07-model-evaluation.md)
- [`08-model-selection-versioning.md`](08-model-selection-versioning.md)
- [`09-deployment.md`](09-deployment.md)
- [`10-model-monitoring.md`](10-model-monitoring.md)

## Notes

- New-scope evaluation runs commit `evaluation.json` + `evaluation_report.md` +
  `metadata.json` to `models/<family>/` (no training artifacts are tracked). Reports and
  metadata can be re-emitted from an existing `evaluation.json` with
  `python -m training.scripts.modeling.regenerate_artifacts` (see `models/README.md`).
- Pre-registered decision rules are the source of truth for acceptance and are restated
  per-phase here so a reviewer never has to open the training scripts to know the rule.
