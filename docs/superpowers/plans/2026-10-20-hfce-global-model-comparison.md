# HFCE Global Model Comparison Implementation Plan

> **For agentic workers:** Execute the tasks inline in the authorized working directory.

**Goal:** Train one pooled LSTM and one pooled Random Forest across the 12 observed HFCE categories and compare them with individual SARIMA models.

**Architecture:** Use expanding quarterly histories with strictly synchronized category cutoffs. Predict log year-over-year growth and reconstruct levels from the known seasonal lag. Select pooled configurations on validation macro SMAPE; freeze them before evaluating test origins. Reuse the existing SARIMA validation-selected specifications, not test-dependent release substitutions.

**Tech Stack:** Existing NumPy, pandas, scikit-learn, PyTorch, statsmodels, pytest, and Ruff dependencies.

## Global Constraints

- Input: `training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv`, 106 quarters per category.
- Train: 2000Q1–2016Q4; validation: 2017Q1–2023Q4; test: 2024Q1–2026Q2.
- Evaluate horizons 1–4 at each origin, truncating at each split boundary.
- Refit models and scalers from scratch at each origin; do not use observations inside the forecast horizon.
- Preserve existing workspace changes and production artifacts.
- Record nonconvergence explicitly; never choose alternative SARIMA orders using test performance.

## Task 1: Build and verify the experiment

**Files:** Create `training/scripts/modeling/compare_hfce_global_models.py` and `tests/test_compare_hfce_global_models.py`.

**Interfaces:** `make_features(history, window)`, `make_samples(history, window)`, `fit_global(history, config, seed)`, `forecast_global(fitted, history, horizon)`, `evaluate(panel, start, end, config, seed, sarima_specs=None)`.

- [x] Implement seasonal growth inputs, category identity, calendar features, and training-only standardization.
- [x] Implement fixed-seed pooled RF/LSTM fitting and recursive horizon forecasting.
- [x] Implement synchronized expanding evaluation and per-category, macro, and horizon metrics.
- [x] Test feature causality, category cutoff isolation, recursive prediction, and chronological folds using synthetic quarterly panels.
- [x] Run `.venv/bin/pytest tests/test_compare_hfce_global_models.py tests/test_train_hfce_sarima_v4.py` and targeted Ruff checks; require all to pass.

## Task 2: Run and record the comparison

**Files:** Generated artifacts in `training/models/hfce_global_comparison/`; committed report and summary in `docs/experiments/`.

- [x] Print the exact split, candidate grid, sample counts, and walk-forward contract before training.
- [x] Evaluate all declared configurations on validation; lock one per pooled family before test evaluation.
- [x] Evaluate seasonal naive, original validation-selected SARIMA, and both pooled winners on matching validation and test origins.
- [x] Save forecast-level CSVs, selected configurations, dependency versions, source hashes, model states, and a four-quarter forecast after full-data refitting.
- [x] Write the results and limitations, including shared time shocks, seasonal feature engineering, and the short test window.
- [x] Add experiment navigation to `INDEX.md` without discarding existing edits.
