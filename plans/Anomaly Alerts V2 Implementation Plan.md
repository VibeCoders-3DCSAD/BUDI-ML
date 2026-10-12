# Anomaly Alerts V2 Implementation Plan

## Goal

Replace served global-artifact anomaly scoring with a per-user, per-category transaction IQR decision using only earlier real expense history, and evaluate that rule on a separate held-out synthetic test corpus. Keep budget overspending behavior functional.

## Source Of Truth

- [Anomaly Alerts V2 Methodology](../docs/services/iqr/v2/methodology.md) (version 2.2.0).
- [Repository standards](../docs/standards/REPOSITORY-STANDARDS.md) and [documentation format](../docs/standards/documentation-format.md).
- Existing interfaces: `app/api/anomaly.py`, `app/schemas/anomaly.py`, `app/schemas/common.py`, `app/services/anomaly_service.py`.

## Non-Goals

- Forecast-residual detection, global anomaly training, fraud classification, or population thresholds.
- Using synthetic transactions as production personal history or as evidence of real-world accuracy.
- Changing the forecasting model or its training data.

## Execution Order

Finalize contract details, implement and test the personal detector, wire serving and readiness, then build an evaluation-only corpus and report. Treat the current forecaster-v3 worktree changes as unrelated user work.

## PR Stacking Strategy

```text
main -> anomaly-v2-contract -> anomaly-v2-detector -> anomaly-v2-serving -> anomaly-v2-evaluation
```

Merge bottom-up; each later PR targets the preceding branch (or use vanilla Git branches based on the previous branch; no Graphite requirement). Aim for reviewable slices; the serving PR is the cutover and must include contract, readiness, and integration tests together.

## Linear Sub-Issue Tracking

Create sub-issues from this plan when ready; no Linear parent issue was provided.

### 1. Lock the data and response contract

- Files: `docs/services/iqr/v2/methodology.md`, `app/schemas/common.py`, `app/schemas/anomaly.py`, `INDEX.md`.
- Specify the exact three-calendar-month boundary, timezone/date parsing and `(timestamp, transaction_id)` order; require unique non-null IDs for scoring and canonical category IDs. Define a trusted user-identity/history-provenance boundary before accepting caller-supplied `user_id` and transactions (the current request has no transaction owner or authenticated history source), plus response identity, nullable bounds/severity for both insufficient states, zero-IQR bounds behavior, and a ruleset version; reflect the now-single-channel methodology in the index.

### 2. Implement the personal IQR decision engine

- Files: `app/services/anomaly_service.py` (or focused `app/services/personal_iqr.py`), `tests/test_anomaly.py` (or `tests/test_personal_iqr.py`).
- Score sorted valid expenses using only prior transactions of the same user/category; gate on six earlier expenses and three calendar months, compute `log1p` plus NumPy `linear` quartiles and 1.5× fences, report `ANOMALOUS`/`NORMAL`/insufficient states and peso-scale bounds, and add every expense to the baseline **after** scoring (even flagged or insufficient ones). Tests cover tie ordering, cross-user/category isolation, repeated amounts/zero IQR, fence equality, lower and upper outliers, changing patterns, and no candidate leakage.

### 3. Cut over API and artifact-dependent readiness

- Files: `app/api/anomaly.py`, `app/schemas/anomaly.py`, `app/schemas/common.py`, `app/services/anomaly_service.py`, `app/models/registry.py`, `app/api/health.py`, `app/core/config.py`, `tests/test_anomaly.py`, `tests/test_health.py`, `tests/conftest.py`.
- Keep `/api/v1/anomaly/detect` and `/detect/batch` but replace model-score responses with per-transaction decision records, enforce the trusted user/history boundary from phase 1 (block cutover if unavailable), reject malformed input instead of silent fallback, and keep `OVERSPENDING`/`BOTH` semantics explicitly mapped to the new decisions. Remove the anomaly artifact as a readiness requirement, report the rule version without pretending a trained winner exists, revise artifact-dependent tests/fixtures, and update the service version for the breaking response contract.

### 4. Build held-out IQR evaluation only

- Files: `training/scripts/generate_transactions_v2.py` (reuse clean generation with `--no-anomalies`), `training/scripts/evaluate_personal_iqr.py` (new), `tests/test_personal_iqr_evaluation.py` (new), `docs/services/iqr/v2/methodology.md`.
- Generate or select clean synthetic user histories with stable transaction IDs; reserve a chronological held-out test interval per user/category, sample eligible test transactions, and inject only 3×–5× amount spikes in a separate `anomaly_test` copy with the provenance fields in the spec. Use the production decision engine with rolling earlier-history baselines (including flagged observations after scoring), report Accuracy/Precision/Recall/F1 with sample size, rate, undefined-metric handling and insufficient-state coverage; do not reuse v1's mixed anomaly injection or feed test interventions into the forecaster corpus.

### 5. Reconcile documentation and verify end to end

- Files: `README.md`, `INDEX.md`, `docs/services/iqr/v1/overview.md`, `docs/models/TEAMMATE-GUIDE.md`, `docs/services/iqr/v2/methodology.md`, `tests/test_anomaly.py`, `tests/test_health.py` (only where claims or tests are affected).
- Mark old trained-anomaly guidance as historical where appropriate, document the served personal rule and evaluation-only dataset, and check API examples and readiness claims against observed behavior. Run `.venv/bin/pytest tests/test_anomaly.py tests/test_health.py tests/test_personal_iqr_evaluation.py`, `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/mypy app`, and `git diff --check`; report pre-existing failures separately.

## Acceptance Criteria

- [ ] Same user's earlier valid expenses in the same canonical category are the sole production anomaly baseline; no globally trained anomaly artifact or synthetic data influences decisions.
- [ ] Production history and `user_id` are derived from a trusted user-scoped source rather than an unverified request claim.
- [ ] Every eligible transaction gets the correct state and required category, count, bounds and severity fields; insufficient history/variability never appears as `NORMAL`.
- [ ] Six prior expenses plus three calendar months are required; scoring is deterministic and a transaction never contributes to its own bounds.
- [ ] Flagged expenses enter later baselines so sustained spending changes can adapt; budget overspending and batch requests still work.
- [ ] `/ready` and `/metrics` truthfully describe the rule-based anomaly capability without requiring `models/anomaly/anomaly_detector.joblib`.
- [ ] A reproducible, held-out `anomaly_test` evaluation reports the four specified metrics and provenance without claiming real-world performance or training IQR.
- [ ] Tests, lint, formatting, type checks, and documentation checks pass (or pre-existing failures are recorded).
