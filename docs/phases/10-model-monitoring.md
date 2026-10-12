# Phase 10 — Model Monitoring

**Status:** Pending (no monitoring implemented yet).

## Purpose

Detect when a served model's assumptions about user behavior drift and trigger retraining,
per the Chapter 1 commitment that models "remain reliable as user circumstances evolve"
(Abdullahi et al., 2025 — concept drift).

## Planned capabilities

| Concern | Approach | Status |
|---------|----------|--------|
| Prediction latency / availability | `/ready`, `/metrics`; P99 tracking (aligns with System Admin Module monitoring) | Pending |
| Input distribution drift | Daily/weekly feature-distribution comparison vs. training feature stats (saved in `metadata.json`/feature columns) | Pending |
| Reference/data drift | Monitor ratio of fallback responses (FALLBACK status in anomaly/forecast) as a drift proxy | Pending |
| Retraining trigger | Explicit rule, e.g., drift metric exceeds threshold over a rolling window → flag for retraining per Phase 6–8 | Pending |
| Data quality gates | Reuse EDA flags (zero-income rows expected; log/Box-Cox transforms; month-1 degeneracy) as input-quality checks | Pending |

## Guardrails

- Any automatic action requires a human confirmation step; no unattended retrain-and-deploy.
- A concept-drift narrative belongs in Chapter 3/4 of the thesis, so log drift events with
  timestamps to make that evaluation possible.
- Retraining must re-run Phases 5–8 and satisfy the same pre-registered acceptance rules.

## References

- `Odin-Literature/docs/standards/batch-6-algorithm-screening.md`: drift/anomaly-background
  papers (Zhong 2025, Zhang & Duan 2025).
- `Odin-Paper/docs/ml/README.md` phase table (Phase 10 currently Pending).