# Phase 9 — Deployment

**Status:** Partial (FastAPI serving API complete; containerization/CI–CD pending).

## Current state

- `app/` serves PFP (`/api/v1/pfp/classify`), forecast (`/api/v1/forecast/predict`), anomaly
  (`/api/v1/anomaly/detect`), and budget (`/api/v1/budget/recommend`) plus batch variants.
- Health/readiness/metrics: `GET /health`, `GET /ready`, `GET /metrics`.
- Loads artifacts via `app/models/loader.py` + `registry.py`.
- Target architecture: `Odin-Paper/docs/ml/1_problem-statement/deployment-architecture.md` (v1.1).

## Readiness checklist (what remains)

Path | Status | Notes
-----|--------|------
Dockerfile per module (8000–8005) | Pending | README "Recommended Next Steps" |
docker-compose.yml | Pending | matching deployment-architecture v1.1 |
CI pipeline (lint + mypy + pytest) | Pending | will use pyproject tooling |
Model artifact versioning | In progress | `models/` + `metadata.json` (Phase 8) |
Budget wiring to forecast+PFP outputs (`/api/v1/analyze`) | Pending | README next step |
Secrets / env handling | Pending | `.env*` gitignored; `!.env.example` |

## Deployment invariants

- Each service exposes `/health`, `/ready`, `/metrics`.
- Model loading is separate from API transport (`app/models/` vs `app/api/`) per
  REPOSITORY-STANDARDS.
- No secrets in images or logs.