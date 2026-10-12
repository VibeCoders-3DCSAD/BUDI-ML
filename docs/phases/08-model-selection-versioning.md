# Phase 8 — Model Selection & Versioning

**Status:** Partial (selection + versioning committed for `pfp`, `forecaster`, `anomaly`;
budget is a deterministic LP pinned in `models/budget/`).

## Purpose

Turn the Phase 7 evaluation results into a single surviving artifact per family, versioned and
servable, with full provenance.

## Selection process

1. Apply the pre-registered acceptance rule (see `07-model-evaluation.md`) per family.
2. If multiple candidates pass, prefer by: **primary metric → interpretability → artifact size
   / inference cost → integration effort** (matches the Chapter 1 scope language:
   "evaluation metrics, computational requirements, interpretability, and suitability for
   integration").
3. Record the decision in `models/<family>/metadata.json` (`decision_rule` + `winner_reason`).

## Versioning

- SemVer-style `model_id`, e.g. `pfp-v1.0.0`, isolated per family.
- `training_commit` + `training_data_hash` are mandatory in `metadata.json` so any artifact can
  be reproduced.
- The winner contract (`winner`, `winner_artifact`, `winner_params`, `threshold`) is uniform
  across families and is what `app/models/registry.py` resolves at serve time — no code change
  is needed to swap a family's winner.

## Promotions

- Promotion to **release** = artifact committed in `models/<family>/` + `metadata.json` +
  `evaluation_report.md` present.
- Serving layer (`app/models/registry.py`) reads `metadata.json`; the winner contract makes
  adding/modifying/removing a family's winner a metadata change rather than a code change.