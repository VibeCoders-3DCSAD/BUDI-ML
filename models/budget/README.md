# models/budget/

Canonical home for the **new-scope** budget optimizer configuration (`metadata.json` + config).
The current Financial Planning Management serving implementation is a deterministic
two-stage linear program (LP) (`scipy.optimize.linprog` with `method="highs"`, continuous
allocations with centavo precision), so there is
no trained artifact; the "model" is the constraint formulation + configuration.

## Decision rule / metrics

- Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences.

## Target artifact

- `budget_config.json` — solver + constraint-contract snapshot (the current LP formulation pins
  `app/services/budget_service.py`).
- `metadata.json` — historical V1 LP evaluation contract (`scipy_linprog`), retained for provenance.

## Committed state

- `evaluation.json` — LP results over 600 synthetic personas (constraint satisfaction,
  utilization, deviation, feasibility breakdown); source of truth for the historical V1 report.
- `evaluation_report.md` — regenerated from `evaluation.json`.

The evaluation is deterministic and fast; rebuild artifacts with:

```bash
python -m training.scripts.modeling.regenerate_artifacts
```
