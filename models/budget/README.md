# models/budget/

Canonical home for the **new-scope** budget optimizer configuration (`metadata.json` + config).
Budget optimization is a deterministic LP (`scipy.linprog`) so there is no trained artifact;
the "model" is the constraint formulation + configuration.

## Decision rule / metrics

- Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences.

## Target artifact

- `budget_config.json` — solver + constraint-contract snapshot (current v1 formulation pins
  `app/services/budget_service.py`).
- `metadata.json` — winner contract for the deterministic formulation (`scipy_linprog`).

## Committed state

- `evaluation.json` — LP results over 600 synthetic personas (constraint satisfaction,
  utilization, deviation, feasibility breakdown); source of truth for the report.
- `evaluation_report.md` — regenerated from `evaluation.json`.

The evaluation is deterministic and fast; rebuild artifacts with:

```bash
python -m training.scripts.modeling.regenerate_artifacts
```
