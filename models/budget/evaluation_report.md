# Budget — Budget Optimization Evaluation Report

**Generated:** 2026-09-08T03:54:48.191820Z
**Folds:** 1
**Decision rule:** LP must satisfy all hard constraints with high utilization and minimal deviation from user preferences

## Winner

- **Tier:** scipy_linprog
- **Artifact:** budget_config.json
- **Reason:** scipy.linprog (exact, fast) selected as current v1 per model candidate roster

## Approval Criteria Result

**Result:** PASS — LP must meet all hard constraints with minimal deviation from user targets.

## Aggregate Results

| Metric | Value |
|--------|-------|
| Constraint satisfaction rate | 1.0000 |
| Budget utilization rate | 1.0000 |
| Mean deviation from user preferences | 0.0410 |
| Feasibility | 541/600 FEASIBLE, 59/600 INFEASIBLE |

## Per-Fold Results

_No per-fold breakdown._
