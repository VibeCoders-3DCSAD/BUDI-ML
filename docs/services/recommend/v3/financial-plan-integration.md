# Budget Recommendation V3: Financial Plan Integration

```json
{
  "document-type": "integration-contract",
  "version": "3.0.0",
  "date": "2026-10-09",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **Source of truth:** This document implements the documentation boundary in
> [`financial-plan-classification-utilization-execution-plan.md`](../../../../../odin/plans/financial-plan-classification-utilization-execution-plan.md).

## Scope

`POST /api/v1/budget/recommend` produces a constrained category-allocation
recommendation. It accepts available funds, category restrictions, optional
target ratios, and an optional expense forecast. It returns category allocations,
feasibility information, and `unallocated_surplus`.

The endpoint does not accept or evaluate Financial Classification V2 output. It
must not use classification to change feasibility, category allocations,
category floors or ceilings, restriction levels, or priority weights.

## Financial Plan Pipeline

```text
available funds + obligations + category rules + expense forecast
    -> /api/v1/budget/recommend
    -> required category allocations and unallocated surplus
    -> Financial Plan classification-surplus policy
    -> advisory optional destination for user review
```

Required debt payments, credit-card statement requirements, savings reservations,
Fixed amounts, and Minimum floors are resolved before optional surplus. A
classification result never makes an otherwise infeasible plan feasible and
never reduces a required commitment.

## Classification-Surplus Policy

Financial Plan retrieves a fresh successful Financial Classification V2
assessment. It applies the following ordered policy only to optional surplus:

1. A `NEGATIVE_FINANCIAL_MARGIN` leaves all optional surplus unallocated for
   user review.
2. Otherwise, `EMERGENCY_FUND_INADEQUATE` recommends a verified eligible
   emergency-fund goal.
3. Otherwise, actionable debt signals recommend a verified active debt that
   permits extra payments.
4. Otherwise, the optional surplus remains user-directed.

Missing, stale, insufficient, or failed assessments are neutral: the Financial
Plan keeps optional surplus user-directed and does not claim a favourable
condition. A high DSTI result alone cannot recommend debt when no eligible debt
can receive an extra payment. Transactor and light-revolver results are
explanation-only in this version.

## Responsibilities

| Component | Responsibility |
| :--- | :--- |
| `/api/v1/budget/recommend` | Constrained category allocation and reporting `unallocated_surplus`. |
| `/api/v1/classification/v2` | Explainable EFC, DSTI, financial-margin, and credit-card-behavior assessments. |
| Financial Plan | Freshness validation, ordered advisory policy, explanation, and immutable accepted-plan snapshot. |
| Savings domain | Validate emergency-goal eligibility and remaining contribution capacity. |
| Debt domain | Select an active extra-payment-eligible debt using the snapshotted Avalanche or Snowball strategy. |
| User | Review, edit, accept, or reject every optional surplus recommendation. |

Debt and savings domain modules calculate the destination allocation. Financial
Plan does not transfer money or execute payments automatically.

## Related Documents

- [Financial Plan classification utilization plan](../../../../../odin/plans/financial-plan-classification-utilization-execution-plan.md)
- [Financial Classification V2 specification](../../classification/v2/specification.md)
- [Budget Recommendation V2 historical documentation](../v2/optimizer.md)
