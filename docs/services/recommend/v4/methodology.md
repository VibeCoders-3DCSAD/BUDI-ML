# Financial Planning Management V4 — MILP Methodology

```json
{
  "document-type": "system-spec",
  "version": "4.0.0",
  "date": "2026.10.11",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **Purpose:** This specification defines Financial Planning Management as a deterministic mixed-integer linear programming (MILP) allocation method. It creates a monthly category-allocation recommendation from available funds, user restrictions, priorities, and optional expense forecasts. It is not a trained machine-learning model.

---

## Methodology Overview

Financial Planning Management solves a constrained allocation problem for one calendar month.

```text
available funds + category rules + priorities + optional expense forecast
                                  |
                         two-stage MILP allocation
                                  |
                  feasible plan or explicit shortfall
```

The method first protects weighted Essential-category coverage, then allocates the remaining feasible funds with Essential categories prioritized over Non-essential categories. The result is deterministic: identical validated inputs produce the same recommendation.

---

## Scope and Inputs

The service operates at `POST /api/v1/budget/recommend` for exactly one calendar month.

| Input | Purpose |
| :--- | :--- |
| Available funds | Maximum funds available for the planning period. |
| Category restrictions | Each category's `FIXED`, `MINIMUM`, or `FLEXIBLE` restriction and its floor and ceiling. |
| Priority group and weight | Essential-first allocation order and relative category importance. |
| Target ratios | Optional preferred allocation proportions when no forecast target is supplied. |
| Expense forecast | Optional category target that caps discretionary allocation. |

Fixed and minimum amounts are hard user commitments. Forecasts and target ratios guide discretionary allocation only; they do not reduce a hard floor or make an infeasible plan feasible.

---

## Decision Variables and Units

Let `i` represent a budget category and let `u = PHP 10` be the allocation unit. The solver selects the non-negative integer variable:

$$
z_i \in \mathbb{Z}_{\geq 0}
$$

where the allocated peso amount is:

$$
x_i = u z_i
$$

Integer units ensure returned allocations are whole PHP 10 amounts. This avoids post-solver rounding that could otherwise violate a floor, ceiling, or total-funds constraint.

---

## Hard Constraints

For each category, the allocation must remain within its validated lower and upper bounds:

$$
l_i \leq x_i \leq U_i
$$

The bounds depend on the user's restriction:

| Restriction | Lower bound | Upper bound |
| :--- | :--- | :--- |
| `FIXED` | configured amount | same configured amount |
| `MINIMUM` | configured floor, rounded up to PHP 10 | lower of configured ceiling and forecast target, but never below the floor |
| `FLEXIBLE` | configured floor, rounded up to PHP 10 | lower of configured ceiling and forecast target, but never below the floor |

When no forecast is present, the target derives from the configured target ratio when supplied. A fixed amount must already be divisible by PHP 10.

The plan cannot allocate more than the available funds `B`:

$$
\sum_i x_i \leq B
$$

The model does not force every remaining peso into a category. Any amount that cannot be allocated without violating a category ceiling remains `unallocated_surplus` for the Financial Plan's separate advisory policy.

---

## Stage 1: Essential Coverage

For each Essential category with allocation range, define its weighted discretionary gap:

$$
g_i = w_i(U_i - l_i)
$$

where `w_i` is the category priority weight. The first MILP maximizes a common coverage variable `r` subject to:

$$
x_i \geq l_i + r g_i
$$

for every eligible Essential category, the category bounds, and the available-funds constraint.

This prevents a higher-priority Essential category from consuming all remaining funds while another Essential category receives only its floor. The stage returns the protected Essential coverage floors used by the final allocation.

---

## Stage 2: Essential-First Allocation

The second MILP retains the protected Essential floors and maximizes weighted allocation:

$$
\max \sum_i \left(w_i + P \cdot I_i\right)z_i
$$

where:

- `I_i = 1` when category `i` is Essential and `0` otherwise; and
- `P` is greater than the maximum possible weighted allocation contribution.

The Essential priority coefficient guarantees that a feasible Non-essential allocation cannot displace unmet Essential allocation. Within that hierarchy, category weights guide funds toward higher-priority categories.

Both stages use `scipy.optimize.milp` with the HiGHS solver.

---

## Feasibility and Output

Before solving, the service checks whether the total hard lower bounds exceed available funds:

$$
\sum_i l_i > B
$$

If this condition holds, the request is `INFEASIBLE`. The service returns the required funds, available funds, shortfall, and affected categories. It does not silently relax fixed amounts or minimum floors.

When the MILP solves successfully, the request is `FEASIBLE` and returns:

| Output | Meaning |
| :--- | :--- |
| Allocations | One whole-PHP-10 recommended amount per category. |
| Utilization rate | Allocated funds divided by requested available funds. |
| Constraint satisfaction | `1.0` only when all returned allocations meet their hard constraints. |
| Unallocated surplus | Available funds not allocated because all valid category capacity is exhausted. |
| Explanations | Restriction-level explanation for each returned category allocation. |

---

## Integration Boundary

The MILP determines required and constrained category allocations. It does not classify a user's financial condition, transfer money, execute debt payments, or force optional surplus into a savings goal or debt.

Financial Classification V2 evaluates the user's financial condition independently through fixed rules. After the MILP has reserved required commitments, the Financial Plan may apply the ordered classification-surplus policy to `unallocated_surplus`. See [Financial Plan Integration](../v3/financial-plan-integration.md) and [Financial Classification V2](../../classification/v2/specification.md).

---

## Final Methodology Label

**Financial Planning Management using Two-Stage Mixed-Integer Linear Programming**

This methodology creates a transparent monthly financial plan by satisfying non-negotiable user constraints, prioritizing Essential-category coverage, and preserving whole-PHP-10 feasibility without relying on trained predictive allocation models.
