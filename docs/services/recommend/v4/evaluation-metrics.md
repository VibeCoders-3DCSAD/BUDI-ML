# Budget LP Evaluation Metrics

Metrics used to evaluate the budget linear program (`app/services/budget_service.py`)
on FIES 2023 households projected with HFCE 2023→2025 growth. Produced by
`training/scripts/modeling/evaluate_budget.py`; results are in
`training/datasets/budget_evaluation/evaluation.json`.

All metrics are computed over feasible plans only.

## Expected Outcomes

| Metric | Expected outcome |
|---|---|
| Constraint Satisfaction Rate | Higher is better; should be 100%. Any plan below 100% breaks a user rule (overspends, violates a floor or ceiling, or changes a FIXED amount) and is a bug, not a trade-off |
| Idle Funds Rate | Lower is better; should be 0%. Money should only stay unallocated when every category has already reached its target |
| Essential Coverage | Higher is better, up to 100%. When funds are short, Essential targets should be funded before Non-Essential ones |

## Results

| Metric | Result |
|---|---|
| Constraint Satisfaction Rate | 100.0% |
| Idle Funds Rate | 0.0% |
| Essential Coverage | 97.1% |

## 1. Constraint Satisfaction Rate

$$
\text{CSR} = \frac{V}{P} \times 100\%
$$

where:

- $V$ = number of plans that break no rule
- $P$ = total number of feasible plans

A plan breaks no rule when:

- the total allocated does not exceed the available funds
- every category gets at least its minimum and at most its maximum
- every FIXED category gets exactly its fixed amount

## 2. Idle Funds Rate

Replaces Budget Utilization Rate (BUR), which penalized keeping surplus money after every
target was already met.

$$
\text{IFR} = \frac{I}{P} \times 100\%
$$

where:

- $I$ = number of plans where $A < \min(B,\ T)$
- $P$ = total number of feasible plans
- $A$ = total amount allocated in the plan
- $B$ = available funds
- $T$ = sum of all category targets

$\min(B,\ T)$ is the most a plan can usefully allocate: all of its funds, or every target
in full, whichever is smaller. A plan below that has money left over while some category is
still under its target.

## 3. Essential Coverage

$$
\text{EC} = \frac{A_E}{T_E} \times 100\%
$$

where:

- $A_E$ = total amount allocated to Essential categories
- $T_E$ = total target of Essential categories

EC is computed per plan, then averaged over all feasible plans.

## Category Target

The target $T_i$ used above is the category's forecast, kept within its floor and ceiling:

$$
T_i = \min\big(\max(\text{forecast}_i,\ F_i),\ C_i\big)
$$

where:

- $T_i$ = target of category $i$
- $\text{forecast}_i$ = forecast amount for category $i$
- $F_i$ = floor (minimum) of category $i$
- $C_i$ = ceiling (maximum) of category $i$

For a FIXED category, $T_i = F_i$.
