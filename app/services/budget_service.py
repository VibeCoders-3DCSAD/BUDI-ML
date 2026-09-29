from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from app.schemas.budget import (
    BudgetAllocation,
    BudgetCategory,
    BudgetExplanation,
    BudgetRecommendation,
    BudgetRequest,
    RestrictionLevel,
)

EPS = 1e-9


def optimize(request: BudgetRequest) -> tuple[BudgetRecommendation, list[BudgetExplanation]]:
    """Solve one monthly Financial Plan allocation without relaxing user rules.

    Decision variables: allocation `x_i` plus deviation `d_i` per category,
    where `d_i >= |x_i - target_i|`. The LP minimizes the weighted sum of
    deviations, so recommendations stay as close as possible to the user's
    target ratios (Deviation-from-User-Preferences KPI).

    Constraints:
      - sum of allocations == available funds,
      - FIXED categories held at their configured amount,
      - MINIMUM categories keep their configured floor,
      - FLEXIBLE categories respect [floor, ceiling].

    Feasibility:
      - FEASIBLE: all constraints met, allocations sum to available funds.
      - INFEASIBLE: no allocation satisfies all hard constraints.
    """
    categories = request.categories
    n = len(categories)
    funds = request.available_funds

    forecast_targets = _forecast_targets(request)
    target_ratios = request.target_ratios or {}
    targets = np.array(
        [
            forecast_targets.get(
                category.category_id, funds * target_ratios.get(category.category_id, 0.0)
            )
            for category in categories
        ]
    )
    weights = np.array([max(cat.priority_weight, EPS) for cat in categories])

    for cat in categories:
        if cat.ceiling < cat.floor:
            raise ValueError(f"category {cat.category_id}: ceiling below floor")
        if cat.restriction_level == RestrictionLevel.FIXED and abs(cat.floor - cat.ceiling) > EPS:
            raise ValueError(
                f"category {cat.category_id}: fixed amount requires equal floor and ceiling"
            )

    # Objective: minimize sum(w_i * d_i)  -> c = [0]*n allocations, [w] deviations
    c = np.concatenate([np.zeros(n), weights])

    # Inequality constraints A_ub @ [x, d] <= b_ub:
    #   x_i - d_i <= target_i   (under-shoot)
    #   -x_i - d_i <= -target_i (over-shoot)
    A_ub, b_ub = [], []
    for i in range(n):
        row_lo = [0.0] * (2 * n)
        row_lo[i] = 1.0
        row_lo[n + i] = -1.0
        A_ub.append(row_lo)
        b_ub.append(targets[i])

        row_hi = [0.0] * (2 * n)
        row_hi[i] = -1.0
        row_hi[n + i] = -1.0
        A_ub.append(row_hi)
        b_ub.append(-targets[i])

    # current_spend remains reporting context; user rules define plan bounds.
    allocation_bounds = [_allocation_bounds(category) for category in categories]
    minimum_required = sum(bound[0] for bound in allocation_bounds)
    maximum_available = sum(bound[1] for bound in allocation_bounds)
    if minimum_required > funds + EPS:
        return _infeasible_recommendation(
            categories, allocation_bounds, funds, minimum_required, "minimum_required"
        ), []
    if maximum_available < funds - EPS:
        return _infeasible_recommendation(
            categories, allocation_bounds, funds, maximum_available, "maximum_capacity"
        ), []
    bounds: list[tuple[float, float | None]] = [*allocation_bounds]
    bounds.extend([(0.0, None)] * n)

    A_eq = [[1.0 if j < n else 0.0 for j in range(2 * n)]]
    b_eq = [funds]

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return _infeasible_recommendation(
            categories, allocation_bounds, funds, minimum_required, "solver"
        ), []

    allocations = np.maximum(result.x[:n], 0.0)
    utilization = float(allocations.sum()) / funds if funds > 0 else 0.0

    constraint_satisfaction = 1.0
    for i, cat in enumerate(categories):
        if (
            cat.restriction_level == RestrictionLevel.FIXED
            and abs(allocations[i] - cat.floor) > EPS
        ):
            constraint_satisfaction = 0.0
        if allocations[i] < cat.floor - EPS or allocations[i] > cat.ceiling + EPS:
            constraint_satisfaction = 0.0

    # A successful equality-constrained LP is fully utilized by definition.
    feasibility = "INFEASIBLE" if constraint_satisfaction < 0.99 else "FEASIBLE"
    recommendation = BudgetRecommendation(
        allocations=[
            BudgetAllocation(category_id=cat.category_id, amount=round(float(a), 2))
            for cat, a in zip(categories, allocations, strict=True)
        ],
        utilization_rate=round(utilization, 4),
        constraint_satisfaction=round(constraint_satisfaction, 4),
        feasibility=feasibility,
    )

    explanations = []
    for cat, a in zip(categories, allocations, strict=True):
        if cat.restriction_level == RestrictionLevel.FIXED:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason=f"FIXED category held at {a:.2f}",
                )
            )
        elif cat.restriction_level == RestrictionLevel.MINIMUM:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason=f"MINIMUM category held at {a:.2f} or above",
                )
            )
        elif a > 0:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason="Allocation set within floor/ceiling bounds",
                )
            )

    return recommendation, explanations


def _allocation_bounds(category: BudgetCategory) -> tuple[float, float]:
    if category.restriction_level == RestrictionLevel.FIXED:
        return (category.floor, category.floor)
    return (category.floor, category.ceiling)


def _forecast_targets(request: BudgetRequest) -> dict[str, float]:
    if request.forecast is None:
        return {}
    return {point.category: point.amount for point in request.forecast.category_forecasts}


def _infeasible_recommendation(
    categories: list[BudgetCategory],
    bounds: list[tuple[float, float]],
    funds: float,
    constrained_total: float,
    reason: str,
) -> BudgetRecommendation:
    shortfall = abs(constrained_total - funds)
    allocated_amount = sum(lower for lower, _upper in bounds)
    return BudgetRecommendation(
        allocations=[
            BudgetAllocation(category_id=category.category_id, amount=round(lower, 2))
            for category, (lower, _upper) in zip(categories, bounds, strict=True)
        ],
        utilization_rate=round(min(allocated_amount / funds, 1.0), 4),
        constraint_satisfaction=0.0,
        feasibility="INFEASIBLE",
        required_funds=round(constrained_total, 2),
        available_funds=round(funds, 2),
        shortfall=round(shortfall, 2),
        conflicting_categories=[
            category.category_id
            for category, (lower, upper) in zip(categories, bounds, strict=True)
            if lower > 0 or (reason == "maximum_capacity" and upper > 0)
        ],
    )
