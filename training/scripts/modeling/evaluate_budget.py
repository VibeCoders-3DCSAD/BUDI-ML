"""Evaluate the served budget LP on FIES 2023 households projected with HFCE growth.

Each sampled household becomes one monthly Financial Plan request:

- Household: drawn from FIES 2023 Volume 2 with probability proportional to
  RFACT (PSA survey weight).
- Baseline spend: FIES columns mapped to the 12 HFCE categories
  (training/config/fies_hfce_category_mapping_v4.json), divided by 12.
- Forecast target: baseline spend x category HFCE nominal growth, 2023 -> 2025.
- Available funds: TOINC / 12 x aggregate HFCE growth x an income shock.
- Priority group, priority weight, and ceiling follow Odin's
  recommendationService.ts (ESSENTIAL_HFCE_CATEGORY_SLUGS,
  applyFinancialPlanPriorityWeights, ceiling = max(available, floor)).
- User rules are not observed in FIES, so three profiles are evaluated:
  flexible (no floors), mixed (housing FIXED, other Essentials MINIMUM 70%),
  strict (housing FIXED, other Essentials MINIMUM 90%).

Metrics are reported per scenario and overall against a proportional baseline
(fund every floor, then close every category's gap to target by one factor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.schemas.budget import BudgetRequest  # noqa: E402
from app.services.budget_service import optimize  # noqa: E402

TOL = 1e-6
INCOME_SHOCKS = (1.0, 0.8, 0.6)
PROFILES = ("flexible", "mixed", "strict")
PROFILE_FLOOR = {"mixed": 0.7, "strict": 0.9}
# Mirrors ESSENTIAL_HFCE_CATEGORY_SLUGS in odin apps/api recommendationService.ts.
ESSENTIAL = frozenset(
    {
        "food",
        "housing_water_utilities",
        "health",
        "communication",
        "transport",
        "education",
        "furnishings_household",
        "clothing_footwear",
    }
)
FIXED_CATEGORY = "housing_water_utilities"


def load_households(fies_path: Path, mapping_path: Path) -> pd.DataFrame:
    """Return monthly category spend, monthly income, and RFACT per household."""
    mapping = json.loads(mapping_path.read_text())["categories"]
    columns = sorted({column for spec in mapping.values() for column in spec["fies_columns"]})
    fies = pd.read_csv(fies_path, usecols=["TOINC", "RFACT", *columns])
    monthly = pd.DataFrame(
        {
            category: fies[spec["fies_columns"]].sum(axis=1) / 12
            for category, spec in mapping.items()
        }
    )
    monthly["income"] = fies["TOINC"] / 12
    monthly["rfact"] = fies["RFACT"]
    return monthly


def hfce_growth(
    hfce_path: Path, base_year: int, target_year: int
) -> tuple[dict[str, float], float]:
    """Return per-category and aggregate nominal HFCE growth between two full years."""
    hfce = pd.read_csv(hfce_path)
    annual = (
        hfce[hfce.year.isin([base_year, target_year])]
        .groupby(["category", "year"])
        .current_price_million_php.sum()
        .unstack()
    )
    growth = (annual[target_year] / annual[base_year]).to_dict()
    return growth, float(annual[target_year].sum() / annual[base_year].sum())


def build_request(
    request_id: str,
    household: pd.Series,
    growth: dict[str, float],
    aggregate_growth: float,
    shock: float,
    profile: str,
) -> BudgetRequest:
    categories = list(growth)
    baseline = {category: float(household[category]) for category in categories}
    forecast = {
        category: round(baseline[category] * growth[category], 2) for category in categories
    }
    funds = round(float(household["income"]) * aggregate_growth * shock, 2)

    weights: dict[str, float] = {}
    for essential in (True, False):
        members = [category for category in categories if (category in ESSENTIAL) == essential]
        total = sum(baseline[category] for category in members)
        for category in members:
            weights[category] = baseline[category] / total if total > 0 else 1 / len(members)

    rows = []
    for category in categories:
        rule, floor = "FLEXIBLE", 0.0
        if profile != "flexible" and category == FIXED_CATEGORY:
            rule, floor = "FIXED", forecast[category]
        elif profile != "flexible" and category in ESSENTIAL:
            rule, floor = "MINIMUM", round(forecast[category] * PROFILE_FLOOR[profile], 2)
        rows.append(
            {
                "category_id": category,
                "target_type": "category",
                "priority_group": "ESSENTIAL" if category in ESSENTIAL else "NON_ESSENTIAL",
                "restriction_level": rule,
                "floor": floor,
                "ceiling": floor if rule == "FIXED" else max(funds, floor),
                "priority_weight": round(weights[category], 6),
            }
        )
    return BudgetRequest(
        request_id=request_id,
        user_id="evaluation",
        available_funds=funds,
        period={"start": "2026-11-01", "end": "2026-11-30"},
        categories=rows,
        forecast={
            "month": "2026-11",
            "total_amount": sum(forecast.values()),
            "category_forecasts": [
                {"category": category, "amount": amount} for category, amount in forecast.items()
            ],
            "model_version": "fies2023-hfce-growth",
            "generated_at": "2026-10-16T00:00:00Z",
            "quality": "PERSONALIZED",
        },
    )


def evaluate_request(request: BudgetRequest) -> dict:
    """Score one LP recommendation and the proportional baseline for the same request."""
    recommendation, _ = optimize(request)
    categories = request.categories
    funds = request.available_funds
    forecast = {point.category: point.amount for point in request.forecast.category_forecasts}
    target = {
        category.category_id: category.floor
        if category.restriction_level == "FIXED"
        else min(max(forecast[category.category_id], category.floor), category.ceiling)
        for category in categories
    }
    floors = sum(category.floor for category in categories)
    row = {
        "feasible": recommendation.feasibility == "FEASIBLE",
        "truly_infeasible": floors > funds + TOL,
    }
    if not row["feasible"]:
        return row

    lp = {allocation.category_id: allocation.amount for allocation in recommendation.allocations}
    gap = sum(target[category.category_id] - category.floor for category in categories)
    factor = min(1.0, max(0.0, (funds - floors) / gap)) if gap > 0 else 0.0
    baseline = {
        category.category_id: category.floor
        + factor * (target[category.category_id] - category.floor)
        for category in categories
    }
    active = [category for category, amount in target.items() if amount > 0]
    essential = [
        category.category_id
        for category in categories
        if category.priority_group == "ESSENTIAL" and target[category.category_id] > 0
    ]
    reachable = min(funds, sum(target.values()))

    def deviation(allocation: dict[str, float]) -> float:
        return sum(abs(allocation[c] - target[c]) for c in target) / sum(target.values())

    def category_deviation(allocation: dict[str, float]) -> float:
        return float(np.mean([abs(allocation[c] - target[c]) / target[c] for c in active]))

    def essential_coverage(allocation: dict[str, float]) -> float:
        total = sum(target[c] for c in essential)
        return sum(allocation[c] for c in essential) / total if total else 1.0

    row.update(
        constraints_satisfied=sum(lp.values()) <= funds + TOL
        and all(
            category.floor - TOL <= lp[category.category_id] <= category.ceiling + TOL
            and (
                category.restriction_level != "FIXED"
                or abs(lp[category.category_id] - category.floor) <= TOL
            )
            for category in categories
        ),
        utilization_raw=sum(lp.values()) / funds,
        utilization_capacity=sum(lp.values()) / reachable if reachable else 1.0,
        deviation_lp=deviation(lp),
        deviation_baseline=deviation(baseline),
        category_deviation_lp=category_deviation(lp),
        category_deviation_baseline=category_deviation(baseline),
        essential_coverage_lp=essential_coverage(lp),
        essential_coverage_baseline=essential_coverage(baseline),
    )
    return row


def summarize(frame: pd.DataFrame) -> dict:
    feasible = frame[frame.feasible]
    infeasible = frame[~frame.feasible]
    metrics = {
        "plans": len(frame),
        "feasible_rate": float(frame.feasible.mean()),
        "infeasibility_detection_accuracy": float(infeasible.truly_infeasible.mean())
        if len(infeasible)
        else 1.0,
    }
    if len(feasible):
        metrics["constraint_satisfaction_rate"] = float(feasible.constraints_satisfied.mean())
        for column in [
            "utilization_raw",
            "utilization_capacity",
            "deviation_lp",
            "deviation_baseline",
            "category_deviation_lp",
            "category_deviation_baseline",
            "essential_coverage_lp",
            "essential_coverage_baseline",
        ]:
            metrics[column] = float(feasible[column].mean())
    return metrics


def run(
    households: pd.DataFrame,
    growth: dict[str, float],
    aggregate_growth: float,
    *,
    n_households: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    weights = households.rfact / households.rfact.sum()
    sample = households.iloc[rng.choice(len(households), size=n_households, p=weights)]
    sample = sample.reset_index(drop=True)
    rows = [
        {
            "profile": profile,
            "income_shock": shock,
            **evaluate_request(
                build_request(str(i), household, growth, aggregate_growth, shock, profile)
            ),
        }
        for profile in PROFILES
        for shock in INCOME_SHOCKS
        for i, household in sample.iterrows()
    ]
    frame = pd.DataFrame(rows)
    report = {
        "task": "budget_optimization",
        "solver": "scipy.optimize.linprog (HiGHS), two-stage LP",
        "source": "FIES 2023 Volume 2 (RFACT-weighted) projected with PSA HFCE 2023->2025 growth",
        "seed": seed,
        "households": n_households,
        "fies_households": len(households),
        "hfce_growth": {"aggregate": aggregate_growth, **growth},
        "overall": summarize(frame),
        "scenarios": {
            f"{profile}_income_x{shock}": summarize(
                frame[(frame.profile == profile) & (frame.income_shock == shock)]
            )
            for profile in PROFILES
            for shock in INCOME_SHOCKS
        },
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fies",
        type=Path,
        default=Path("training/datasets/fies/fies_2023_volume2_household_summary.csv"),
    )
    parser.add_argument(
        "--hfce",
        type=Path,
        default=Path("training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"),
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("training/config/fies_hfce_category_mapping_v4.json")
    )
    parser.add_argument("--output", type=Path, default=Path("training/datasets/budget_evaluation"))
    parser.add_argument("--households", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    households = load_households(args.fies, args.mapping)
    growth, aggregate_growth = hfce_growth(args.hfce, 2023, 2025)
    frame, report = run(
        households, growth, aggregate_growth, n_households=args.households, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "plans.csv", index=False)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
