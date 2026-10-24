"""Evaluate the served budget LP on FIES 2025 households.

Each sampled household becomes one monthly Financial Plan request for an average
2025 month. Every input is observed 2025 data, so nothing is projected. Parameter
sources are listed in docs/services/recommend/v4/evaluation-metrics.md.

- Household: drawn from the FIES 2025 public-use file with probability
  proportional to RFACT (PSA survey weight).
- Spending: the 12 HFCE purposes per RAW/uncleaned/2025/CLEANING_RULES.md,
  divided by 12. Imputed rent is removed, and housing is split into rent
  (ACTRENT) and utilities (the rest of HOUSING_WATER).
- Target: the household's realised 2025 spending, a perfect-forecast target, so
  the LP is tested independently of forecast error. A seasonal stress variant
  scales targets by observed HFCE 2025-Q4 / the 2025 quarterly mean.
- Available funds: (TOINC - imputed rent) / 12 x an income shock.
- Priority group, priority weight, and ceiling follow Odin's
  recommendationService.ts (ESSENTIAL_HFCE_CATEGORY_SLUGS,
  applyFinancialPlanPriorityWeights, ceiling = max(available, floor)).
- Rent is FIXED in every profile. Other Essentials take a MINIMUM floor at a
  swept share of target, or at the PSA 2025 food and poverty thresholds
  (subsistence profile).

Metrics follow docs/services/recommend/v4/evaluation-metrics-study.md: Constraint
Satisfaction Rate, Idle Funds Rate, Essential Coverage, and Priority Validation
Rate, computed on feasible plans and reported per scenario cell with its
Feasibility Rate.
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
FIES_YEAR = 2025
# Typical shock: about -7% average post-typhoon household income in FIES data
# (Anttila-Hughes & Hsiang, 2013; Schleypen et al., 2024). Severe, pandemic-type
# stress case: about half of income lost (World Bank, 2022, q47a1).
INCOME_SHOCKS = (1.0, 0.93, 0.5)
# Hypothetical user strictness swept as a sensitivity factor (Kleijnen, 2015); 0%
# matches the app default of no user minimum. The 50/70/90% levels are ours.
FLOOR_SHARES = {"floor_0": 0.0, "floor_50": 0.5, "floor_70": 0.7, "floor_90": 0.9}
SUBSISTENCE = "subsistence"
PROFILES = (*FLOOR_SHARES, SUBSISTENCE)
# PSA 2025p per-capita annual poverty threshold, Philippines (PSA, 2026, Ref.
# 2026-273, Table 1); PhP 14,634 a month for a family of five.
POVERTY_THRESHOLD = 35_120.89 / 12
# PSA's 2025 release has no food threshold table, so it is derived with the 2023
# food share of the poverty threshold (PhP 22,994 / PhP 33,295.99, Table 1).
FOOD_THRESHOLD = POVERTY_THRESHOLD * 22_994 / 33_295.99
# Non-food basics the PSA poverty line prices besides housing (rent is FIXED).
SUBSISTENCE_NON_FOOD = (
    "housing_utilities",
    "clothing_footwear",
    "transport",
    "health",
    "education",
)
# LP category -> HFCE purpose, for the seasonal stress multiplier.
LP_CATEGORIES = {
    "food": "food",
    "alcohol_tobacco": "alcohol_tobacco",
    "clothing_footwear": "clothing_footwear",
    "housing_rent": "housing_water_utilities",
    "housing_utilities": "housing_water_utilities",
    "furnishings_household": "furnishings_household",
    "health": "health",
    "transport": "transport",
    "communication": "communication",
    "recreation_culture": "recreation_culture",
    "education": "education",
    "restaurants_hotels": "restaurants_hotels",
    "miscellaneous_goods_services": "miscellaneous_goods_services",
}
# FIES 2025 columns per purpose (RAW/uncleaned/2025/CLEANING_RULES.md). Food and
# housing are derived in load_households.
FIES_COLUMNS = {
    "alcohol_tobacco": ["ALCOHOL", "TOBACCO"],
    "clothing_footwear": ["CLOTH"],
    "furnishings_household": ["FURNISHING"],
    "health": ["HEALTH"],
    "transport": ["TRANSPORT"],
    "communication": ["COMMUNICATION"],
    "recreation_culture": ["RECREATION"],
    "education": ["EDUCATION"],
    "restaurants_hotels": ["FOOD_OUTSIDE"],
    "miscellaneous_goods_services": ["MISCELLANEOUS"],
}
FIXED_CATEGORY = "housing_rent"
# Mirrors ESSENTIAL_HFCE_CATEGORY_SLUGS in odin apps/api recommendationService.ts.
ESSENTIAL = frozenset(
    {
        "food",
        "housing_rent",
        "housing_utilities",
        "health",
        "communication",
        "transport",
        "education",
        "furnishings_household",
        "clothing_footwear",
    }
)
# Sensitivity run: communication and furnishings are not in the PSA basic-needs basket.
ESSENTIAL_PSA_BASKET = ESSENTIAL - {"communication", "furnishings_household"}
# A label for request validation only; targets are an average 2025 month.
PLAN_PERIOD = {"start": "2025-06-01", "end": "2025-06-30"}


def load_households(fies_path: Path) -> pd.DataFrame:
    """Return monthly cash spend per LP category, cash income, size, and RFACT."""
    columns = sorted({column for names in FIES_COLUMNS.values() for column in names})
    fies = pd.read_csv(
        fies_path,
        usecols=[
            "TOINC",
            "RFACT",
            "FSIZE",
            "FOOD",
            "HOUSING_WATER",
            "ACTRENT",
            "IMPUTED_RENT",
            *columns,
        ],
    )
    monthly = pd.DataFrame(
        {category: fies[names].sum(axis=1) / 12 for category, names in FIES_COLUMNS.items()}
    )
    # Outside-home meals are restaurants_hotels; food keeps food eaten at home.
    monthly["food"] = (fies["FOOD"] - fies["FOOD_OUTSIDE"]).clip(lower=0) / 12
    # Owner-occupiers' imputed rent is in both income and housing but is never paid.
    monthly["housing_rent"] = fies["ACTRENT"] / 12
    monthly["housing_utilities"] = (
        fies["HOUSING_WATER"] - fies["IMPUTED_RENT"] - fies["ACTRENT"]
    ).clip(lower=0) / 12
    monthly["income"] = (fies["TOINC"] - fies["IMPUTED_RENT"]).clip(lower=0) / 12
    monthly["fsize"] = fies["FSIZE"]
    monthly["rfact"] = fies["RFACT"]
    return monthly


def seasonal_multipliers(hfce_path: Path, quarter: int) -> dict[str, float]:
    """Return each HFCE purpose's observed FIES_YEAR quarter / its FIES_YEAR quarterly mean."""
    hfce = pd.read_csv(hfce_path)
    year = hfce[hfce.year == FIES_YEAR]
    levels = year.pivot_table(
        index="category", columns="quarter", values="current_price_million_php", aggfunc="sum"
    )
    return (levels[quarter] / levels.mean(axis=1)).to_dict()


def _floors(
    categories: list[str],
    household: pd.Series,
    baseline: dict[str, float],
    target: dict[str, float],
    essential: frozenset[str],
    profile: str,
) -> dict[str, float]:
    """Return the MINIMUM floor of each non-rent Essential under a profile."""
    members = [c for c in categories if c in essential and c != FIXED_CATEGORY]
    if profile != SUBSISTENCE:
        return {c: round(target[c] * FLOOR_SHARES[profile], 2) for c in members}

    size = float(household["fsize"])
    floors = dict.fromkeys(members, 0.0)
    if "food" in floors:
        floors["food"] = round(size * FOOD_THRESHOLD, 2)
    pooled = size * (POVERTY_THRESHOLD - FOOD_THRESHOLD)
    non_food = [c for c in SUBSISTENCE_NON_FOOD if c in members]
    total = sum(baseline[c] for c in non_food)
    for c in non_food:
        share = baseline[c] / total if total > 0 else 1 / len(non_food)
        floors[c] = round(pooled * share, 2)
    return floors


def build_request(
    request_id: str,
    household: pd.Series,
    shock: float,
    profile: str,
    *,
    multipliers: dict[str, float] | None = None,
    essential: frozenset[str] = ESSENTIAL,
) -> BudgetRequest:
    categories = [c for c in LP_CATEGORIES if c in household.index]
    baseline = {c: float(household[c]) for c in categories}
    multipliers = multipliers or {}
    target = {c: round(baseline[c] * multipliers.get(LP_CATEGORIES[c], 1.0), 2) for c in categories}
    funds = round(float(household["income"]) * shock, 2)
    floors = _floors(categories, household, baseline, target, essential, profile)

    weights: dict[str, float] = {}
    for is_essential in (True, False):
        members = [c for c in categories if (c in essential) == is_essential]
        total = sum(baseline[c] for c in members)
        for c in members:
            weights[c] = baseline[c] / total if total > 0 else 1 / len(members)

    rows = []
    for category in categories:
        rule, floor = "FLEXIBLE", 0.0
        if category == FIXED_CATEGORY:
            rule, floor = "FIXED", target[category]
        elif floors.get(category, 0.0) > 0:
            rule, floor = "MINIMUM", floors[category]
        rows.append(
            {
                "category_id": category,
                "target_type": "category",
                "priority_group": "ESSENTIAL" if category in essential else "NON_ESSENTIAL",
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
        period=PLAN_PERIOD,
        categories=rows,
        forecast={
            "month": PLAN_PERIOD["start"][:7],
            "total_amount": sum(target.values()),
            "category_forecasts": [
                {"category": category, "amount": amount} for category, amount in target.items()
            ],
            "model_version": "fies-2025-realised",
            "generated_at": "2026-10-18T00:00:00Z",
            "quality": "PERSONALIZED",
        },
    )


def evaluate_request(request: BudgetRequest) -> dict:
    """Score one LP recommendation on the four budget recommendation metrics."""
    recommendation, _ = optimize(request)
    categories = request.categories
    funds = request.available_funds
    forecast = {point.category: point.amount for point in request.forecast.category_forecasts}
    # T_i is the target itself; it may exceed a category's ceiling.
    target = {category.category_id: forecast[category.category_id] for category in categories}
    row = {"feasible": recommendation.feasibility == "FEASIBLE"}
    if not row["feasible"]:
        return row

    lp = {allocation.category_id: allocation.amount for allocation in recommendation.allocations}
    essential = [category for category in categories if category.priority_group == "ESSENTIAL"]
    non_essential = [category for category in categories if category.priority_group != "ESSENTIAL"]
    below_target = any(lp[c.category_id] < target[c.category_id] - TOL for c in categories)
    idle = funds - sum(lp.values()) if below_target else 0.0
    essential_target = sum(target[c.category_id] for c in essential)
    essential_short = any(
        lp[c.category_id] < target[c.category_id] - TOL and lp[c.category_id] < c.ceiling - TOL
        for c in essential
    )
    non_essential_above_floor = any(lp[c.category_id] > c.floor + TOL for c in non_essential)

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
        idle_funds_rate=max(idle, 0.0) / funds,
        essential_coverage=sum(min(lp[c.category_id], target[c.category_id]) for c in essential)
        / essential_target
        if essential_target > 0
        else 1.0,
        priority_respected=not (essential_short and non_essential_above_floor),
    )
    return row


def summarize(frame: pd.DataFrame) -> dict:
    feasible = frame[frame.feasible]
    metrics = {
        "plans": len(frame),
        "feasible_plans": len(feasible),
        "feasibility_rate": float(frame.feasible.mean()) if len(frame) else 0.0,
    }
    if len(feasible):
        metrics.update(
            constraint_satisfaction_rate=float(feasible.constraints_satisfied.mean()),
            idle_funds_rate=float(feasible.idle_funds_rate.mean()),
            essential_coverage=float(feasible.essential_coverage.mean()),
            priority_validation_rate=float(feasible.priority_respected.mean()),
        )
    return metrics


def run(
    households: pd.DataFrame,
    variants: dict[str, dict],
    *,
    n_households: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Score every variant x profile x shock cell on the same sampled households.

    Each variant is {"multipliers": ..., "essential": ...}.
    """
    rng = np.random.default_rng(seed)
    weights = households.rfact / households.rfact.sum()
    sample = households.iloc[rng.choice(len(households), size=n_households, p=weights)]
    sample = sample.reset_index(drop=True)
    # Deciles of unshocked monthly cash income across the sampled households.
    deciles = pd.qcut(sample.income.rank(method="first"), min(10, len(sample)), labels=False) + 1
    rows = [
        {
            "variant": name,
            "profile": profile,
            "income_shock": shock,
            "income_decile": int(deciles[i]),
            **evaluate_request(
                build_request(
                    str(i),
                    household,
                    shock,
                    profile,
                    multipliers=variant["multipliers"],
                    essential=variant["essential"],
                )
            ),
        }
        for name, variant in variants.items()
        for profile in PROFILES
        for shock in INCOME_SHOCKS
        for i, household in sample.iterrows()
    ]
    frame = pd.DataFrame(rows)

    def cell(name: str, profile: str, shock: float) -> pd.DataFrame:
        return frame[
            (frame.variant == name) & (frame.profile == profile) & (frame.income_shock == shock)
        ]

    report = {
        "task": "budget_optimization",
        "solver": "scipy.optimize.linprog (HiGHS), two-stage LP",
        "source": "FIES 2025 public-use file (RFACT-weighted, cash income), realised 2025 targets",
        "seed": seed,
        "households": n_households,
        "fies_households": len(households),
        "poverty_threshold_monthly_per_capita": POVERTY_THRESHOLD,
        "food_threshold_monthly_per_capita": FOOD_THRESHOLD,
        "income_shocks": list(INCOME_SHOCKS),
        "profiles": list(PROFILES),
        "variants": {
            name: {
                "multipliers": variant["multipliers"],
                "essential": sorted(variant["essential"]),
                "cells": {
                    f"{profile}_income_x{shock}": summarize(cell(name, profile, shock))
                    for profile in PROFILES
                    for shock in INCOME_SHOCKS
                },
            }
            for name, variant in variants.items()
        },
        "feasibility_by_income_decile": {
            f"{profile}_income_x{shock}": {
                str(decile): float(group.feasible.mean())
                for decile, group in cell(next(iter(variants)), profile, shock).groupby(
                    "income_decile"
                )
            }
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
        default=Path("RAW/uncleaned/2025/Volume 1/FIES_PUF_2025_Volume1.CSV"),
    )
    parser.add_argument(
        "--hfce",
        type=Path,
        default=Path("training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("training/datasets/budget_evaluation"))
    parser.add_argument("--households", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    households = load_households(args.fies)
    variants = {
        "base": {"multipliers": {}, "essential": ESSENTIAL},
        "q4_seasonal_stress": {
            "multipliers": seasonal_multipliers(args.hfce, 4),
            "essential": ESSENTIAL,
        },
        "psa_basket_essentials": {"multipliers": {}, "essential": ESSENTIAL_PSA_BASKET},
    }
    frame, report = run(households, variants, n_households=args.households, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "plans.csv", index=False)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["variants"]["base"]["cells"], indent=2))


if __name__ == "__main__":
    main()
