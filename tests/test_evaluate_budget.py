import pandas as pd
import pytest
from training.scripts.modeling.evaluate_budget import (
    ESSENTIAL_PSA_BASKET,
    FOOD_THRESHOLD,
    INCOME_SHOCKS,
    POVERTY_THRESHOLD,
    PROFILES,
    build_request,
    evaluate_request,
    load_households,
    run,
    seasonal_multipliers,
)


def _household(income: float, **spend: float) -> pd.Series:
    return pd.Series(
        {
            "food": 10_000.0,
            "housing_rent": 3_000.0,
            "housing_utilities": 2_000.0,
            "restaurants_hotels": 2_000.0,
            **spend,
            "income": income,
            "fsize": 4,
            "rfact": 1.0,
        }
    )


def _by_id(request):
    return {category.category_id: category for category in request.categories}


def test_scenario_grid_matches_documented_design():
    assert PROFILES == ("floor_0", "floor_50", "floor_70", "floor_90", "subsistence")
    assert INCOME_SHOCKS == (1.0, 0.93, 0.5)


def test_floor_0_matches_app_default_except_fixed_rent():
    by_id = _by_id(build_request("1", _household(30_000), 1.0, "floor_0"))

    assert by_id["housing_rent"].restriction_level == "FIXED"
    assert all(
        category.restriction_level == "FLEXIBLE" and category.floor == 0
        for category_id, category in by_id.items()
        if category_id != "housing_rent"
    )


def test_income_shock_scales_available_funds_only():
    request = build_request("1", _household(30_000), 0.5, "floor_0")
    targets = {point.category: point.amount for point in request.forecast.category_forecasts}

    assert request.available_funds == 15_000
    assert targets["food"] == 10_000


def test_seasonal_multiplier_scales_targets_and_shared_housing():
    request = build_request(
        "1",
        _household(30_000),
        1.0,
        "floor_0",
        multipliers={"food": 1.2, "housing_water_utilities": 1.1},
    )
    targets = {point.category: point.amount for point in request.forecast.category_forecasts}

    assert targets["food"] == 12_000
    assert targets["housing_rent"] == 3_300
    assert targets["housing_utilities"] == 2_200
    assert targets["restaurants_hotels"] == 2_000


def test_psa_basket_variant_moves_communication_to_non_essential():
    household = _household(30_000, communication=500.0)
    by_id = _by_id(build_request("1", household, 1.0, "floor_70", essential=ESSENTIAL_PSA_BASKET))

    assert by_id["communication"].priority_group == "NON_ESSENTIAL"
    assert by_id["communication"].floor == 0


def test_load_households_removes_imputed_rent_and_outside_food(tmp_path):
    fies = tmp_path / "fies.csv"
    row = dict.fromkeys(
        [
            "ALCOHOL",
            "TOBACCO",
            "CLOTH",
            "FURNISHING",
            "HEALTH",
            "TRANSPORT",
            "COMMUNICATION",
            "RECREATION",
            "EDUCATION",
            "MISCELLANEOUS",
        ],
        0,
    )
    row.update(
        TOINC=120_000,
        RFACT=1.0,
        FSIZE=3,
        ACTRENT=0,
        IMPUTED_RENT=24_000,
        HOUSING_WATER=36_000,
        FOOD=60_000,
        FOOD_OUTSIDE=12_000,
    )
    pd.DataFrame([row]).to_csv(fies, index=False)

    household = load_households(fies).iloc[0]

    assert household["income"] == 8_000
    assert household["housing_rent"] == 0
    assert household["housing_utilities"] == 1_000
    assert household["food"] == 4_000
    assert household["restaurants_hotels"] == 1_000


def test_seasonal_multipliers_use_observed_quarter_over_year_mean(tmp_path):
    hfce = tmp_path / "hfce.csv"
    pd.DataFrame(
        {
            "year": [2025] * 4,
            "quarter": [1, 2, 3, 4],
            "category": ["food"] * 4,
            "current_price_million_php": [90.0, 100.0, 100.0, 110.0],
        }
    ).to_csv(hfce, index=False)

    assert seasonal_multipliers(hfce, 4) == {"food": pytest.approx(1.1)}


def test_build_request_targets_realised_spend_and_fixes_rent():
    request = build_request("1", _household(30_000), 1.0, "floor_70")
    by_id = {category.category_id: category for category in request.categories}
    targets = {point.category: point.amount for point in request.forecast.category_forecasts}

    assert targets["food"] == 10_000
    assert by_id["housing_rent"].restriction_level == "FIXED"
    assert by_id["housing_rent"].floor == 3_000
    assert by_id["housing_utilities"].floor == 1_400
    assert by_id["food"].floor == 7_000
    assert by_id["food"].ceiling == 30_000
    assert by_id["restaurants_hotels"].priority_group == "NON_ESSENTIAL"
    assert by_id["restaurants_hotels"].priority_weight == 1.0


def test_build_request_subsistence_floors_follow_poverty_thresholds():
    household = _household(
        30_000, transport=1_000.0, communication=500.0, furnishings_household=400.0
    )
    by_id = _by_id(build_request("1", household, 1.0, "subsistence"))
    pooled = 4 * (POVERTY_THRESHOLD - FOOD_THRESHOLD)

    assert by_id["food"].floor == pytest.approx(4 * FOOD_THRESHOLD, abs=0.01)
    # The non-food pool is split by baseline share: utilities 2,000, transport 1,000.
    assert by_id["housing_utilities"].floor == pytest.approx(pooled * 2 / 3, abs=0.01)
    assert by_id["transport"].floor == pytest.approx(pooled / 3, abs=0.01)
    # Outside the PSA basic-needs basket: no subsistence floor.
    assert by_id["communication"].restriction_level == "FLEXIBLE"
    assert by_id["furnishings_household"].restriction_level == "FLEXIBLE"
    assert by_id["housing_rent"].restriction_level == "FIXED"


def test_evaluate_request_scores_surplus_plan():
    row = evaluate_request(build_request("1", _household(30_000), 1.0, "floor_0"))

    assert row["feasible"] and row["constraints_satisfied"]
    assert row["idle_funds_rate"] == 0.0
    assert row["essential_coverage"] == 1.0
    assert row["priority_respected"]


def test_evaluate_request_funds_essentials_first_when_short():
    row = evaluate_request(build_request("1", _household(13_000), 1.0, "floor_0"))

    assert row["feasible"] and row["constraints_satisfied"]
    assert row["idle_funds_rate"] == 0.0
    assert row["essential_coverage"] < 1.0
    assert row["priority_respected"]


def test_evaluate_request_skips_metrics_for_infeasible_plan():
    row = evaluate_request(build_request("1", _household(5_000), 1.0, "floor_90"))

    assert row == {"feasible": False}


def test_run_reports_every_cell_with_feasibility():
    households = pd.DataFrame([_household(30_000), _household(12_000)])
    variants = {"base": {"multipliers": {}, "essential": frozenset({"food"})}}
    _frame, report = run(households, variants, n_households=4, seed=1)

    cells = report["variants"]["base"]["cells"]
    assert len(cells) == 15
    assert cells["floor_0_income_x1.0"]["plans"] == 4
    assert cells["floor_0_income_x1.0"]["feasibility_rate"] == 1.0
    assert cells["floor_0_income_x1.0"]["constraint_satisfaction_rate"] == 1.0
    assert len(report["feasibility_by_income_decile"]) == 15
