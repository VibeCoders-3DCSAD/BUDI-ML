import pandas as pd
from training.scripts.modeling.evaluate_budget import build_request, evaluate_request, run

GROWTH = {"food": 1.1, "housing_water_utilities": 1.0, "restaurants_hotels": 1.2}


def _household(income: float) -> pd.Series:
    return pd.Series(
        {
            "food": 10_000.0,
            "housing_water_utilities": 5_000.0,
            "restaurants_hotels": 2_000.0,
            "income": income,
            "rfact": 1.0,
        }
    )


def test_build_request_mirrors_odin_rules():
    request = build_request("1", _household(30_000), GROWTH, 1.0, 1.0, "mixed")
    by_id = {category.category_id: category for category in request.categories}

    assert by_id["housing_water_utilities"].restriction_level == "FIXED"
    assert by_id["housing_water_utilities"].floor == 5_000
    assert by_id["food"].restriction_level == "MINIMUM"
    assert by_id["food"].floor == 7_700
    assert by_id["food"].ceiling == 30_000
    assert by_id["restaurants_hotels"].priority_group == "NON_ESSENTIAL"
    assert by_id["restaurants_hotels"].priority_weight == 1.0


def test_evaluate_request_scores_surplus_plan():
    row = evaluate_request(build_request("1", _household(30_000), GROWTH, 1.0, 1.0, "flexible"))

    assert row["feasible"] and row["constraints_satisfied"]
    assert row["utilization_capacity"] == 1.0
    assert row["deviation_lp"] == 0.0


def test_evaluate_request_flags_truly_infeasible_plan():
    row = evaluate_request(build_request("1", _household(5_000), GROWTH, 1.0, 1.0, "strict"))

    assert not row["feasible"]
    assert row["truly_infeasible"]


def test_run_reports_every_scenario():
    households = pd.DataFrame([_household(30_000), _household(12_000)])
    _frame, report = run(households, GROWTH, 1.0, n_households=4, seed=1)

    assert len(report["scenarios"]) == 9
    assert report["overall"]["plans"] == 36
    assert report["overall"]["constraint_satisfaction_rate"] == 1.0
