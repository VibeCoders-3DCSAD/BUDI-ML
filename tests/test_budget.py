from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.conftest import trusted_post


def _budget_payload():
    return {
        "request_id": "req-budget-1",
        "user_id": "test-user-1",
        "available_funds": 25000.0,
        "period": {"start": "2024-01-01", "end": "2024-01-31"},
        "categories": [
            {
                "category_id": "essentials_food",
                "priority_group": "ESSENTIAL",
                "restriction_level": "MINIMUM",
                "floor": 6000.0,
                "ceiling": 9000.0,
                "priority_weight": 0.4,
                "current_spend": 6800.0,
            },
            {
                "category_id": "essentials_rent",
                "priority_group": "ESSENTIAL",
                "restriction_level": "FIXED",
                "floor": 8000.0,
                "ceiling": 8000.0,
                "priority_weight": 0.3,
                "current_spend": 8000.0,
            },
            {
                "category_id": "discretionary_leisure",
                "priority_group": "NON_ESSENTIAL",
                "restriction_level": "FLEXIBLE",
                "floor": 0.0,
                "ceiling": 5000.0,
                "priority_weight": 0.2,
                "current_spend": 2000.0,
            },
            {
                "category_id": "savings",
                "priority_group": "NON_ESSENTIAL",
                "restriction_level": "FLEXIBLE",
                "floor": 0.0,
                "ceiling": 6000.0,
                "priority_weight": 0.1,
                "current_spend": 0.0,
            },
        ],
        "target_ratios": {
            "essentials_food": 0.28,
            "essentials_rent": 0.32,
            "discretionary_leisure": 0.15,
            "savings": 0.25,
        },
    }


def test_budget_recommend(client):
    resp = trusted_post(client, "/api/v1/budget/recommend", _budget_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "req-budget-1"
    rec = body["recommendation"]
    allocations = {a["category_id"]: a["amount"] for a in rec["allocations"]}
    total = sum(allocations.values())
    assert abs(total - 24750.0) < 1.0
    assert rec["unallocated_surplus"] == 250.0
    assert allocations["essentials_rent"] == 8000.0  # FIXED
    assert rec["utilization_rate"] == 0.99
    assert rec["feasibility"] == "FEASIBLE"
    assert body["metadata"]["model_version"] == "budget-scipy_linprog"
    assert body["metadata"]["strategy_used"] == "hierarchical_lp_v2"


def test_budget_keeps_forecast_surplus_out_of_expense_categories(client):
    payload = _budget_payload()
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 15_000.0,
        "category_forecasts": [
            {"category": "essentials_food", "amount": 7_000.0},
            {"category": "essentials_rent", "amount": 8_000.0},
            {"category": "discretionary_leisure", "amount": 0.0},
            {"category": "savings", "amount": 0.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    allocations = {item["category_id"]: item["amount"] for item in recommendation["allocations"]}
    assert allocations == {
        "essentials_food": 7_000.0,
        "essentials_rent": 8_000.0,
        "discretionary_leisure": 0.0,
        "savings": 0.0,
    }
    assert recommendation["unallocated_surplus"] == 10_000.0


def test_budget_allocates_scarce_funds_across_essential_targets_by_weight(client):
    payload = _budget_payload()
    payload["available_funds"] = 10_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.8,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.2,
        },
        {
            "category_id": "recreation",
            "priority_group": "NON_ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 18_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 8_000.0},
            {"category": "housing", "amount": 6_000.0},
            {"category": "recreation", "amount": 4_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 8_000.0, "housing": 2_000.0, "recreation": 0.0}


def test_budget_does_not_starve_lower_weight_essential(client):
    payload = _budget_payload()
    payload["available_funds"] = 5_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.8,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.2,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 14_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 8_000.0},
            {"category": "housing", "amount": 6_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == pytest.approx({"food": 4210.526315789473, "housing": 789.4736842105264})


def test_budget_preserves_forecast_cap_centavos(client):
    payload = _budget_payload()
    payload["available_funds"] = 10_005.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 20_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 8_005.99,
        "category_forecasts": [{"category": "food", "amount": 8_005.99}],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["allocations"] == [{"category_id": "food", "amount": 8_005.99}]
    assert recommendation["unallocated_surplus"] == 1_999.01


def test_budget_keeps_whole_peso_forecast_caps(client):
    payload = _budget_payload()
    payload["available_funds"] = 3_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 2_000.0,
            "priority_weight": 1.0,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 2_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 2_170.0,
        "category_forecasts": [
            {"category": "food", "amount": 1_083.0},
            {"category": "housing", "amount": 1_087.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 1_083.0, "housing": 1_087.0}


def test_budget_preserves_forecast_cap_centavos_across_categories(client):
    payload = _budget_payload()
    payload["available_funds"] = 6_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 5_000.0,
            "priority_weight": 1.0,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 1_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 5_100.67,
        "category_forecasts": [
            {"category": "food", "amount": 4_460.57},
            {"category": "housing", "amount": 640.10},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 4_460.57, "housing": 640.1}


def test_budget_preserves_half_peso_forecast_caps(client):
    payload = _budget_payload()
    payload["available_funds"] = 10.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 1.5,
        "category_forecasts": [{"category": "food", "amount": 1.5}],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    assert response.json()["recommendation"]["allocations"] == [
        {"category_id": "food", "amount": 1.5}
    ]


def test_budget_rejects_bad_restriction(client):
    payload = _budget_payload()
    payload["categories"][0]["ceiling"] = 5000.0  # ceiling below the MINIMUM floor
    resp = trusted_post(client, "/api/v1/budget/recommend", payload)
    assert resp.status_code == 422


def test_budget_rejects_fixed_category_with_unequal_floor_and_ceiling(client):
    payload = _budget_payload()
    payload["categories"][1]["ceiling"] = 9_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 422
    assert "fixed amount requires equal floor and ceiling" in str(response.json()["detail"])


def test_budget_returns_structured_infeasibility(client):
    payload = _budget_payload()
    payload["available_funds"] = 5_000.0
    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["feasibility"] == "INFEASIBLE"
    assert recommendation["required_funds"] == 14_000.0
    assert recommendation["available_funds"] == 5_000.0
    assert recommendation["shortfall"] == 9_000.0
    assert "essentials_food" in recommendation["conflicting_categories"]
    assert recommendation["utilization_rate"] == 1.0


def test_budget_uses_configured_fixed_amount_not_current_spend(client):
    payload = _budget_payload()
    payload["categories"][1]["current_spend"] = 1_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations["essentials_rent"] == 8_000.0


def test_budget_keeps_zero_flexible_allocation_rows(client):
    payload = _budget_payload()
    payload["available_funds"] = 14_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations["discretionary_leisure"] == 0.0
    assert allocations["savings"] == 0.0


def test_budget_rejects_forecast_for_another_plan_month(client):
    payload = _budget_payload()
    payload["forecast"] = {
        "month": "2024-02",
        "total_amount": 17_000.0,
        "category_forecasts": [
            {"category": "essentials_food", "amount": 7_000.0},
            {"category": "essentials_rent", "amount": 8_000.0},
            {"category": "discretionary_leisure", "amount": 1_000.0},
            {"category": "savings", "amount": 1_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 422
    assert "forecast month must match" in str(response.json()["detail"])
