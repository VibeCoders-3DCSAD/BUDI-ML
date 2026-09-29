from __future__ import annotations

from datetime import UTC, datetime

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
                "restriction_level": "MINIMUM",
                "floor": 6000.0,
                "ceiling": 9000.0,
                "priority_weight": 0.4,
                "current_spend": 6800.0,
            },
            {
                "category_id": "essentials_rent",
                "restriction_level": "FIXED",
                "floor": 8000.0,
                "ceiling": 8000.0,
                "priority_weight": 0.3,
                "current_spend": 8000.0,
            },
            {
                "category_id": "discretionary_leisure",
                "restriction_level": "FLEXIBLE",
                "floor": 0.0,
                "ceiling": 5000.0,
                "priority_weight": 0.2,
                "current_spend": 2000.0,
            },
            {
                "category_id": "savings",
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
    assert abs(total - 25000.0) < 1.0
    assert allocations["essentials_rent"] == 8000.0  # FIXED
    assert rec["utilization_rate"] > 0.99
    assert rec["feasibility"] == "FEASIBLE"
    assert body["metadata"]["model_version"] == "budget-scipy_linprog"
    assert body["metadata"]["strategy_used"] == "tier2_lp"


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
