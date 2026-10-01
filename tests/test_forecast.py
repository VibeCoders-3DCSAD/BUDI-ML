from __future__ import annotations

import logging
from datetime import date, timedelta

from app.models.registry import ModuleModel
from app.services import forecast_service

from tests.conftest import trusted_post


class _StubHfceModel:
    class _Inner:
        endog = [100.0, 110.0, 90.0, 100.0]

    model = _Inner()

    def forecast(self, _steps):
        return [120.0]


def test_forecast_returns_only_the_next_calendar_month_from_one_completed_month(caplog):
    caplog.set_level(logging.INFO, logger="app.services.forecast_service")
    model = ModuleModel(
        module="hfce_sarima_v4",
        model={
            "food": {
                "model": _StubHfceModel(),
                "transform": "identity",
                "period_end": "2026Q3",
            }
        },
        evaluation={},
        feature_columns=[],
    )
    transactions = [
        {
            "date": "2026-01-10",
            "amount": 2_500.0,
            "category": "food",
            "transaction_type": "expense",
        },
        {"date": "2026-10-02", "amount": 800.0, "category": "food", "transaction_type": "expense"},
    ]

    points = forecast_service.forecast(
        model,
        transactions,
        ["food"],
        as_of=date(2026, 10, 7),
        user_id="test-user",
    )

    assert [point.month for point in points] == ["2026-11"]
    assert points[0].quarter == "2026Q4"
    assert points[0].user_baseline_amount == 2_500.0
    assert points[0].hfce_multiplier == 1.2
    assert points[0].amount == 3_000.0
    assert "completed_months=['2026-01'] monthly_amounts_php=[2500.0] recency_weights=[1]" in caplog.text
    assert "weighted_total_php=2500.00 weight_total=1 baseline_php=2500.00" in caplog.text
    assert "user_id=test-user category=food baseline_php=2500.00" in caplog.text
    assert "hfce_multiplier=1.200000 trend_multiplier=1.000000 final_amount_php=3000.00" in caplog.text


def test_forecast_excludes_partial_month_spend_from_its_baseline():
    model = ModuleModel(
        module="hfce_sarima_v4",
        model={
            "food": {
                "model": _StubHfceModel(),
                "transform": "identity",
                "period_end": "2026Q3",
            }
        },
        evaluation={},
        feature_columns=[],
    )
    transactions = [
        {
            "date": "2026-01-10",
            "amount": 1_000.0,
            "category": "food",
            "transaction_type": "expense",
        },
        {
            "date": "2026-02-10",
            "amount": 3_000.0,
            "category": "food",
            "transaction_type": "expense",
        },
        {
            "date": "2026-10-02",
            "amount": 100_000.0,
            "category": "food",
            "transaction_type": "expense",
        },
    ]

    points = forecast_service.forecast(model, transactions, ["food"], as_of=date(2026, 10, 3))

    assert points[0].user_baseline_amount == 2_333.33
    assert points[0].month == "2026-11"


def test_predict_returns_personalized_hfce_category_forecast(client):
    payload = {
        "user_id": "test-user-hfce",
        "categories": ["food"],
        "historical_transactions": [
            {
                "date": "2025-10-10",
                "amount": 1_000.0,
                "category": "food",
                "transaction_type": "expense",
            },
            {
                "date": "2026-01-10",
                "amount": 1_200.0,
                "category": "food",
                "transaction_type": "expense",
            },
            {
                "date": "2026-04-10",
                "amount": 1_300.0,
                "category": "food",
                "transaction_type": "expense",
            },
        ],
    }

    response = trusted_post(client, "/api/v1/forecast/predict", payload, "test-user-hfce")

    assert response.status_code == 200
    point = response.json()["forecasts"][0]
    target_month = (date.today().replace(day=1) + timedelta(days=32)).replace(day=1)
    assert point["category"] == "food"
    assert point["month"] == target_month.strftime("%Y-%m")
    assert point["quarter"] == f"{target_month.year}Q{((target_month.month - 1) // 3) + 1}"
    assert point["amount"] > 0
    assert point["user_baseline_amount"] == 1_266.67
    assert point["hfce_multiplier"] > 0
    snapshot = response.json()["snapshot"]
    assert snapshot["month"] == point["month"]
    assert snapshot["total_amount"] == point["amount"]
    assert snapshot["category_forecasts"] == [{"category": "food", "amount": point["amount"]}]
    assert snapshot["model_version"] == response.json()["model_version"]
    assert snapshot["generated_at"].endswith("Z") or "+00:00" in snapshot["generated_at"]
    assert snapshot["quality"] == "PERSONALIZED"


def test_predict_rejects_history_without_completed_current_year_month(client):
    payload = {
        "user_id": "test-user-hfce",
        "categories": ["food"],
        "historical_transactions": [
            {
                "date": date.today().replace(day=1).isoformat(),
                "amount": 999.0,
                "category": "food",
                "transaction_type": "expense",
            }
        ],
    }

    response = trusted_post(client, "/api/v1/forecast/predict", payload, "test-user-hfce")

    assert response.status_code == 422
    assert "completed current-year" in response.json()["detail"]


def test_legacy_hfce_endpoint_is_removed(client):
    response = client.post("/api/v1/forecast/hfce-quarterly")

    assert response.status_code == 404


def test_registry_does_not_load_v3_random_forest(monkeypatch):
    from app.models import registry as registry_module

    class _Loader:
        calls = []

        def load_json(self, module, name):
            self.calls.append((module, name))
            if (module, name) == ("forecaster_v2", "evaluation.json"):
                return {"winner": "tier3_sarima", "winner_artifact": "tier3_sarima.joblib"}
            if (module, name) == ("forecaster_v2", "metadata.json"):
                return {}
            raise FileNotFoundError

        def resolve(self, module):
            assert module == "forecaster_v2"
            from pathlib import Path

            return Path("models/forecaster_v2")

    monkeypatch.setattr(
        registry_module.ModelLoader,
        "load_joblib",
        lambda _self, _module, _artifact: {"kind": "sarima"},
    )
    loader = _Loader()
    registry_module.ModelRegistry(loader=loader)._load_forecaster()

    assert all(module != "forecaster_v3" for module, _ in loader.calls)
