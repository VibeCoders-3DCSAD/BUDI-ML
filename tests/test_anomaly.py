from __future__ import annotations

from math import exp

import numpy as np
from app.schemas.anomaly import AnomalyRequest, AnomalyTransaction, DecisionState
from app.services.anomaly_service import detect

from tests.conftest import trusted_post


def transaction(
    identifier: str,
    amount: float,
    date: str,
    category: str = "food",
    transaction_type: str = "expense",
) -> AnomalyTransaction:
    return AnomalyTransaction(
        transaction_id=identifier,
        amount=amount,
        date=date,
        category=category,
        transaction_type=transaction_type,
    )


def eligible_history(amounts: list[float] | None = None) -> list[AnomalyTransaction]:
    amounts = amounts or [100, 110, 120, 130, 140, 150]
    dates = ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15", "2026-03-01", "2026-03-15"]
    return [
        transaction(f"history-{index}", amount, dates[index])
        for index, amount in enumerate(amounts)
    ]


def api_transactions(count: int = 20) -> list[dict]:
    return [
        transaction(
            f"api-{index}",
            100 + index,
            f"2026-{1 + index // 5:02d}-{1 + index % 5:02d}",
        ).model_dump()
        for index in range(count)
    ]


def test_personal_iqr_detects_upper_and_lower_outliers_without_candidate_leakage():
    decisions = detect(
        eligible_history()
        + [transaction("upper", 1000, "2026-04-01"), transaction("lower", 1, "2026-04-02")]
    )
    by_id = {decision.transaction_id: decision for decision in decisions}
    assert by_id["upper"].state == DecisionState.ANOMALOUS
    assert by_id["lower"].state == DecisionState.ANOMALOUS
    assert by_id["upper"].usable_history_count == 6
    assert by_id["upper"].historical_median == 125
    assert by_id["upper"].usual_spending == 125
    assert by_id["upper"].severity is not None and by_id["upper"].severity > 0


def test_personal_iqr_tie_ordering_is_deterministic():
    history = eligible_history()
    first = transaction("a", 160, "2026-04-01")
    second = transaction("b", 160, "2026-04-01")
    decisions = detect(history + [second, first])
    by_id = {decision.transaction_id: decision for decision in decisions}
    assert by_id["a"].usable_history_count == 6
    assert by_id["b"].usable_history_count == 7
    assert by_id["a"].state == DecisionState.NORMAL


def test_personal_iqr_fence_equality_is_normal():
    history = eligible_history()
    values = np.log1p([item.amount for item in history])
    q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
    lower = float(q1 - 1.5 * (q3 - q1))
    upper = float(q3 + 1.5 * (q3 - q1))
    lower_decision = detect(
        history + [transaction("at-lower-fence", exp(lower) - 1, "2026-04-01")]
    )[-1]
    upper_decision = detect(
        history + [transaction("at-upper-fence", exp(upper) - 1, "2026-04-01")]
    )[-1]
    assert lower_decision.state == DecisionState.NORMAL
    assert upper_decision.state == DecisionState.NORMAL


def test_personal_iqr_reports_insufficient_history_and_variability():
    history = eligible_history([100] * 6)
    decisions = detect(history + [transaction("candidate", 300, "2026-04-01")])
    by_id = {decision.transaction_id: decision for decision in decisions}
    assert by_id["history-5"].state == DecisionState.INSUFFICIENT_HISTORY
    assert by_id["candidate"].state == DecisionState.INSUFFICIENT_VARIABILITY
    assert by_id["candidate"].lower_bound is None
    assert by_id["candidate"].severity is None


def test_personal_iqr_isolates_categories_and_ignores_income():
    history = eligible_history() + [
        transaction("income", 99999, "2026-03-20", transaction_type="income")
    ]
    decisions = detect(
        history + [transaction("transport", 1000, "2026-04-01", category="transport")]
    )
    by_id = {decision.transaction_id: decision for decision in decisions}
    assert "income" not in by_id
    assert by_id["transport"].state == DecisionState.INSUFFICIENT_HISTORY
    assert by_id["transport"].usable_history_count == 0


def test_personal_iqr_adapts_after_flagged_expenses():
    decisions = detect(
        eligible_history()
        + [
            transaction("spike", 1000, "2026-04-01"),
            transaction("later", 1000, "2026-04-02"),
        ]
    )
    by_id = {decision.transaction_id: decision for decision in decisions}
    assert by_id["spike"].state == DecisionState.ANOMALOUS
    assert by_id["later"].usable_history_count == 7


def test_anomaly_detect_uses_trusted_identity_header(client):
    payload = {"transactions": api_transactions()}
    response = trusted_post(client, "/api/v1/anomaly/detect", payload)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test-user-1"
    assert body["ruleset_version"] == "personal-transaction-iqr-v2.3.0"
    assert len(body["decisions"]) > 0
    assert all(
        item["state"] in {state.value for state in DecisionState} for item in body["decisions"]
    )
    scored_decision = next(
        item
        for item in body["decisions"]
        if item["state"] in {DecisionState.ANOMALOUS, DecisionState.NORMAL}
    )
    assert scored_decision["lower_bound"] is not None
    assert scored_decision["upper_bound"] is not None
    assert scored_decision["historical_median"] is not None
    assert scored_decision["usual_spending"] is not None


def test_anomaly_rejects_untrusted_identity_and_malformed_transactions(client):
    payload = {"transactions": api_transactions()}
    assert client.post("/api/v1/anomaly/detect", json=payload).status_code == 401
    payload["transactions"][0].pop("transaction_id")
    response = trusted_post(client, "/api/v1/anomaly/detect", payload)
    assert response.status_code == 422


def test_anomaly_rejects_tampered_history_payload(client):
    payload = {"transactions": api_transactions()}
    response = trusted_post(client, "/api/v1/anomaly/detect", payload)
    assert response.status_code == 200
    payload["transactions"][-1]["amount"] = 999999
    response = client.post(
        "/api/v1/anomaly/detect",
        json=payload,
        headers={
            "X-Odin-User-Id": "test-user-1",
            "X-Odin-History-Source": "odin_api",
            "X-Odin-History-Signature": "not-a-valid-signature",
        },
    )
    assert response.status_code == 401


def test_anomaly_blocks_scoring_without_trusted_history_configuration(client, monkeypatch):
    monkeypatch.delenv("ODIN_TRUSTED_HISTORY_SHARED_SECRET")
    response = trusted_post(client, "/api/v1/anomaly/detect", {"transactions": api_transactions()})
    assert response.status_code == 503


def test_anomaly_user_scopes_are_isolated_at_the_signed_gateway_boundary(client):
    first = trusted_post(
        client, "/api/v1/anomaly/detect", {"transactions": api_transactions()}, "user-a"
    )
    second = trusted_post(
        client, "/api/v1/anomaly/detect", {"transactions": api_transactions()}, "user-b"
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["user_id"] == "user-a"
    assert second.json()["user_id"] == "user-b"


def test_anomaly_detect_batch_and_overspending(client):
    transactions = api_transactions(60)
    payload = {
        "requests": [
            {"transactions": transactions[:30]},
            {
                "transactions": transactions[30:],
                "detection_type": "OVERSPENDING",
                "budget_allocations": [{"category_id": "food", "budget_amount": 1.0}],
            },
        ]
    }
    response = trusted_post(client, "/api/v1/anomaly/detect/batch", payload)
    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["decisions"]
    assert second["decisions"] == []
    assert any(item["category"] == "food" for item in second["overspending_transactions"])


def test_request_rejects_noncanonical_category_and_timezone_less_timestamp():
    base = {
        "transaction_id": "one",
        "amount": 10,
        "category": "Food",
        "transaction_type": "expense",
    }
    try:
        AnomalyRequest(transactions=[base | {"date": "2026-01-01"}])
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical category was accepted")
    try:
        AnomalyRequest(transactions=[base | {"category": "food", "date": "2026-01-01T12:00:00"}])
    except ValueError:
        pass
    else:
        raise AssertionError("timezone-less timestamp was accepted")
