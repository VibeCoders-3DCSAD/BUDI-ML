from __future__ import annotations

from tests.conftest import load_transactions, trusted_post


def test_pfp_standard_classify(client):
    txns = load_transactions()
    payload = {
        "user_id": "test-user-1",
        "classification_mode": "STANDARD",
        "payload": {"historical_transactions": txns},
    }
    resp = trusted_post(client, "/api/v1/pfp/classify", payload)
    assert resp.status_code == 200
    classification = resp.json()["classification"]
    assert classification["status"] == "SUCCESS"
    assert (
        classification["prediction"].count("/") == 2 or classification["prediction"].count("_") == 2
    )


def test_pfp_questionnaire_classify(client):
    payload = {
        "user_id": "test-user-2",
        "classification_mode": "QUESTIONNAIRE",
        "payload": {
            "questionnaire_answers": {
                "income_variability": "variable",
                "obligation_level": "high",
                "emergency_runway": "low",
            }
        },
    }
    resp = trusted_post(client, "/api/v1/pfp/classify", payload, "test-user-2")
    assert resp.status_code == 200
    classification = resp.json()["classification"]
    assert classification["prediction"] == "VARIABLE_OBLIGATED_AT_RISK"
    assert classification["status"] == "SUCCESS"


def test_pfp_rejects_unsigned_and_mismatched_user_identity(client):
    payload = {
        "user_id": "test-user-2",
        "classification_mode": "QUESTIONNAIRE",
        "payload": {"questionnaire_answers": {}},
    }
    assert client.post("/api/v1/pfp/classify", json=payload).status_code == 401
    assert trusted_post(client, "/api/v1/pfp/classify", payload, "another-user").status_code == 401


def test_pfp_rejects_invalid_mode(client):
    payload = {
        "user_id": "test-user-3",
        "classification_mode": "ENSEMBLE",
        "payload": {"historical_transactions": []},
    }
    resp = trusted_post(client, "/api/v1/pfp/classify", payload, "test-user-3")
    assert resp.status_code == 422
