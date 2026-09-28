from __future__ import annotations

import pandas as pd
from training.scripts.modeling.evaluate_personal_iqr import build_anomaly_test


def test_evaluation_builds_held_out_provenance_and_metrics():
    rows = []
    for user in ("one", "two"):
        for index in range(16):
            rows.append(
                {
                    "transaction_id": f"{user}-{index}",
                    "date": f"2025-{1 + index // 4:02d}-{1 + index % 4:02d}",
                    "amount": 100 + index * 5,
                    "category": "food",
                    "transaction_type": "expense",
                    "persona_id": user,
                    "is_anomalous": False,
                }
            )
    test, report = build_anomaly_test(pd.DataFrame(rows), test_fraction=0.5, positive_rate=0.5)
    assert {
        "source_id",
        "original_amount",
        "modified_amount",
        "factor",
        "injection_type",
        "expected_label",
    } <= set(test)
    assert test["expected_label"].sum() > 0
    assert test.loc[test["expected_label"] == 1, "factor"].between(3, 5).all()
    assert set(report["metrics"]) == {"accuracy", "precision", "recall", "f1"}
    assert report["dataset"] == "anomaly_test"


def test_evaluation_rejects_pre_injected_anomalies():
    frame = pd.DataFrame(
        [
            {
                "transaction_id": "one",
                "date": "2025-01-01",
                "amount": 100,
                "category": "food",
                "transaction_type": "expense",
                "persona_id": "one",
                "is_anomalous": True,
            }
        ]
    )
    try:
        build_anomaly_test(frame)
    except ValueError as error:
        assert "--no-anomalies" in str(error)
    else:
        raise AssertionError("pre-injected anomaly input was accepted")


def test_evaluation_rejects_an_empty_held_out_set():
    frame = pd.DataFrame(
        [
            {
                "transaction_id": "one",
                "date": "2025-01-01",
                "amount": 100,
                "category": "food",
                "transaction_type": "expense",
                "persona_id": "one",
                "is_anomalous": False,
            }
        ]
    )
    try:
        build_anomaly_test(frame)
    except ValueError as error:
        assert "no held-out" in str(error)
    else:
        raise AssertionError("empty held-out corpus was accepted")


def test_evaluation_injects_at_least_one_eligible_positive():
    rows = [
        {
            "transaction_id": f"one-{index}",
            "date": f"2025-{1 + index // 2:02d}-{1 + index % 2:02d}",
            "amount": 100 + index * 10,
            "category": "food",
            "transaction_type": "expense",
            "persona_id": "one",
            "is_anomalous": False,
        }
        for index in range(10)
    ]
    test, _ = build_anomaly_test(pd.DataFrame(rows), test_fraction=0.2, positive_rate=0.1)
    assert test["expected_label"].sum() == 1
