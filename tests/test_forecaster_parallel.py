from __future__ import annotations

import time

import pandas as pd
from training.scripts.pipeline.feature_engineering_forecaster import process_personas


def test_parallel_persona_processing_matches_serial_output():
    transactions = pd.DataFrame(
        [
            {
                "persona_id": persona_id,
                "transaction_id": f"txn-{persona_id}",
                "date": pd.Timestamp("2023-01-15"),
                "amount": 100.0,
                "transaction_type": "expense",
            }
            for persona_id in ["persona-1", "persona-2"]
        ]
    )
    summaries = pd.DataFrame(
        [
            {
                "persona_id": persona_id,
                "year_month": period,
                "total_expenses": expense,
            }
            for persona_id in ["persona-1", "persona-2"]
            for period, expense in [("2023-01", 100.0), ("2023-02", 200.0)]
        ]
    )
    persona_ids = ["persona-2", "persona-1"]
    groups = {}
    for persona_id, group in transactions.groupby("persona_id"):
        groups[persona_id] = group
    summary_groups = {}
    for persona_id, group in summaries.groupby("persona_id"):
        summary_groups[persona_id] = group

    serial = process_personas(persona_ids, groups, summary_groups, 1, "train", time.time())
    parallel = process_personas(persona_ids, groups, summary_groups, 2, "train", time.time())

    assert [frame["user_id"].iloc[0] for frame in parallel] == persona_ids
    for serial_frame, parallel_frame in zip(serial, parallel, strict=True):
        pd.testing.assert_frame_equal(serial_frame, parallel_frame)
