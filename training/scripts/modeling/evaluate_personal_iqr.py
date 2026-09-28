"""Evaluate the served personal IQR rule on a held-out synthetic corpus only."""

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

from app.schemas.anomaly import AnomalyTransaction, DecisionState  # noqa: E402
from app.services.anomaly_service import RULESET_VERSION, detect  # noqa: E402


def _decision_states(frame: pd.DataFrame) -> dict[str, DecisionState]:
    states: dict[str, DecisionState] = {}
    for _, group in frame.groupby(["user_id", "category"], sort=False):
        transactions = [
            AnomalyTransaction(
                transaction_id=str(row.transaction_id),
                date=str(row.timestamp),
                amount=float(row.modified_amount),
                category=str(row.category),
                transaction_type="expense",
            )
            for row in group.itertuples(index=False)
        ]
        states.update(
            {decision.transaction_id: decision.state for decision in detect(transactions)}
        )
    return states


def build_anomaly_test(
    transactions: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    positive_rate: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Chronologically reserve and inject a clean, independently labeled test copy."""
    required = {
        "transaction_id",
        "date",
        "amount",
        "category",
        "transaction_type",
        "is_anomalous",
    }
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions missing required columns: {sorted(missing)}")
    if not 0 < test_fraction < 1 or not 0 <= positive_rate <= 1:
        raise ValueError("test_fraction must be in (0, 1) and positive_rate in [0, 1]")

    user_column = "user_id" if "user_id" in transactions else "persona_id"
    if transactions["is_anomalous"].fillna(False).astype(bool).any():
        raise ValueError("input must be generated with --no-anomalies")
    clean = transactions.loc[
        (transactions["transaction_type"] == "expense") & (transactions["amount"] > 0)
    ].copy()
    clean["user_id"] = clean[user_column].astype(str)
    clean["timestamp"] = pd.to_datetime(clean["date"], utc=True, errors="raise")
    clean["transaction_id"] = clean["transaction_id"].astype(str)
    clean = clean.sort_values(["user_id", "category", "timestamp", "transaction_id"])
    if clean.empty:
        raise ValueError("input contains no clean positive expense transactions")
    if clean["transaction_id"].duplicated().any():
        raise ValueError("transaction_id values must be globally unique")

    held_out = []
    for _, group in clean.groupby(["user_id", "category"], sort=False):
        split_index = max(1, int(np.floor(len(group) * (1 - test_fraction))))
        held_out.append(group.iloc[split_index:])
    test = pd.concat(held_out, ignore_index=True)
    if test.empty:
        raise ValueError("input has no held-out transactions; increase per-category history")
    test["source_id"] = test["transaction_id"]
    test["original_amount"] = test["amount"].astype(float)
    test["modified_amount"] = test["original_amount"]
    test["factor"] = 1.0
    test["injection_type"] = "none"
    test["expected_label"] = 0

    # Determine eligibility before intervention, using only the clean chronological corpus.
    clean_for_scoring = clean.assign(
        modified_amount=clean["amount"].astype(float),
    )
    states = _decision_states(clean_for_scoring)
    eligible_ids = test.loc[
        test["transaction_id"].map(states).eq(DecisionState.NORMAL), "transaction_id"
    ].to_numpy()
    if positive_rate > 0 and not len(eligible_ids):
        raise ValueError("input has no eligible held-out transactions for amount-spike injection")
    selected_count = (
        min(len(eligible_ids), max(1, int(np.floor(len(eligible_ids) * positive_rate))))
        if positive_rate > 0
        else 0
    )
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(eligible_ids, size=selected_count, replace=False).tolist())
    selected_mask = test["transaction_id"].isin(selected)
    factors = rng.uniform(3.0, 5.0, size=int(selected_mask.sum()))
    test.loc[selected_mask, "factor"] = factors
    test.loc[selected_mask, "modified_amount"] *= factors
    test.loc[selected_mask, "injection_type"] = "amount_spike"
    test.loc[selected_mask, "expected_label"] = 1

    intervention = clean.merge(
        test[["transaction_id", "modified_amount"]], on="transaction_id", how="left"
    )
    intervention["modified_amount"] = intervention["modified_amount"].fillna(intervention["amount"])
    predicted = _decision_states(intervention)
    test["decision_state"] = test["transaction_id"].map(predicted).map(str)
    test["is_scoreable"] = test["decision_state"].isin(
        [DecisionState.ANOMALOUS.value, DecisionState.NORMAL.value]
    )
    scoreable = test.loc[test["is_scoreable"]]
    expected = scoreable["expected_label"].to_numpy(dtype=int)
    prediction = scoreable["decision_state"].eq(DecisionState.ANOMALOUS.value).to_numpy(dtype=int)
    true_positive = int(((prediction == 1) & (expected == 1)).sum())
    false_positive = int(((prediction == 1) & (expected == 0)).sum())
    false_negative = int(((prediction == 0) & (expected == 1)).sum())
    true_negative = int(((prediction == 0) & (expected == 0)).sum())

    def metric(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    report = {
        "ruleset_version": RULESET_VERSION,
        "dataset": "anomaly_test",
        "positive_label": "controlled 3x-5x amount_spike intervention",
        "sample_size": int(len(scoreable)),
        "held_out_sample_size": int(len(test)),
        "injected_positive_rate": round(float(test["expected_label"].mean()), 4),
        "metrics": {
            "accuracy": metric(true_positive + true_negative, len(scoreable)),
            "precision": metric(true_positive, true_positive + false_positive),
            "recall": metric(true_positive, true_positive + false_negative),
            "f1": metric(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        },
        "undefined_metrics": [
            name
            for name, value in {
                "accuracy": metric(true_positive + true_negative, len(scoreable)),
                "precision": metric(true_positive, true_positive + false_positive),
                "recall": metric(true_positive, true_positive + false_negative),
                "f1": metric(
                    2 * true_positive, 2 * true_positive + false_positive + false_negative
                ),
            }.items()
            if value is None
        ],
        "insufficient_state_coverage": test["decision_state"].value_counts().to_dict(),
    }
    return test, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean v2 transactions parquet")
    parser.add_argument(
        "--output", default="training/datasets/anomaly_test", help="Evaluation output"
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--positive-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    test, report = build_anomaly_test(
        pd.read_parquet(args.input),
        test_fraction=args.test_fraction,
        positive_rate=args.positive_rate,
        seed=args.seed,
    )
    test.to_parquet(output / "anomaly_test.parquet", index=False)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
