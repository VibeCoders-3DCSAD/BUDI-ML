from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from math import exp, log1p

import numpy as np

from app.schemas.anomaly import (
    AnomalyDecision,
    AnomalyTransaction,
    DecisionState,
    OverspendingTransaction,
)

RULESET_VERSION = "personal-transaction-iqr-v2.4.0"
MINIMUM_HISTORY = 6
IQR_MULTIPLIER = 1.5
logger = logging.getLogger(__name__)


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _has_three_calendar_months(
    history: list[AnomalyTransaction], candidate: AnomalyTransaction
) -> bool:
    earliest = _timestamp(history[0].date)
    current = _timestamp(candidate.date)
    return (current.year - earliest.year) * 12 + current.month - earliest.month >= 3


def _amount_bounds(lower: float, upper: float) -> tuple[float, float]:
    return max(0.0, exp(lower) - 1), max(0.0, exp(upper) - 1)


def detect(transactions: list[AnomalyTransaction]) -> list[AnomalyDecision]:
    """Score valid expenses against only their earlier same-category history."""
    ordered = sorted(
        transactions,
        key=lambda transaction: (_timestamp(transaction.date), transaction.transaction_id),
    )
    histories: dict[str, list[AnomalyTransaction]] = defaultdict(list)
    decisions: list[AnomalyDecision] = []
    state_counts: Counter[DecisionState] = Counter()
    skipped_non_expenses = 0

    for transaction in ordered:
        if transaction.transaction_type != "expense":
            skipped_non_expenses += 1
            continue

        history = histories[transaction.category]
        if len(history) < MINIMUM_HISTORY or not _has_three_calendar_months(history, transaction):
            decision = AnomalyDecision(
                transaction_id=transaction.transaction_id,
                category=transaction.category,
                usable_history_count=len(history),
                state=DecisionState.INSUFFICIENT_HISTORY,
                reason="More earlier expenses across three calendar months are required.",
            )
        else:
            values = np.log1p([item.amount for item in history])
            q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
            iqr = float(q3 - q1)
            if iqr == 0:
                decision = AnomalyDecision(
                    transaction_id=transaction.transaction_id,
                    category=transaction.category,
                    usable_history_count=len(history),
                    state=DecisionState.INSUFFICIENT_VARIABILITY,
                    reason="Earlier expenses do not have enough variability for IQR bounds.",
                )
            else:
                lower = float(q1 - IQR_MULTIPLIER * iqr)
                upper = float(q3 + IQR_MULTIPLIER * iqr)
                transformed_amount = log1p(transaction.amount)
                historical_median = float(np.median([item.amount for item in history]))
                state = (
                    DecisionState.ANOMALOUS
                    if transformed_amount < lower or transformed_amount > upper
                    else DecisionState.NORMAL
                )
                severity = max(lower - transformed_amount, 0.0, transformed_amount - upper) / iqr
                lower_amount, upper_amount = _amount_bounds(lower, upper)
                decision = AnomalyDecision(
                    transaction_id=transaction.transaction_id,
                    category=transaction.category,
                    usable_history_count=len(history),
                    state=state,
                    lower_bound=round(lower_amount, 2),
                    upper_bound=round(upper_amount, 2),
                    historical_median=round(historical_median, 2) if historical_median > 0 else None,
                    usual_spending=round(historical_median, 2) if historical_median > 0 else None,
                    severity=round(severity, 4),
                    reason=(
                        "This transaction is outside your usual spending range for this category."
                        if state is DecisionState.ANOMALOUS
                        else "This transaction is within your usual spending range for this category."
                    ),
                )
        decisions.append(decision)
        state_counts[decision.state] += 1
        if decision.state is DecisionState.ANOMALOUS:
            logger.info(
                "anomaly detected transaction_id=%s category=%s history_count=%d severity=%s",
                decision.transaction_id,
                decision.category,
                decision.usable_history_count,
                decision.severity,
            )
        # Flagged observations stay isolated; evaluation has no user recurring confirmations.
        if decision.state is not DecisionState.ANOMALOUS:
            history.append(transaction)
    logger.info(
        "anomaly scoring complete inputs=%d scored=%d skipped_non_expenses=%d anomalous=%d normal=%d insufficient_history=%d insufficient_variability=%d",
        len(transactions),
        len(decisions),
        skipped_non_expenses,
        state_counts[DecisionState.ANOMALOUS],
        state_counts[DecisionState.NORMAL],
        state_counts[DecisionState.INSUFFICIENT_HISTORY],
        state_counts[DecisionState.INSUFFICIENT_VARIABILITY],
    )
    return decisions


def overspending_detect(
    transactions: list[AnomalyTransaction],
    budget_allocations: list[dict],
) -> list[OverspendingTransaction]:
    """Rule-based overspending detection: category spend vs budget."""
    budget_map = {budget["category_id"]: budget["budget_amount"] for budget in budget_allocations}
    category_totals: dict[str, float] = defaultdict(float)
    for transaction in transactions:
        if transaction.transaction_type == "expense":
            category_totals[transaction.category] += transaction.amount

    results: list[OverspendingTransaction] = []
    for category, total in category_totals.items():
        budget = budget_map.get(category)
        if budget is not None and total > budget:
            for transaction in transactions:
                if transaction.category == category and transaction.transaction_type == "expense":
                    results.append(
                        OverspendingTransaction(
                            transaction_id=transaction.transaction_id,
                            budget_excess=round(transaction.amount * (1 - budget / total), 2),
                            category=category,
                        )
                    )
    return results
