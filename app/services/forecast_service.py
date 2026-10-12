from __future__ import annotations

import logging
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from app.models.registry import ModuleModel
from app.schemas.forecast import ForecastPoint
from app.services.features import transactions_to_frame

logger = logging.getLogger(__name__)


def _round_to_whole_peso(amount: float) -> float:
    return float(Decimal(str(amount)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _inverse_hfce_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype="float64")
    if transform == "log1p":
        return np.expm1(values)
    if transform == "log":
        return np.exp(values)
    raise ValueError(f"unknown HFCE target transform: {transform}")


def _hfce_history_and_forecast(artifact: dict, forecast_steps: int) -> tuple[np.ndarray, float]:
    """Forecast either a statsmodels artifact or the retained seasonal-naive baseline."""
    if artifact.get("family", "sarima") != "seasonal_naive":
        transform = artifact["transform"]
        history = _inverse_hfce_values(
            np.asarray(artifact["model"].model.endog, dtype="float64").reshape(-1), transform
        )
        forecast = _inverse_hfce_values(
            np.asarray(artifact["model"].forecast(forecast_steps)), transform
        )
        return history, float(forecast[-1])

    history = np.asarray(artifact["model"], dtype="float64").tolist()
    for _ in range(forecast_steps):
        history.append(history[-4])
    return np.asarray(history[:-forecast_steps]), float(history[-1])


def _monthly_baseline(
    expenses: pd.DataFrame, category: str, as_of: date
) -> tuple[float, np.ndarray, list[str]]:
    current_month = pd.Period(as_of, freq="M")
    completed = expenses.loc[
        expenses["date"].dt.to_period("M").lt(current_month)
        & expenses["date"].dt.year.eq(as_of.year)
        & expenses["category"].eq(category)
    ]
    monthly = completed.groupby(completed["date"].dt.to_period("M"))["amount"].sum()
    if monthly.empty:
        raise ValueError(f"forecast requires a completed current-year {category} month")
    values = monthly.to_numpy(dtype="float64")
    weights = np.arange(1, len(values) + 1, dtype="float64")
    return (
        float(np.average(values, weights=weights)),
        values,
        [str(month) for month in monthly.index],
    )


def forecast(
    model: ModuleModel,
    transactions: list[dict],
    categories: list[str],
    as_of: date | None = None,
    user_id: str | None = None,
) -> list[ForecastPoint]:
    """Project each HFCE category for the next calendar month only."""
    as_of = as_of or date.today()
    frame = transactions_to_frame(transactions)
    expenses = frame.loc[
        frame["transaction_type"].eq("expense") & frame["date"].dt.date.le(as_of)
    ].copy()
    if expenses.empty:
        raise ValueError("forecast requires at least one expense transaction")

    target_month = pd.Period(as_of, freq="M") + 1
    target_quarter = target_month.asfreq("Q")
    points = []
    for category in dict.fromkeys(categories):
        artifact = model.model.get(category)
        if artifact is None:
            raise ValueError(f"forecast does not support category: {category}")
        user_baseline, monthly_history, baseline_months = _monthly_baseline(
            expenses, category, as_of
        )
        baseline_weights = np.arange(1, len(monthly_history) + 1, dtype="float64")
        weighted_total = float(np.dot(monthly_history, baseline_weights))
        logger.info(
            "forecast baseline user_id=%s category=%s completed_months=%s "
            "monthly_amounts_php=%s recency_weights=%s weighted_total_php=%.2f "
            "weight_total=%.0f baseline_php=%.2f",
            user_id,
            category,
            baseline_months,
            [round(float(amount), 2) for amount in monthly_history],
            [int(weight) for weight in baseline_weights],
            weighted_total,
            baseline_weights.sum(),
            user_baseline,
        )

        artifact_quarter = pd.Period(artifact["period_end"], freq="Q")
        forecast_steps = target_quarter.ordinal - artifact_quarter.ordinal
        if forecast_steps < 1:
            raise ValueError("forecast data is unavailable for the next calendar month")
        national_history, national_forecast = _hfce_history_and_forecast(artifact, forecast_steps)
        national_baseline = float(np.median(national_history[-4:]))
        multiplier = national_forecast / national_baseline
        amount = user_baseline * multiplier
        logger.info(
            "forecast prediction user_id=%s category=%s baseline_php=%.2f "
            "hfce_multiplier=%.6f final_amount_php=%.2f",
            user_id,
            category,
            user_baseline,
            multiplier,
            amount,
        )
        points.append(
            ForecastPoint(
                category=category,
                month=str(target_month),
                quarter=str(target_quarter),
                amount=_round_to_whole_peso(amount),
                user_baseline_amount=round(user_baseline, 2),
                hfce_multiplier=round(multiplier, 6),
                hfce_forecast_amount_million_php=round(national_forecast, 2),
                explanation=f"Next-month {category} plan based on your monthly baseline and HFCE trend.",
            )
        )
    return points
