from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt

from app.models.registry import ModuleModel
from app.schemas.forecast import ForecastPoint
from app.services.features import transactions_to_frame


def _inverse_hfce_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype="float64")
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"unknown HFCE target transform: {transform}")


def _monthly_baseline(
    expenses: pd.DataFrame, category: str, as_of: date
) -> tuple[float, np.ndarray]:
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
    return float(np.average(values, weights=weights)), values


def _monthly_weights(values: np.ndarray, baseline: float, horizon: int) -> np.ndarray:
    if len(values) == 1 or horizon == 0:
        return np.ones(horizon, dtype="float64")
    try:
        predicted = np.asarray(
            Holt(values, damped_trend=True, initialization_method="estimated")
            .fit(optimized=True)
            .forecast(horizon),
            dtype="float64",
        )
    except ValueError, np.linalg.LinAlgError:
        return np.ones(horizon, dtype="float64")
    return np.maximum(predicted, 0.0) / baseline


def forecast(
    model: ModuleModel, transactions: list[dict], categories: list[str], as_of: date | None = None
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
        user_baseline, monthly_history = _monthly_baseline(expenses, category, as_of)

        transform = artifact["transform"]
        national_history = _inverse_hfce_values(
            np.asarray(artifact["model"].model.endog, dtype="float64").reshape(-1), transform
        )
        national_baseline = float(np.median(national_history[-4:]))
        artifact_quarter = pd.Period(artifact["period_end"], freq="Q")
        forecast_steps = target_quarter.ordinal - artifact_quarter.ordinal
        if forecast_steps < 1:
            raise ValueError("forecast data is unavailable for the next calendar month")
        national_forecast = float(
            _inverse_hfce_values(np.asarray(artifact["model"].forecast(forecast_steps)), transform)[
                -1
            ]
        )
        multiplier = national_forecast / national_baseline
        trend_multiplier = _monthly_weights(monthly_history, user_baseline, 1)[0]
        points.append(
            ForecastPoint(
                category=category,
                month=str(target_month),
                quarter=str(target_quarter),
                amount=round(user_baseline * multiplier * trend_multiplier, 2),
                user_baseline_amount=round(user_baseline, 2),
                hfce_multiplier=round(multiplier, 6),
                hfce_forecast_amount_million_php=round(national_forecast, 2),
                explanation=f"Next-month {category} plan based on your monthly baseline and HFCE trend.",
            )
        )
    return points
