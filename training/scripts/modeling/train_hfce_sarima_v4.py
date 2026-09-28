"""Train and evaluate one quarterly SARIMA model per observed HFCE category."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.scripts.data.build_hfce_category_forecast_dataset_v4 import PURPOSES, validate_dataset

LOGGER = logging.getLogger(__name__)
SEASONAL_PERIOD = 4
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.20
WALK_FORWARD_HORIZON = 4
BASE_ORDERS = (
    ("010_100", (0, 1, 0), (1, 0, 0, 4)),
    ("110_100", (1, 1, 0), (1, 0, 0, 4)),
    ("011_101", (0, 1, 1), (1, 0, 1, 4)),
)
TRANSFORMS = ("identity", "log1p")


@dataclass(frozen=True)
class FittedCandidate:
    result: Any
    family: str
    transform: str


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a regular series chronologically into 70%, 20%, and 10% partitions."""
    train_size = int(len(series) * TRAIN_SHARE)
    validation_size = int(len(series) * VALIDATION_SHARE)
    if (
        train_size < SEASONAL_PERIOD * 2
        or validation_size == 0
        or len(series) - train_size - validation_size == 0
    ):
        raise ValueError("HFCE series is too short for the configured 70/20/10 quarterly split")
    return (
        series.iloc[:train_size],
        series.iloc[train_size : train_size + validation_size],
        series.iloc[train_size + validation_size :],
    )


def error_metrics(
    actual: pd.Series, predicted: np.ndarray, previous_actual: float
) -> dict[str, float]:
    """Return error and directional metrics for a held-out quarterly forecast."""
    values = actual.to_numpy(dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    if len(values) != len(forecast):
        raise ValueError("Forecast length does not match the held-out series")
    error = values - forecast
    denominator = np.abs(values) + np.abs(forecast)
    smape_terms = np.divide(
        2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0
    )
    prior_values = np.concatenate(([previous_actual], values[:-1]))
    directional_matches = np.sign(values - prior_values) == np.sign(forecast - prior_values)
    return {
        "mae_million_php": float(np.abs(error).mean()),
        "rmse_million_php": float(np.sqrt(np.square(error).mean())),
        "mape_percent": float((np.abs(error) / values).mean() * 100),
        "smape_percent": float(smape_terms.mean() * 100),
        "mda_percent": float(directional_matches.mean() * 100),
    }


def average_metrics(metrics_by_category: list[dict[str, float]]) -> dict[str, float]:
    """Return an unweighted macro average across category-model metrics."""
    if not metrics_by_category:
        raise ValueError("Cannot average an empty collection of category metrics")
    metric_names = metrics_by_category[0].keys()
    if any(metrics.keys() != metric_names for metrics in metrics_by_category[1:]):
        raise ValueError("Category metric sets differ and cannot be averaged")
    return {
        metric: float(np.mean([metrics[metric] for metrics in metrics_by_category]))
        for metric in metric_names
    }


def candidates(model_family: str) -> tuple[dict[str, object], ...]:
    """Return raw/log candidates for one explicitly selected model family."""
    if model_family not in {"sarima", "sarimax_intervention"}:
        raise ValueError(f"Unknown model family: {model_family}")
    return tuple(
        {
            "name": f"{family}_{order_name}_{transform}",
            "family": family,
            "order": order,
            "seasonal_order": seasonal_order,
            "transform": transform,
        }
        for family in (model_family,)
        for transform in TRANSFORMS
        for order_name, order, seasonal_order in BASE_ORDERS
    )


def intervention_features(
    periods: pd.PeriodIndex,
    covid_start: pd.Period,
    covid_end: pd.Period,
    post_covid_start: pd.Period,
) -> pd.DataFrame:
    """Build temporary-shock and persistent-level-shift intervention variables."""
    return pd.DataFrame(
        {
            "covid": ((periods >= covid_start) & (periods <= covid_end)).astype("float64"),
            "post_covid": (periods >= post_covid_start).astype("float64"),
        },
        index=periods,
    )


def transform_series(series: pd.Series, transform: str) -> pd.Series:
    """Apply a candidate target transform before fitting."""
    if transform == "identity":
        return series
    if transform == "log1p":
        return np.log1p(series)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_forecast(values: np.ndarray, transform: str) -> np.ndarray:
    """Convert fitted-scale forecasts back to current-price million PHP."""
    if transform == "identity":
        return np.asarray(values, dtype="float64")
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unknown target transform: {transform}")


def fit_sarima(
    series: pd.Series,
    candidate: dict[str, object],
    interventions: tuple[pd.Period, pd.Period, pd.Period],
) -> FittedCandidate:
    """Fit SARIMA or SARIMAX while preserving the complete quarterly series."""
    covid_start, covid_end, post_covid_start = interventions
    transformed = transform_series(series, str(candidate["transform"]))
    exog = (
        intervention_features(series.index, covid_start, covid_end, post_covid_start)
        if candidate["family"] == "sarimax_intervention"
        else None
    )
    model = SARIMAX(
        transformed,
        order=candidate["order"],
        seasonal_order=candidate["seasonal_order"],
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    if not model.mle_retvals.get("converged", False):
        raise ValueError(f"{candidate['name']} did not converge")
    return FittedCandidate(model, str(candidate["family"]), str(candidate["transform"]))


def forecast_sarima(
    model: FittedCandidate,
    periods: pd.PeriodIndex,
    interventions: tuple[pd.Period, pd.Period, pd.Period],
) -> np.ndarray:
    """Forecast a target period and inverse-transform values before scoring."""
    covid_start, covid_end, post_covid_start = interventions
    exog = (
        intervention_features(periods, covid_start, covid_end, post_covid_start)
        if model.family == "sarimax_intervention"
        else None
    )
    return inverse_forecast(model.result.forecast(len(periods), exog=exog), model.transform)


def walk_forward_folds(
    train: pd.Series, validation: pd.Series
) -> list[tuple[pd.Series, pd.Series]]:
    """Expand history across validation in four-quarter forecast blocks."""
    folds = []
    history = train
    for start in range(0, len(validation), WALK_FORWARD_HORIZON):
        held_out = validation.iloc[start : start + WALK_FORWARD_HORIZON]
        folds.append((history, held_out))
        history = pd.concat([history, held_out])
    return folds


def select_candidate(
    train: pd.Series,
    validation: pd.Series,
    interventions: tuple[pd.Period, pd.Period, pd.Period],
    model_family: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Choose the lowest-MAPE candidate from expanding walk-forward validation."""
    results: list[dict[str, object]] = []
    for candidate in candidates(model_family):
        LOGGER.info("Evaluating %s", candidate["name"])
        try:
            actual_folds, forecast_folds, fold_periods = [], [], []
            for history, held_out in walk_forward_folds(train, validation):
                model = fit_sarima(history, candidate, interventions)
                actual_folds.append(held_out)
                forecast_folds.append(forecast_sarima(model, held_out.index, interventions))
                fold_periods.append(
                    {
                        "train": [str(history.index[0]), str(history.index[-1])],
                        "validation": [str(held_out.index[0]), str(held_out.index[-1])],
                    }
                )
            metrics = error_metrics(
                pd.concat(actual_folds),
                np.concatenate(forecast_folds),
                previous_actual=float(train.iloc[-1]),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            LOGGER.warning("Skipping %s: %s", candidate["name"], error)
            continue
        results.append(
            {
                "name": candidate["name"],
                "metrics": metrics,
                "params": candidate,
                "walk_forward_folds": fold_periods,
            }
        )
    if not results:
        raise ValueError("No SARIMA candidate could fit the validation split")
    winner = min(results, key=lambda result: result["metrics"]["mape_percent"])
    return winner["params"], {"candidates": results, "winner": winner}


def load_category_series(dataset: pd.DataFrame, category: str) -> pd.Series:
    """Return one category's ordered, quarterly current-price HFCE sequence."""
    category_rows = dataset.loc[dataset["category"].eq(category)].sort_values(["year", "quarter"])
    index = pd.PeriodIndex(category_rows["year_quarter"], freq="Q")
    return pd.Series(
        category_rows["current_price_million_php"].to_numpy(dtype="float64"),
        index=index,
        name=category,
    )


def train_models(
    dataset_path: str | Path,
    output_dir: str | Path,
    covid_start: str = "2020Q1",
    covid_end: str = "2021Q4",
    post_covid_start: str = "2022Q1",
    model_family: str = "sarima",
) -> dict[str, object]:
    """Evaluate, test, and refit SARIMA/SARIMAX candidates per category."""
    source = Path(dataset_path)
    dataset = pd.read_csv(source)
    validate_dataset(
        dataset, expected_categories=set(PURPOSES.values()), require_continuous_periods=True
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    interventions = tuple(
        pd.Period(period, freq="Q") for period in (covid_start, covid_end, post_covid_start)
    )
    if model_family not in {"sarima", "sarimax_intervention"}:
        raise ValueError(f"Unknown model family: {model_family}")
    if not interventions[0] <= interventions[1] < interventions[2]:
        raise ValueError("COVID intervention dates must satisfy start <= end < post-COVID start")
    report: dict[str, object] = {
        "family": "hfce_sarima_v4",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target": "current_price_million_php",
        "frequency": "quarterly",
        "seasonal_period": SEASONAL_PERIOD,
        "model_family_policy": f"{model_family}_for_all_categories",
        "split": {"train": TRAIN_SHARE, "validation": VALIDATION_SHARE, "test": 0.10},
        "selection_method": {
            "name": "expanding_window_walk_forward",
            "horizon_quarters": WALK_FORWARD_HORIZON,
            "selection_metric": "mape_percent",
        },
        "intervention_contract": {
            "covid": [str(interventions[0]), str(interventions[1])],
            "post_covid_start": str(interventions[2]),
            "covid_observations": "retained",
        },
        "categories": {},
    }
    for category in sorted(PURPOSES.values()):
        LOGGER.info("Training quarterly %s candidates for %s", model_family, category)
        series = load_category_series(dataset, category)
        train, validation, test = split_series(series)
        _, selection = select_candidate(train, validation, interventions, model_family)
        selected_candidate = None
        test_metrics = None
        final_model = None
        for candidate_result in sorted(
            selection["candidates"], key=lambda result: result["metrics"]["mape_percent"]
        ):
            candidate = candidate_result["params"]
            try:
                tested_model = fit_sarima(pd.concat([train, validation]), candidate, interventions)
                final_model = fit_sarima(series, candidate, interventions)
            except ValueError as error:
                LOGGER.warning("Skipping %s for final refit: %s", candidate["name"], error)
                continue
            selected_candidate = candidate
            selection["winner"] = candidate_result
            test_metrics = error_metrics(
                test,
                forecast_sarima(tested_model, test.index, interventions),
                previous_actual=float(validation.iloc[-1]),
            )
            break
        if selected_candidate is None or test_metrics is None or final_model is None:
            raise ValueError(f"No SARIMA candidate converged for final {category} refit")
        artifact = f"{category}.joblib"
        joblib.dump(
            {
                "category": category,
                "target": "current_price_million_php",
                "seasonal_period": SEASONAL_PERIOD,
                "period_start": str(series.index[0]),
                "period_end": str(series.index[-1]),
                "family": final_model.family,
                "transform": final_model.transform,
                "intervention_contract": report["intervention_contract"],
                "model": final_model.result,
            },
            destination / artifact,
        )
        report["categories"][category] = {
            "artifact": artifact,
            "split_periods": {
                "train": [str(train.index[0]), str(train.index[-1])],
                "validation": [str(validation.index[0]), str(validation.index[-1])],
                "test": [str(test.index[0]), str(test.index[-1])],
            },
            "selected_candidate": selected_candidate,
            "validation": selection,
            "test_metrics": test_metrics,
        }
        LOGGER.info("%s test MAPE: %.2f%%", category, test_metrics["mape_percent"])
    category_reports = report["categories"].values()
    report["metric_averages"] = {
        "method": "unweighted_macro_average_across_12_category_models",
        "validation": average_metrics(
            [
                category_report["validation"]["winner"]["metrics"]
                for category_report in category_reports
            ]
        ),
        "test": average_metrics(
            [category_report["test_metrics"] for category_report in category_reports]
        ),
    }
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Wrote %s category models and evaluation to %s", len(PURPOSES), destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/models/hfce_sarima_v4"))
    parser.add_argument("--covid-start", default="2020Q1")
    parser.add_argument("--covid-end", default="2021Q4")
    parser.add_argument("--post-covid-start", default="2022Q1")
    parser.add_argument(
        "--model-family", choices=("sarima", "sarimax_intervention"), default="sarima"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_models(
        args.input,
        args.output_dir,
        covid_start=args.covid_start,
        covid_end=args.covid_end,
        post_covid_start=args.post_covid_start,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()
