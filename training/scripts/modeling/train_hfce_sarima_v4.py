"""Train and evaluate one quarterly SARIMA model per observed HFCE category."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.scripts.data.build_hfce_category_forecast_dataset_v4 import PURPOSES, validate_dataset

LOGGER = logging.getLogger(__name__)
SEASONAL_PERIOD = 4
TRAIN_END = "2021Q4"
VALIDATION_START = "2022Q1"
VALIDATION_END = "2023Q4"
TEST_START = "2024Q1"
WALK_FORWARD_HORIZON = 4
MAX_NON_SEASONAL_ORDER = 3
MAX_SEASONAL_ORDER = 2
TRANSFORMS = ("identity", "log1p")


@dataclass(frozen=True)
class FittedCandidate:
    result: Any
    family: str
    transform: str


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split at fixed COVID-aware and untouched post-COVID boundaries."""
    train = series.loc[:TRAIN_END]
    validation = series.loc[VALIDATION_START:VALIDATION_END]
    test = series.loc[TEST_START:]
    if len(train) < SEASONAL_PERIOD * 2 or len(validation) < WALK_FORWARD_HORIZON or test.empty:
        raise ValueError("HFCE series does not cover the required 2000Q1-2026Q2 evaluation windows")
    return train, validation, test


def error_metrics(
    actual: pd.Series, predicted: np.ndarray, previous_actual: float
) -> dict[str, float]:
    """Return error and directional metrics for a held-out quarterly forecast."""
    values = actual.to_numpy(dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    if len(values) != len(forecast):
        raise ValueError("Forecast length does not match the held-out series")
    if not np.isfinite(forecast).all():
        raise ValueError("Forecast contains non-finite values")
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
    """Return the bounded SARIMA order grid for one explicitly selected family."""
    if model_family not in {"sarima", "sarimax_intervention"}:
        raise ValueError(f"Unknown model family: {model_family}")
    return tuple(
        {
            "name": (f"{model_family}_{p}{d}{q}_{P}{D}{Q}_{transform}"),
            "family": model_family,
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, SEASONAL_PERIOD),
            "transform": transform,
        }
        for p, d, q, P, D, Q, transform in product(
            range(MAX_NON_SEASONAL_ORDER + 1),
            range(2),
            range(MAX_NON_SEASONAL_ORDER + 1),
            range(MAX_SEASONAL_ORDER + 1),
            range(2),
            range(MAX_SEASONAL_ORDER + 1),
            TRANSFORMS,
        )
        # Differencing both the annual cycle and the level is rarely defensible
        # for a 106-observation quarterly series and needlessly removes signal.
        if not (d and D)
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
    maxiter: int = 75,
) -> FittedCandidate:
    """Fit SARIMA or SARIMAX while preserving the complete quarterly series."""
    covid_start, covid_end, post_covid_start = interventions
    transformed = transform_series(series, str(candidate["transform"]))
    exog = (
        intervention_features(series.index, covid_start, covid_end, post_covid_start)
        if candidate["family"] == "sarimax_intervention"
        else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            transformed,
            order=candidate["order"],
            seasonal_order=candidate["seasonal_order"],
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=maxiter)
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


def aicc(model: Any) -> float:
    """Return finite-sample AIC, used only to break validation-score ties."""
    observations = int(model.nobs)
    parameters = len(model.params)
    denominator = observations - parameters - 1
    if denominator <= 0:
        return float("inf")
    return float(model.aic + (2 * parameters * (parameters + 1) / denominator))


def residual_diagnostics(model: FittedCandidate) -> dict[str, object]:
    """Summarize a Ljung-Box residual-autocorrelation check for reporting."""
    residuals = np.asarray(model.result.resid, dtype="float64")
    residuals = residuals[np.isfinite(residuals)]
    lag = min(8, len(residuals) // 5)
    if lag < 1:
        return {"status": "not_run", "reason": "too_few_residuals"}
    result = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    p_value = float(result["lb_pvalue"].iloc[0])
    return {
        "test": "Ljung-Box",
        "lag": lag,
        "p_value": p_value,
        "status": "pass" if p_value >= 0.05 else "warning_autocorrelation_detected",
    }


def select_candidate(
    train: pd.Series,
    validation: pd.Series,
    interventions: tuple[pd.Period, pd.Period, pd.Period],
    model_family: str,
    category: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Choose by expanding-window 4-quarter sMAPE, then MAE and mean AICc."""
    results: list[dict[str, object]] = []
    candidate_grid = candidates(model_family)
    for position, candidate in enumerate(candidate_grid, start=1):
        LOGGER.info(
            "Training %s, candidate %s/%s: SARIMA%s%s, transform=%s",
            category,
            position,
            len(candidate_grid),
            candidate["order"],
            candidate["seasonal_order"],
            candidate["transform"],
        )
        try:
            actual_folds, forecast_folds, fold_periods, fold_aicc = [], [], [], []
            for history, held_out in walk_forward_folds(train, validation):
                model = fit_sarima(history, candidate, interventions)
                fold_aicc.append(aicc(model.result))
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
            LOGGER.debug("Skipping %s: %s", candidate["name"], error)
            continue
        results.append(
            {
                "name": candidate["name"],
                "metrics": metrics,
                "mean_aicc": float(np.mean(fold_aicc)),
                "params": candidate,
                "walk_forward_folds": fold_periods,
            }
        )
    if not results:
        raise ValueError("No SARIMA candidate could fit the validation split")
    ranked = sorted(
        results,
        key=lambda result: (
            result["metrics"]["smape_percent"],
            result["metrics"]["mae_million_php"],
            result["mean_aicc"],
        ),
    )
    return ranked[0]["params"], {
        "evaluated_candidate_count": len(results),
        "rejected_candidate_count": len(candidate_grid) - len(results),
        "leaderboard": ranked[:10],
        "winner": ranked[0],
        "selection_rule": "lowest rolling 4-quarter sMAPE; ties resolved by MAE then mean AICc",
    }


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
    categories_to_train: set[str] | None = None,
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
        "split": {
            "train": [str(TRAIN_END)],
            "validation": [str(VALIDATION_START), str(VALIDATION_END)],
            "test": [str(TEST_START), "latest_available"],
        },
        "selection_method": {
            "name": "expanding_window_walk_forward",
            "horizon_quarters": WALK_FORWARD_HORIZON,
            "selection_metric": "smape_percent",
            "tie_breakers": ["mae_million_php", "mean_aicc"],
        },
        "search_space": {
            "non_seasonal": {"p": [0, 3], "d": [0, 1], "q": [0, 3]},
            "seasonal": {"P": [0, 2], "D": [0, 1], "Q": [0, 2], "s": 4},
            "transforms": list(TRANSFORMS),
            "excluded": ["d=1 and D=1"],
            "candidate_count_per_category": len(candidates(model_family)),
        },
        "intervention_contract": {
            "covid": [str(interventions[0]), str(interventions[1])],
            "post_covid_start": str(interventions[2]),
            "covid_observations": "retained",
        },
        "categories": {},
    }
    categories = sorted(categories_to_train or set(PURPOSES.values()))
    unknown_categories = set(categories) - set(PURPOSES.values())
    if unknown_categories:
        raise ValueError(f"Unknown HFCE categories requested: {sorted(unknown_categories)}")
    for category in categories:
        LOGGER.info("Training quarterly %s candidates for %s", model_family, category)
        series = load_category_series(dataset, category)
        train, validation, test = split_series(series)
        _, selection = select_candidate(train, validation, interventions, model_family, category)
        selected_candidate = None
        test_metrics = None
        final_model = None
        for candidate_result in sorted(
            selection["leaderboard"],
            key=lambda result: (
                result["metrics"]["smape_percent"],
                result["metrics"]["mae_million_php"],
                result["mean_aicc"],
            ),
        ):
            candidate = candidate_result["params"]
            try:
                tested_model = fit_sarima(pd.concat([train, validation]), candidate, interventions)
                final_model = fit_sarima(series, candidate, interventions)
            except ValueError as error:
                LOGGER.debug("Skipping %s for final refit: %s", candidate["name"], error)
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
            "residual_diagnostics": residual_diagnostics(final_model),
        }
        LOGGER.info("%s test MAPE: %.2f%%", category, test_metrics["mape_percent"])
    category_reports = report["categories"].values()
    report["metric_averages"] = {
        "method": f"unweighted_macro_average_across_{len(categories)}_category_models",
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
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(PURPOSES.values()),
        help="Train only these categories; used to parallelize a full grid run.",
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
        categories_to_train=set(args.categories) if args.categories else None,
    )


if __name__ == "__main__":
    main()
