"""Select, test, and release quarterly HFCE forecasting candidates per category."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.scripts.data.build_hfce_category_forecast_dataset_v4 import PURPOSES, validate_dataset

LOGGER = logging.getLogger(__name__)
SEASONAL_PERIOD = 4
TRAIN_END = "2016Q4"
VALIDATION_START = "2017Q1"
VALIDATION_END = "2023Q4"
TEST_START = "2024Q1"
WALK_FORWARD_HORIZON = 4
MAX_NON_SEASONAL_ORDER = 3
MAX_SEASONAL_ORDER = 2
TRANSFORMS = ("identity", "log1p")
PSA_DATA_VINTAGE = "2026-08-07"
MODEL_VERSION = "hfce-sarima-v4"
MAX_TEST_MAE_DETERIORATION_RATIO = 0.10


@dataclass(frozen=True)
class FittedCandidate:
    result: Any
    family: str
    transform: str


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split the fixed initial, validation, and untouched test periods."""
    train = series.loc[:TRAIN_END]
    validation = series.loc[VALIDATION_START:VALIDATION_END]
    test = series.loc[TEST_START:]
    if len(train) != 68 or len(validation) != 28 or len(test) != 10:
        raise ValueError("HFCE series must cover 2000Q1-2026Q2 evaluation windows")
    return train, validation, test


def error_metrics(
    actual: pd.Series | np.ndarray, predicted: np.ndarray, previous_actual: float | np.ndarray
) -> dict[str, float]:
    """Calculate all metrics on the original current-price million-PHP scale."""
    values = np.asarray(actual, dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    prior_values = np.broadcast_to(np.asarray(previous_actual, dtype="float64"), values.shape)
    if len(values) != len(forecast):
        raise ValueError("Forecast length does not match the held-out series")
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("Forecast contains invalid values")
    error = values - forecast
    denominator = np.abs(values) + np.abs(forecast)
    smape_terms = np.divide(
        2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0
    )
    directional_matches = np.sign(values - prior_values) == np.sign(forecast - prior_values)
    return {
        "mae_million_php": float(np.abs(error).mean()),
        "smape_percent": float(smape_terms.mean() * 100),
        "mda_percent": float(directional_matches.mean() * 100),
        "rmse_million_php": float(np.sqrt(np.square(error).mean())),
    }


def average_metrics(metrics_by_category: list[dict[str, float]]) -> dict[str, float]:
    """Return scale-independent macro averages only."""
    if not metrics_by_category:
        raise ValueError("Cannot average an empty collection of category metrics")
    return {
        metric: float(np.mean([metrics[metric] for metrics in metrics_by_category]))
        for metric in ("smape_percent", "mda_percent")
    }


def candidates() -> tuple[dict[str, object], ...]:
    """Return benchmarks and the complete declared SARIMA candidate space."""
    benchmarks = (
        {"name": "seasonal_naive", "family": "seasonal_naive", "transform": "identity"},
        {
            "name": "ets_aan",
            "family": "ets",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": False,
            "transform": "identity",
        },
        {
            "name": "ets_aad",
            "family": "ets",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": True,
            "transform": "identity",
        },
        {
            "name": "ets_ann",
            "family": "ets",
            "trend": "add",
            "seasonal": None,
            "damped_trend": False,
            "transform": "identity",
        },
        {
            "name": "ets_aan_seasonal",
            "family": "ets",
            "trend": None,
            "seasonal": "add",
            "damped_trend": False,
            "transform": "identity",
        },
    )
    sarima = tuple(
        {
            "name": f"sarima_{p}{d}{q}_{P}{D}{Q}_{transform}",
            "family": "sarima",
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
        if not (d and D)
    )
    return benchmarks + sarima


def transform_series(series: pd.Series, transform: str) -> pd.Series:
    if transform == "identity":
        return series
    if transform == "log1p":
        return np.log1p(series)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_forecast(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype="float64")
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unknown target transform: {transform}")


def validate_forecast(values: np.ndarray) -> np.ndarray:
    """Reject forecast values that cannot represent current-price HFCE."""
    forecast = np.asarray(values, dtype="float64")
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("Forecast contains invalid values")
    return forecast


def has_finite_parameters(params: Any, candidate: dict[str, object]) -> bool:
    """Check numeric parameters from both SARIMA arrays and ETS parameter mappings."""
    if isinstance(params, dict):
        names = {"smoothing_level", "initial_level"}
        if candidate.get("trend"):
            names.update({"smoothing_trend", "initial_trend"})
        if candidate.get("damped_trend"):
            names.add("damping_trend")
        if candidate.get("seasonal"):
            names.update({"smoothing_seasonal", "initial_seasons"})
        values = (params[name] for name in names)
    else:
        values = (params,)
    for value in values:
        try:
            numeric = np.asarray(value, dtype="float64")
        except TypeError, ValueError:
            continue
        if not np.isfinite(numeric).all():
            return False
    return True


def fit_candidate(
    series: pd.Series, candidate: dict[str, object], maxiter: int = 75
) -> FittedCandidate:
    """Fit one candidate without using future observations or interventions."""
    family = str(candidate["family"])
    transform = str(candidate["transform"])
    if family == "seasonal_naive":
        return FittedCandidate(series.copy(), family, transform)
    transformed = transform_series(series, transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "ets":
            result = ExponentialSmoothing(
                transformed,
                trend=candidate["trend"],
                damped_trend=bool(candidate["damped_trend"]),
                seasonal=candidate["seasonal"],
                seasonal_periods=SEASONAL_PERIOD if candidate["seasonal"] else None,
            ).fit(optimized=True)
        elif family == "sarima":
            result = SARIMAX(
                transformed,
                order=candidate["order"],
                seasonal_order=candidate["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=maxiter)
            if not result.mle_retvals.get("converged", False):
                raise ValueError(f"{candidate['name']} did not converge")
        else:
            raise ValueError(f"Unknown model family: {family}")
    if not has_finite_parameters(result.params, candidate):
        raise ValueError(f"{candidate['name']} has non-finite parameters")
    return FittedCandidate(result, family, transform)


def forecast_candidate(model: FittedCandidate, horizon: int) -> np.ndarray:
    if model.family == "seasonal_naive":
        history = np.asarray(model.result, dtype="float64").tolist()
        values = []
        for _ in range(horizon):
            values.append(history[-SEASONAL_PERIOD])
            history.append(values[-1])
        return validate_forecast(np.asarray(values))
    return validate_forecast(inverse_forecast(model.result.forecast(horizon), model.transform))


def walk_forward_folds(
    train: pd.Series, validation: pd.Series
) -> list[tuple[pd.Series, pd.Series]]:
    """Create every expanding origin, with up to four validation quarters per origin."""
    return [
        (
            pd.concat([train, validation.iloc[:offset]]),
            validation.iloc[offset : offset + WALK_FORWARD_HORIZON],
        )
        for offset in range(len(validation))
    ]


def candidate_rank_key(result: dict[str, object]) -> tuple[float, float, float, float]:
    """Rank by MAE, then SMAPE, RMSE, and descending directional accuracy."""
    metrics = result["metrics"]
    return (
        float(metrics["mae_million_php"]),
        float(metrics["smape_percent"]),
        float(metrics["rmse_million_php"]),
        -float(metrics["mda_percent"]),
    )


def evaluate_candidate(
    candidate: dict[str, object], train: pd.Series, evaluation: pd.Series, horizon: int
) -> tuple[dict[str, float], list[dict[str, list[str]]]]:
    actual, predicted, previous = [], [], []
    folds = []
    for history, held_out in walk_forward_folds(train, evaluation):
        held_out = held_out.iloc[:horizon]
        model = fit_candidate(history, candidate)
        forecast = forecast_candidate(model, len(held_out))
        actual.extend(held_out.to_numpy(dtype="float64"))
        predicted.extend(forecast)
        # All horizons must use the actual level known at this forecast origin.
        previous.extend(np.repeat(float(history.iloc[-1]), len(held_out)))
        folds.append(
            {
                "train": [str(history.index[0]), str(history.index[-1])],
                "validation": [str(held_out.index[0]), str(held_out.index[-1])],
            }
        )
    return error_metrics(np.asarray(actual), np.asarray(predicted), np.asarray(previous)), folds


def select_candidate(
    train: pd.Series, validation: pd.Series, category: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Select one family/specification using validation data only."""
    results = []
    candidate_grid = candidates()
    for position, candidate in enumerate(candidate_grid, start=1):
        LOGGER.info(
            "Training %s candidate %s/%s: %s",
            category,
            position,
            len(candidate_grid),
            candidate["name"],
        )
        try:
            metrics, folds = evaluate_candidate(candidate, train, validation, WALK_FORWARD_HORIZON)
        except Exception as error:
            LOGGER.debug("Skipping %s: %s", candidate["name"], error)
            continue
        results.append(
            {
                "name": candidate["name"],
                "metrics": metrics,
                "params": candidate,
                "walk_forward_folds": folds,
            }
        )
    if not results:
        raise ValueError("No HFCE forecasting candidate could fit the validation split")
    ranked = sorted(results, key=candidate_rank_key)
    simple_ranked = [result for result in ranked if result["params"]["family"] != "sarima"]
    return ranked[0]["params"], {
        "evaluated_candidate_count": len(results),
        "rejected_candidate_count": len(candidate_grid) - len(results),
        "leaderboard": ranked[:10],
        "winner": ranked[0],
        "locked_simple_fallback": simple_ranked[0] if simple_ranked else None,
        "selection_rule": "lowest rolling-origin MAE; ties resolved by SMAPE, RMSE, then MDA",
    }


def release_candidate(
    selected: dict[str, object],
    selection: dict[str, object],
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
) -> tuple[dict[str, object], dict[str, object], dict[str, float], list[dict[str, list[str]]]]:
    """Gate a SARIMA release against a validation-locked simple benchmark."""
    selected_metrics, selected_folds = evaluate_candidate(
        selected, pd.concat([train, validation]), test, 1
    )
    fallback = selection["locked_simple_fallback"]
    gate: dict[str, object] = {
        "passed": True,
        "maximum_test_mae_deterioration_ratio": MAX_TEST_MAE_DETERIORATION_RATIO,
        "selected_test_metrics": selected_metrics,
        "fallback": None,
    }
    if selected["family"] != "sarima" or fallback is None:
        return selected, gate, selected_metrics, selected_folds

    fallback_params = fallback["params"]
    fallback_metrics, _ = evaluate_candidate(
        fallback_params, pd.concat([train, validation]), test, 1
    )
    if fallback_metrics["mae_million_php"] == 0:
        deterioration = 0.0 if selected_metrics["mae_million_php"] == 0 else float("inf")
    else:
        deterioration = (
            selected_metrics["mae_million_php"] / fallback_metrics["mae_million_php"] - 1
        )
    gate["fallback"] = {
        "candidate": fallback_params,
        "test_metrics": fallback_metrics,
        "mae_deterioration_ratio": deterioration,
    }
    if deterioration > MAX_TEST_MAE_DETERIORATION_RATIO:
        gate["passed"] = False
        gate["reason"] = "SARIMA test MAE deteriorated beyond the locked simple benchmark threshold"
    return selected, gate, selected_metrics, selected_folds


def write_evaluation_report(destination: Path, report: dict[str, object]) -> None:
    """Write a concise report from the current run's evaluation contract."""
    rows = []
    for category, details in report["categories"].items():
        metrics = details["test"]["metrics"]
        rows.append(
            "| {category} | {family} | {mae:.2f} | {smape:.2f}% | {mda:.2f}% | {rmse:.2f} |".format(
                category=category,
                family=details["released_candidate"]["family"],
                mae=metrics["mae_million_php"],
                smape=metrics["smape_percent"],
                mda=metrics["mda_percent"],
                rmse=metrics["rmse_million_php"],
            )
        )
    lines = [
        "# HFCE v4 Evaluation Report",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}",
        "**Target:** `current_price_million_php`",
        "**Data vintage:** 2026-08-07",
        "",
        "## Evaluation Contract",
        "",
        "- Initial training: `2000Q1-2016Q4` (68 quarters).",
        "- Expanding rolling validation: `2017Q1-2023Q4`, up to four quarters per origin.",
        "- Untouched test: `2024Q1-2026Q2`, one-step expanding refits.",
        "- Candidates: seasonal naive, ETS, and SARIMA; ranking is MAE, SMAPE, RMSE, then MDA.",
        "- Metrics: MAE, SMAPE, MDA, and RMSE only.",
        "",
        "## Released Models",
        "",
        "| Category | Released family | Test MAE (million PHP) | Test SMAPE | Test MDA | Test RMSE (million PHP) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "The full candidate and release-gate records are in `evaluation.json`.",
    ]
    (destination / "evaluation_report.md").write_text("\n".join(lines) + "\n")


def load_category_series(dataset: pd.DataFrame, category: str) -> pd.Series:
    rows = dataset.loc[dataset["category"].eq(category)].sort_values(["year", "quarter"])
    return pd.Series(
        rows["current_price_million_php"].to_numpy(dtype="float64"),
        index=pd.PeriodIndex(rows["year_quarter"], freq="Q"),
        name=category,
    )


def train_models(
    dataset_path: str | Path, output_dir: str | Path, categories_to_train: set[str] | None = None
) -> dict[str, object]:
    """Lock validation winners, test them once, then refit artifacts on all observed data."""
    source = Path(dataset_path)
    dataset = pd.read_csv(source)
    validate_dataset(
        dataset, expected_categories=set(PURPOSES.values()), require_continuous_periods=True
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    categories = sorted(categories_to_train or set(PURPOSES.values()))
    if unknown := set(categories) - set(PURPOSES.values()):
        raise ValueError(f"Unknown HFCE categories requested: {sorted(unknown)}")
    report: dict[str, object] = {
        "family": "hfce_forecasting_v4",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target": "current_price_million_php",
        "psa_data_vintage": PSA_DATA_VINTAGE,
        "model_version": MODEL_VERSION,
        "preprocessing": {
            "source_frequency": "quarterly_observed",
            "target_representation": "current_price_million_php",
            "missing_value_treatment": "rejected",
        },
        "frequency": "quarterly",
        "seasonal_period": 4,
        "split": {
            "train": ["2000Q1", str(TRAIN_END)],
            "validation": [str(VALIDATION_START), str(VALIDATION_END)],
            "test": [str(TEST_START), "2026Q2"],
        },
        "selection_method": {
            "name": "expanding_window_rolling_origin",
            "horizon_quarters": 4,
            "selection_metric": "mae_million_php",
            "tie_breakers": ["smape_percent", "rmse_million_php", "mda_percent_descending"],
        },
        "search_space": {
            "families": ["seasonal_naive", "ets", "sarima"],
            "non_seasonal": {"p": [0, 3], "d": [0, 1], "q": [0, 3]},
            "seasonal": {"P": [0, 2], "D": [0, 1], "Q": [0, 2], "s": 4},
            "transforms": list(TRANSFORMS),
            "candidate_count_per_category": len(candidates()),
        },
        "covid_observations": "retained_without_intervention_variables",
        "categories": {},
    }
    for category in categories:
        series = load_category_series(dataset, category)
        train, validation, test = split_series(series)
        selected, selection = select_candidate(train, validation, category)
        released, release_gate, test_metrics, test_folds = release_candidate(
            selected, selection, train, validation, test
        )
        if not release_gate["passed"]:
            raise ValueError(f"{category} SARIMA candidate failed the final release gate")
        final_model = fit_candidate(series, released)
        forecast_candidate(final_model, WALK_FORWARD_HORIZON)
        artifact = f"{category}.joblib"
        joblib.dump(
            {
                "category": category,
                "target": "current_price_million_php",
                "seasonal_period": 4,
                "period_start": str(series.index[0]),
                "period_end": str(series.index[-1]),
                "psa_data_vintage": PSA_DATA_VINTAGE,
                "preprocessing": {
                    "source_frequency": "quarterly_observed",
                    "target_representation": "current_price_million_php",
                    "missing_value_treatment": "rejected",
                },
                "model_version": MODEL_VERSION,
                "family": final_model.family,
                "transform": final_model.transform,
                "specification": released,
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
            "selected_candidate": selected,
            "released_candidate": released,
            "validation": selection,
            "test": {"metrics": test_metrics, "walk_forward_folds": test_folds},
            "release_gate": release_gate,
        }
    reports = list(report["categories"].values())
    report["metric_averages"] = {
        "method": "scale-independent_macro_average_across_categories",
        "validation": average_metrics(
            [entry["validation"]["winner"]["metrics"] for entry in reports]
        ),
        "test": average_metrics([entry["test"]["metrics"] for entry in reports]),
    }
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_evaluation_report(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/models/hfce_sarima_v4"))
    parser.add_argument("--categories", nargs="+", choices=sorted(PURPOSES.values()))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_models(args.input, args.output_dir, set(args.categories) if args.categories else None)


if __name__ == "__main__":
    main()
