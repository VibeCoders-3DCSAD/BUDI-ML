"""Leakage-safe pooled HFCE LSTM/RF experiment against local SARIMA and seasonal naive.

Run with the repository virtual environment; see --help. Production artifacts are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.scripts.data.build_hfce_category_forecast_dataset_v4 import PURPOSES, validate_dataset
from training.scripts.modeling.train_hfce_sarima_v4 import (
    error_metrics,
    inverse_forecast,
    split_series,
    transform_series,
    validate_forecast,
)

LOGGER = logging.getLogger(__name__)
SEED = 42
TRAIN_SIZE, VALIDATION_END, TEST_END = 68, 96, 106
HORIZON = 4


class GlobalLSTM(nn.Module):
    """A shared recurrent network with a scalar seasonal-growth output."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(values)
        return self.output(sequence[:, -1]).squeeze(-1)


@dataclass
class FittedGlobal:
    config: dict[str, Any]
    categories: list[str]
    model: Any
    scaler: StandardScaler
    target_scaler: StandardScaler


def candidate_grid() -> list[dict[str, Any]]:
    """Small, declared search space; epochs are fixed, never test-stopped."""
    return [
        {"family": "random_forest", "window": window, "min_samples_leaf": leaf, "trees": 200}
        for window, leaf in ((8, 2), (8, 5), (12, 2), (12, 5))
    ] + [
        {"family": "lstm", "window": window, "hidden_size": hidden, "epochs": epochs}
        for window, hidden, epochs in ((8, 16, 100), (8, 32, 200), (12, 16, 100), (12, 32, 200))
    ]


def make_features(history: pd.DataFrame, window: int) -> np.ndarray:
    """Return category-major sequences using only the supplied quarterly history.

    Each step has relative log level, QoQ growth, YoY growth, quarter sin/cos,
    next-quarter sin/cos, and one-hot category identity. Log levels relative to
    the last known level remove cross-category scale and deterministic inflation.
    """
    if window < 4 or len(history) < window + 4:
        raise ValueError("Features require a window >= 4 and four preceding quarters")
    values = history.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Log seasonal-growth features require finite positive HFCE")
    logs = np.log(values)
    positions = np.arange(len(history) - window, len(history))
    angle = 2 * np.pi * (history.index.quarter.to_numpy()[positions] - 1) / 4
    next_angle = 2 * np.pi * ((history.index[-1] + 1).quarter - 1) / 4
    categories = len(history.columns)
    sequences = []
    for category in range(categories):
        identity = np.broadcast_to(np.eye(categories)[category], (window, categories))
        sequences.append(
            np.column_stack(
                [
                    logs[positions, category] - logs[-1, category],
                    logs[positions, category] - logs[positions - 1, category],
                    logs[positions, category] - logs[positions - 4, category],
                    np.sin(angle),
                    np.cos(angle),
                    np.full(window, np.sin(next_angle)),
                    np.full(window, np.cos(next_angle)),
                    identity,
                ]
            )
        )
    return np.asarray(sequences, dtype="float32")


def make_samples(history: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled labeled windows with targets strictly inside the history."""
    if len(history) <= window + 4:
        raise ValueError("Insufficient history for any labeled window")
    features, targets = [], []
    for target in range(window + 4, len(history)):
        features.append(make_features(history.iloc[:target], window))
        targets.append(np.log(history.iloc[target].to_numpy() / history.iloc[target - 4]))
    return np.concatenate(features), np.concatenate(targets).astype("float32")


def fit_global(history: pd.DataFrame, config: dict[str, Any], seed: int) -> FittedGlobal:
    """Fit one model across all categories, with scalers fit on this origin only."""
    x, y = make_samples(history, config["window"])
    scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
    target_scaler = StandardScaler().fit(y[:, None])
    y = target_scaler.transform(y[:, None]).ravel().astype("float32")
    if config["family"] == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config["trees"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=seed,
            n_jobs=1,
        ).fit(x.reshape(len(x), -1), y)
    elif config["family"] == "lstm":
        torch.manual_seed(seed)
        model = GlobalLSTM(x.shape[-1], config["hidden_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
        model.train()
        for _ in range(config["epochs"]):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("LSTM training produced nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
    else:
        raise ValueError(f"Unknown pooled family: {config['family']}")
    return FittedGlobal(config, list(history.columns), model, scaler, target_scaler)


def forecast_global(fitted: FittedGlobal, history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Predict recursively, feeding forecasts rather than future observations back in."""
    if list(history.columns) != fitted.categories:
        raise ValueError("Forecast category order differs from fitted order")
    history = history.copy()
    predictions = []
    for _ in range(horizon):
        x = make_features(history, fitted.config["window"])
        x = fitted.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        if fitted.config["family"] == "random_forest":
            growth = fitted.model.predict(x.reshape(len(x), -1))
        else:
            with torch.no_grad():
                growth = fitted.model(torch.tensor(x, dtype=torch.float32)).numpy()
        growth = fitted.target_scaler.inverse_transform(growth[:, None]).ravel()
        prediction = validate_forecast(history.iloc[-4].to_numpy() * np.exp(growth))
        predictions.append(prediction)
        history.loc[history.index[-1] + 1] = prediction
    return np.asarray(predictions)


def save_global(
    fitted: FittedGlobal, destination: Path, trained_through: str, source_sha256: str
) -> None:
    """Persist a standalone joblib bundle without pickling a script-local torch class."""
    payload = {
        "format_version": 1,
        "config": fitted.config,
        "categories": fitted.categories,
        "scaler": fitted.scaler,
        "target_scaler": fitted.target_scaler,
        "target": "current_price_million_php",
        "target_transform": "log_year_over_year_growth",
        "frequency": "quarterly",
        "seasonal_period": 4,
        "trained_through": trained_through,
        "source_sha256": source_sha256,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "torch")
        },
    }
    if fitted.config["family"] == "lstm":
        payload["state_dict"] = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in fitted.model.state_dict().items()
        }
        payload["input_size"] = fitted.model.recurrent.input_size
    else:
        payload["model"] = fitted.model
    joblib.dump(payload, destination, compress=3)


def load_global(source: Path) -> FittedGlobal:
    """Load a trusted local RF or LSTM joblib bundle for forecast_global."""
    payload = joblib.load(source)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported pooled HFCE artifact version")
    config = payload["config"]
    if config["family"] == "lstm":
        model = GlobalLSTM(payload["input_size"], config["hidden_size"])
        model.load_state_dict(
            {name: torch.from_numpy(value) for name, value in payload["state_dict"].items()}
        )
        model.eval()
    else:
        model = payload["model"]
    return FittedGlobal(
        config, payload["categories"], model, payload["scaler"], payload["target_scaler"]
    )


def origins(start: int, end: int, horizon: int = HORIZON) -> list[tuple[int, int]]:
    """Expanding history cutoffs with forecasts truncated at the evaluation boundary."""
    if not 0 < start < end or horizon < 1:
        raise ValueError("Invalid walk-forward boundaries")
    return [(cutoff, min(horizon, end - cutoff)) for cutoff in range(start, end)]


def evaluate(
    panel: pd.DataFrame,
    start: int,
    end: int,
    config: dict[str, Any],
    seed: int,
    sarima_specs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Refit at every origin using a synchronized cutoff across all categories."""
    records = []
    for cutoff, horizon in origins(start, end):
        history = panel.iloc[:cutoff]
        converged = dict.fromkeys(panel.columns, True)
        if config["family"] in {"lstm", "random_forest"}:
            fitted = fit_global(history, config, seed)
            forecast = forecast_global(fitted, history, horizon)
        elif config["family"] == "seasonal_naive":
            forecast = np.asarray([history.iloc[-4 + step].to_numpy() for step in range(horizon)])
        elif config["family"] == "sarima":
            columns = []
            for category in panel.columns:
                spec = sarima_specs[category]
                series = transform_series(history[category], spec["transform"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = SARIMAX(
                        series,
                        order=spec["order"],
                        seasonal_order=spec["seasonal_order"],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=300)
                converged[category] = bool(result.mle_retvals.get("converged", False))
                columns.append(
                    validate_forecast(inverse_forecast(result.forecast(horizon), spec["transform"]))
                )
            forecast = np.column_stack(columns)
        else:
            raise ValueError(f"Unknown evaluation family: {config['family']}")
        for step in range(horizon):
            for col, category in enumerate(panel.columns):
                records.append(
                    {
                        "origin": str(history.index[-1]),
                        "target": str(panel.index[cutoff + step]),
                        "horizon": step + 1,
                        "category": category,
                        "actual": float(panel.iloc[cutoff + step, col]),
                        "predicted": float(forecast[step, col]),
                        "previous_actual": float(history.iloc[-1, col]),
                        "converged": converged[category],
                    }
                )
        if (cutoff - start) % 7 == 0 or cutoff == end - 1:
            LOGGER.info(
                "%s window=%s origin=%s", config["family"], config.get("window"), history.index[-1]
            )
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> dict[str, Any]:
    """Keep macro scale-free metrics separate from pooled monetary errors."""
    per_category = {
        category: error_metrics(group.actual, group.predicted, group.previous_actual)
        for category, group in records.groupby("category")
    }
    return {
        "macro_smape_percent": float(np.mean([v["smape_percent"] for v in per_category.values()])),
        "macro_mda_percent": float(np.mean([v["mda_percent"] for v in per_category.values()])),
        "pooled": error_metrics(records.actual, records.predicted, records.previous_actual),
        "per_category": per_category,
        "forecast_count": len(records),
        "nonconverged_origin_category_count": int(
            len(records.loc[~records.converged, ["origin", "category"]].drop_duplicates())
        ),
    }


def all_summaries(records: pd.DataFrame) -> dict[str, Any]:
    return {
        "all_horizons": summarize(records),
        "by_horizon": {str(h): summarize(group) for h, group in records.groupby("horizon")},
    }


def convergence_sensitivity(output: Path) -> dict[str, Any]:
    """Exclude the same failed SARIMA origin/category pairs from every family."""
    sarima = pd.read_csv(output / "sarima_test_predictions.csv")
    excluded = sarima.loc[~sarima.converged, ["origin", "category"]].drop_duplicates()
    result = {"excluded_origin_categories": excluded.to_dict("records"), "test": {}}
    for family in ("seasonal_naive", "sarima", "random_forest", "lstm"):
        records = pd.read_csv(output / f"{family}_test_predictions.csv")
        matched = records.merge(
            excluded.assign(excluded=True), on=["origin", "category"], how="left"
        )
        result["test"][family] = all_summaries(
            matched.loc[matched.excluded.isna()].drop(columns="excluded")
        )
    return result


def plot_test_forecasts(output: Path, destination: Path) -> None:
    """Save a standalone comparison of held-out seasonal trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = ["seasonal_naive", "sarima", "random_forest", "lstm"]
    frames = {name: pd.read_csv(output / f"{name}_test_predictions.csv") for name in families}
    categories = sorted(frames["sarima"].category.unique())
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    for axis, category in zip(axes.flat, categories, strict=True):
        actual = frames["sarima"].query("category == @category and horizon == 1")
        x = np.arange(len(actual))
        axis.plot(x, actual.actual / 1000, color="black", linewidth=2, label="Actual")
        for family in families:
            rows = frames[family].query("category == @category and horizon == 1")
            axis.plot(x, rows.predicted / 1000, linewidth=1.2, label=family)
        axis.set_title(category.replace("_", " "), fontsize=10)
        axis.set_ylabel("Billion PHP")
        axis.set_xticks(x[::2], actual.target.iloc[::2], rotation=30)
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.97))
    figure.suptitle("HFCE held-out one-quarter forecasts: 2024Q1–2026Q2")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HFCE Pooled LSTM and Random Forest Experiment",
        "",
        f"Run: {report['created_utc']}. Seed: {report['seed']} (single seed).",
        "",
        "## Split and validation",
        "",
        "| Partition | Period | Quarters/category | Observations (12 categories) |",
        "| :--- | :--- | ---: | ---: |",
        "| Initial train | 2000Q1–2016Q4 | 68 | 816 |",
        "| Validation | 2017Q1–2023Q4 | 28 | 336 |",
        "| Test | 2024Q1–2026Q2 | 10 | 120 |",
        "",
        "Validation has 28 expanding origins; test has 10. Each origin forecasts up to four quarters, truncated at its partition boundary. All categories share the same cutoff. Scalers and models refit from scratch using only observations before that origin. Test observations enter training only after they become available in chronological order. No model settings change after validation selection.",
        "",
        "Four configurations per pooled family are ranked by macro category SMAPE across validation origins/horizons. Monetary MAE is reported separately because category scales differ. The first 12 or 16 quarters are feature context, yielding 672 or 624 labeled initial training windows, respectively; windows overlap and are not independent observations.",
        "",
        "## Models and seasonal representation",
        "",
        "One RF or LSTM pools all 12 categories. Each input contains category identity, relative log levels, QoQ/YoY growth, observed quarter and target quarter sin/cos features. The target is log year-over-year growth. Reconstruction multiplies the same-quarter prior-year level by predicted growth; multi-quarter predictions are recursive. Thus these are explicitly seasonal models, not raw-level RF/LSTM baselines. The seasonal lag permits RF forecasts to grow with the latest observed levels.",
        "",
        "LSTM uses one recurrent layer, full-batch Adam (learning rate 0.01, weight decay 0.001), gradient clipping 1.0, fixed epochs, CPU and one thread. Both pooled families use training-only input and target standardization. No early stopping, future covariates, interpolation, or COVID intervention is used.",
        "",
        "SARIMA uses the original per-category validation-selected specifications from the existing evaluation file, with 300 optimizer iterations. It does not use test-dependent release substitutions. Finite nonconverged fits remain in evaluation and are counted explicitly; no test-based model replacement occurs. This is a controlled refit comparison, not an exact reproduction of released SARIMA metrics. The original SARIMA search ranked per-category monetary MAE; the pooled search ranks macro SMAPE. Search budgets and seasonal representations also differ, so this compares practical forecasting setups rather than isolating architecture alone.",
        "",
        "## Selected pooled configurations",
        "",
        "```json",
        json.dumps(report["selected"], indent=2),
        "```",
        "",
        "## Results",
        "",
        f"![Held-out seasonal trajectories]({path.with_suffix('.png').name})",
        "",
        "SMAPE is equally weighted across categories; lower is better. MDA compares the forecast and target with the last observation available at the origin. Monetary errors use million PHP. Overlapping multi-horizon forecasts are counted per origin and horizon, not as independent quarters.",
        "",
        "| Split | Model | Horizon | Macro SMAPE (%) | Macro MDA (%) | Pooled MAE | Pooled RMSE |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for split, models in report["results"].items():
        for family, scores in models.items():
            for horizon, metrics in [
                ("1", scores["by_horizon"]["1"]),
                ("4", scores["by_horizon"]["4"]),
                ("1–4", scores["all_horizons"]),
            ]:
                pooled = metrics["pooled"]
                lines.append(
                    f"| {split} | {family} | {horizon} | {metrics['macro_smape_percent']:.3f} | {metrics['macro_mda_percent']:.2f} | {pooled['mae_million_php']:.2f} | {pooled['rmse_million_php']:.2f} |"
                )
    lines += [
        "",
        "## Test category SMAPE (one quarter ahead)",
        "",
        "| Category | Seasonal naive | SARIMA | Pooled RF | Pooled LSTM |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    families = ["seasonal_naive", "sarima", "random_forest", "lstm"]
    for category in report["categories"]:
        scores = [
            report["results"]["test"][f]["by_horizon"]["1"]["per_category"][category][
                "smape_percent"
            ]
            for f in families
        ]
        lines.append(f"| {category} | " + " | ".join(f"{v:.3f}" for v in scores) + " |")
    conclusions = []
    test_results = report["results"]["test"]
    for horizon in ("1", "4"):
        winner = min(
            families,
            key=lambda family: test_results[family]["by_horizon"][horizon]["macro_smape_percent"],
        )
        conclusions.append(f"Lowest test macro SMAPE at horizon {horizon}: `{winner}`.")
    for family in ("random_forest", "lstm"):
        wins = sum(
            test_results[family]["by_horizon"]["1"]["per_category"][category]["smape_percent"]
            < test_results["sarima"]["by_horizon"]["1"]["per_category"][category]["smape_percent"]
            for category in report["categories"]
        )
        conclusions.append(f"`{family}` beats SARIMA one-step SMAPE in {wins}/12 categories.")
    for split in ("validation", "test"):
        count = report["results"][split]["sarima"]["all_horizons"][
            "nonconverged_origin_category_count"
        ]
        conclusions.append(f"SARIMA nonconverged origin/category fits in {split}: {count}.")
    if sensitivity := report.get("converged_only_sensitivity"):
        scores = "; ".join(
            f"{family}: {sensitivity['test'][family]['by_horizon']['1']['macro_smape_percent']:.3f}%"
            for family in families
        )
        conclusions.append(
            "After excluding the same nonconverged SARIMA origin/category pairs from every model, "
            f"one-step macro SMAPE is {scores}."
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        " ".join(conclusions),
        "",
        "A pooled model can represent seasonal HFCE, but seasonality alone does not establish superiority over local SARIMA. Judge the held-out errors by horizon and category. The 12 categories share macroeconomic shocks and do not provide 12 independent economic histories. The 10-quarter test window, one seed, small search grid, validation-period COVID shock, and explicit seasonal growth engineering limit generalization. This experiment does not establish performance on personal household spending or on a new category.",
        "",
        "The test period was already used by the pre-existing SARIMA experiment; it is held out from this new pooled selection, not a newly collected prospective sample. Further tuning after inspecting these results requires a new evaluation window. Full-data final refits are for future forecasts only and are never used to report test scores.",
        "",
        "See the adjacent JSON for candidate validation scores, category/horizon metrics, SARIMA convergence counts, hashes, and dependency versions. Forecast-level CSVs, scalers, fitted models, split assignments, and four-quarter future forecasts are in the generated output directory recorded there.",
        "",
        "## Saved models",
        "",
        "Both selected models are refit through 2026Q2 and saved as self-contained `random_forest.joblib` and `lstm.joblib` bundles. Each includes category ordering, configuration, input/target scalers, training cutoff and data hash. The LSTM bundle embeds its learned weight arrays and architecture dimensions; no separate weights file is required.",
        "",
        "```python",
        "from pathlib import Path",
        "from training.scripts.modeling.compare_hfce_global_models import load_global, forecast_global",
        "",
        'model = load_global(Path("training/models/hfce_global_comparison/lstm.joblib"))',
        "# history: positive quarterly HFCE DataFrame, PeriodIndex(freq='Q'),",
        "# columns ordered exactly as model.categories; forecasts return horizon x category.",
        "predictions = forecast_global(model, history, horizon=4)",
        "```",
        "",
        "## Reproduction",
        "",
        "Run from the repository root with the existing virtual environment and local observed HFCE dataset:",
        "",
        "```bash",
        ".venv/bin/python training/scripts/modeling/compare_hfce_global_models.py",
        ".venv/bin/pytest -q tests/test_compare_hfce_global_models.py tests/test_train_hfce_sarima_v4.py",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source, output = args.input, args.output_dir
    frame = pd.read_csv(source)
    validate_dataset(
        frame, expected_categories=set(PURPOSES.values()), require_continuous_periods=True
    )
    frame["period"] = pd.PeriodIndex(frame.year_quarter, freq="Q")
    panel = (
        frame.pivot(index="period", columns="category", values="current_price_million_php")
        .sort_index()
        .sort_index(axis=1)
    )
    for category in panel:
        split_series(panel[category])
    # Reject timelines that happen to have the expected counts but different dates.
    if not panel.index.equals(pd.period_range("2000Q1", "2026Q2", freq="Q")):
        raise ValueError("Unexpected HFCE evaluation timeline")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline = json.loads(args.sarima_report.read_text())
    if source_hash != baseline["source_sha256"]:
        raise ValueError("SARIMA report was selected on a different dataset")
    specs = {category: baseline["categories"][category]["selected_candidate"] for category in panel}
    output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    assignments = frame[["year_quarter", "category"]].copy()
    assignments["split"] = np.where(
        frame.period <= pd.Period("2016Q4"),
        "train",
        np.where(frame.period <= pd.Period("2023Q4"), "validation", "test"),
    )
    assignments.to_csv(output / "split_assignments.csv", index=False)
    report: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "source": str(source),
        "source_sha256": source_hash,
        "sarima_report_sha256": hashlib.sha256(args.sarima_report.read_bytes()).hexdigest(),
        "output_dir": str(output),
        "categories": list(panel.columns),
        "split": {
            "train": ["2000Q1", "2016Q4", 68],
            "validation": ["2017Q1", "2023Q4", 28],
            "test": ["2024Q1", "2026Q2", 10],
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "torch", "scikit-learn", "statsmodels"]
        },
        "grid": candidate_grid(),
        "selection_metric": "validation all-horizon macro category SMAPE",
        "selected": {},
        "validation_candidates": [],
        "sarima_specs": specs,
        "results": {"validation": {}, "test": {}},
        "artifacts": {},
    }
    LOGGER.info(
        "SPLIT: train=68 quarters/category; validation=28; test=10; categories=%s",
        len(panel.columns),
    )
    LOGGER.info("GRID: %s", json.dumps(candidate_grid()))
    validation_records = {}
    for config in candidate_grid():
        LOGGER.info("Validating %s", config)
        records = evaluate(panel, TRAIN_SIZE, VALIDATION_END, config, args.seed)
        scores = all_summaries(records)
        report["validation_candidates"].append({"config": config, "scores": scores})
        family = config["family"]
        score = scores["all_horizons"]["macro_smape_percent"]
        if family not in validation_records or score < validation_records[family][0]:
            validation_records[family] = (score, records, config)
        LOGGER.info("Validation macro SMAPE=%.4f", score)
        (output / "selection_progress.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    for family, (_, records, config) in validation_records.items():
        report["selected"][family] = config
        report["results"]["validation"][family] = all_summaries(records)
        records.to_csv(output / f"{family}_validation_predictions.csv", index=False)
    # Materialize the locked choices before evaluating any test target.
    (output / "locked_selection.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    LOGGER.info("LOCKED before test: %s", report["selected"])
    for family in ["seasonal_naive", "sarima", "random_forest", "lstm"]:
        config = report["selected"].get(family, {"family": family})
        if family in {"seasonal_naive", "sarima"}:
            records = evaluate(panel, TRAIN_SIZE, VALIDATION_END, config, args.seed, specs)
            report["results"]["validation"][family] = all_summaries(records)
            records.to_csv(output / f"{family}_validation_predictions.csv", index=False)
        records = evaluate(panel, VALIDATION_END, TEST_END, config, args.seed, specs)
        report["results"]["test"][family] = all_summaries(records)
        records.to_csv(output / f"{family}_test_predictions.csv", index=False)
        if family in report["selected"]:
            fitted = fit_global(panel, config, args.seed)
            artifact_path = output / f"{family}.joblib"
            save_global(fitted, artifact_path, str(panel.index[-1]), source_hash)
            forecast = forecast_global(fitted, panel, HORIZON)
            restored = load_global(artifact_path)
            if not np.allclose(forecast_global(restored, panel, HORIZON), forecast):
                raise ValueError(f"{family} artifact reload changed forecasts")
            report["artifacts"][family] = {
                "path": str(artifact_path),
                "format": "self-contained joblib bundle",
                "bytes": artifact_path.stat().st_size,
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                "reload_forecast_verified": True,
            }
            future = pd.DataFrame(
                forecast,
                index=pd.period_range(panel.index[-1] + 1, periods=HORIZON, freq="Q"),
                columns=panel.columns,
            )
            future.to_csv(output / f"{family}_future_forecast.csv", index_label="quarter")
    report["converged_only_sensitivity"] = convergence_sensitivity(output)
    report["completed_utc"] = datetime.now(UTC).isoformat()
    args.report.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    write_report(args.report, report)
    plot_test_forecasts(output, args.report.with_suffix(".png"))
    LOGGER.info("Completed: %s", args.report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"),
    )
    parser.add_argument(
        "--sarima-report", type=Path, default=Path("models/hfce_sarima_v4/evaluation.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training/models/hfce_global_comparison")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/experiments/hfce-global-model-comparison.md")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
