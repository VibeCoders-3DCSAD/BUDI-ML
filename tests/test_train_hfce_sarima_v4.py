from __future__ import annotations

import pandas as pd
from training.scripts.modeling.train_hfce_sarima_v4 import (
    average_metrics,
    candidates,
    error_metrics,
    intervention_features,
    split_series,
    walk_forward_folds,
)


def test_split_series_uses_contiguous_70_20_10_quarterly_windows() -> None:
    series = pd.Series(range(106), index=pd.period_range("2000Q1", periods=106, freq="Q"))

    train, validation, test = split_series(series)

    assert len(train) == 74
    assert len(validation) == 21
    assert len(test) == 11
    assert str(train.index[0]) == "2000Q1"
    assert str(train.index[-1]) == "2018Q2"
    assert str(validation.index[0]) == "2018Q3"
    assert str(validation.index[-1]) == "2023Q3"
    assert str(test.index[0]) == "2023Q4"
    assert str(test.index[-1]) == "2026Q2"


def test_error_metrics_reports_smape_and_directional_accuracy() -> None:
    actual = pd.Series([110.0, 90.0])

    metrics = error_metrics(actual, [100.0, 100.0], previous_actual=100.0)

    assert metrics["mae_million_php"] == 10.0
    assert metrics["rmse_million_php"] == 10.0
    assert metrics["smape_percent"] == 10.025062656641603
    assert metrics["mda_percent"] == 50.0


def test_average_metrics_uses_an_unweighted_category_mean() -> None:
    average = average_metrics(
        [
            {"mae_million_php": 10.0, "smape_percent": 5.0, "mda_percent": 80.0},
            {"mae_million_php": 30.0, "smape_percent": 15.0, "mda_percent": 60.0},
        ]
    )

    assert average == {"mae_million_php": 20.0, "smape_percent": 10.0, "mda_percent": 70.0}


def test_intervention_features_retains_covid_and_post_covid_quarters() -> None:
    periods = pd.period_range("2019Q4", periods=10, freq="Q")

    features = intervention_features(
        periods,
        pd.Period("2020Q1", freq="Q"),
        pd.Period("2021Q4", freq="Q"),
        pd.Period("2022Q1", freq="Q"),
    )

    assert features.loc[pd.Period("2019Q4", freq="Q")].tolist() == [0.0, 0.0]
    assert features.loc[pd.Period("2020Q1", freq="Q")].tolist() == [1.0, 0.0]
    assert features.loc[pd.Period("2021Q4", freq="Q")].tolist() == [1.0, 0.0]
    assert features.loc[pd.Period("2022Q1", freq="Q")].tolist() == [0.0, 1.0]


def test_walk_forward_folds_expand_across_the_validation_window() -> None:
    series = pd.Series(range(106), index=pd.period_range("2000Q1", periods=106, freq="Q"))
    train, validation, _ = split_series(series)

    folds = walk_forward_folds(train, validation)

    assert [len(history) for history, _ in folds] == [74, 78, 82, 86, 90, 94]
    assert [len(held_out) for _, held_out in folds] == [4, 4, 4, 4, 4, 1]


def test_candidates_require_intervention_sarimax_for_every_category() -> None:
    assert {candidate["family"] for candidate in candidates("sarimax_intervention")} == {
        "sarimax_intervention"
    }


def test_candidates_can_produce_a_baseline_sarima_comparison() -> None:
    assert {candidate["family"] for candidate in candidates("sarima")} == {"sarima"}
