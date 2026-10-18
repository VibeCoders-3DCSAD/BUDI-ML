from __future__ import annotations

import pandas as pd
import training.scripts.modeling.train_hfce_sarima_v4 as hfce_training
from training.scripts.modeling.train_hfce_sarima_v4 import (
    apply_release_gate,
    average_metrics,
    candidate_rank_key,
    candidates,
    error_metrics,
    evaluate_candidate,
    fit_candidate,
    release_candidate,
    select_candidate,
    split_series,
    walk_forward_folds,
)


def test_split_series_uses_the_locked_methodology_windows() -> None:
    series = pd.Series(range(106), index=pd.period_range("2000Q1", periods=106, freq="Q"))

    train, validation, test = split_series(series)

    assert (len(train), len(validation), len(test)) == (68, 28, 10)
    assert str(train.index[-1]) == "2016Q4"
    assert str(validation.index[0]) == "2017Q1"
    assert str(validation.index[-1]) == "2023Q4"
    assert str(test.index[0]) == "2024Q1"
    assert str(test.index[-1]) == "2026Q2"


def test_error_metrics_reports_the_specified_metrics_and_per_target_direction() -> None:
    metrics = error_metrics([110.0, 90.0], [100.0, 100.0], [100.0, 110.0])

    assert metrics["mae_million_php"] == 10.0
    assert metrics["rmse_million_php"] == 10.0
    assert metrics["smape_percent"] == 10.025062656641603
    assert metrics["mda_percent"] == 50.0


def test_average_metrics_only_reports_scale_independent_category_averages() -> None:
    average = average_metrics(
        [
            {
                "mae_million_php": 10.0,
                "smape_percent": 5.0,
                "mda_percent": 80.0,
                "rmse_million_php": 11.0,
            },
            {
                "mae_million_php": 30.0,
                "smape_percent": 15.0,
                "mda_percent": 60.0,
                "rmse_million_php": 31.0,
            },
        ]
    )

    assert average == {"smape_percent": 10.0, "mda_percent": 70.0}


def test_walk_forward_folds_expand_at_every_validation_origin() -> None:
    series = pd.Series(range(106), index=pd.period_range("2000Q1", periods=106, freq="Q"))
    train, validation, _ = split_series(series)

    folds = walk_forward_folds(train, validation)

    assert len(folds) == 28
    assert [len(history) for history, _ in folds[:2]] == [68, 69]
    assert [len(held_out) for _, held_out in folds[-4:]] == [4, 3, 2, 1]


def test_candidates_include_benchmarks_and_all_declared_sarima_orders() -> None:
    grid = candidates()
    sarima = [candidate for candidate in grid if candidate["family"] == "sarima"]

    assert {candidate["family"] for candidate in grid} == {"seasonal_naive", "ets", "sarima"}
    assert len(sarima) == 864
    assert all(
        not (candidate["order"][1] and candidate["seasonal_order"][1]) for candidate in sarima
    )
    assert {candidate["seasonal_order"][3] for candidate in sarima} == {4}


def test_ets_candidate_is_not_rejected_for_using_parameter_mapping() -> None:
    series = pd.Series(range(1, 21), index=pd.period_range("2000Q1", periods=20, freq="Q"))
    candidate = next(candidate for candidate in candidates() if candidate["name"] == "ets_aan")

    fitted = fit_candidate(series, candidate)

    assert fitted.family == "ets"


def test_candidate_rank_key_prioritizes_mae_then_specified_tie_breakers() -> None:
    candidates_by_metric = [
        {
            "metrics": {
                "mae_million_php": 11.0,
                "smape_percent": 1.0,
                "rmse_million_php": 1.0,
                "mda_percent": 100.0,
            }
        },
        {
            "metrics": {
                "mae_million_php": 10.0,
                "smape_percent": 6.0,
                "rmse_million_php": 9.0,
                "mda_percent": 60.0,
            }
        },
        {
            "metrics": {
                "mae_million_php": 10.0,
                "smape_percent": 5.0,
                "rmse_million_php": 11.0,
                "mda_percent": 70.0,
            }
        },
    ]

    assert sorted(candidates_by_metric, key=candidate_rank_key)[0] is candidates_by_metric[2]


def test_selection_releases_the_best_sarima_not_a_simple_benchmark(monkeypatch) -> None:
    train = pd.Series([1.0])
    validation = pd.Series([2.0])
    seasonal_naive = {"name": "seasonal_naive", "family": "seasonal_naive"}
    sarima = {"name": "sarima", "family": "sarima"}

    monkeypatch.setattr(hfce_training, "candidates", lambda: (seasonal_naive, sarima))
    monkeypatch.setattr(
        hfce_training,
        "evaluate_candidate",
        lambda candidate, *_args: (
            {
                "mae_million_php": 1.0 if candidate["family"] != "sarima" else 2.0,
                "smape_percent": 1.0,
                "mda_percent": 100.0,
                "rmse_million_php": 1.0,
            },
            [],
        ),
    )

    selected, selection = select_candidate(train, validation, "food")

    assert selected is sarima
    assert selection["locked_simple_fallback"]["params"] is seasonal_naive


def test_validation_mda_uses_each_origin_last_observed_value(monkeypatch) -> None:
    train = pd.Series([100.0], index=pd.period_range("2016Q4", periods=1, freq="Q"))
    validation = pd.Series([110.0, 90.0], index=pd.period_range("2017Q1", periods=2, freq="Q"))

    monkeypatch.setattr(hfce_training, "fit_candidate", lambda history, candidate: history)
    monkeypatch.setattr(
        hfce_training,
        "forecast_candidate",
        lambda model, horizon: [105.0] * horizon,
    )

    metrics, _ = evaluate_candidate({"name": "test"}, train, validation, horizon=2)

    assert metrics["mda_percent"] == 66.66666666666666


def test_release_records_a_weaker_sarima_benchmark_comparison(monkeypatch) -> None:
    selected = {"name": "sarima", "family": "sarima"}
    fallback = {"name": "seasonal_naive", "family": "seasonal_naive"}
    selection = {"locked_simple_fallback": {"params": fallback}}
    train = pd.Series([1.0])
    validation = pd.Series([2.0])
    test = pd.Series([3.0])

    def fake_evaluate(candidate, *_args):
        mae = 11.1 if candidate["name"] == "sarima" else 10.0
        return (
            {
                "mae_million_php": mae,
                "smape_percent": 1.0,
                "mda_percent": 100.0,
                "rmse_million_php": mae,
            },
            [],
        )

    monkeypatch.setattr(hfce_training, "evaluate_candidate", fake_evaluate)

    released, comparison, metrics, _ = release_candidate(
        selected, selection, train, validation, test
    )

    assert released is selected
    assert metrics["mae_million_php"] == 11.1
    assert comparison["fallback"]["candidate"] is fallback
    assert comparison["fallback"]["mae_deterioration_ratio"] == 0.10999999999999988


def test_release_records_equal_zero_mae_benchmark_comparison(monkeypatch) -> None:
    selected = {"name": "sarima", "family": "sarima"}
    fallback = {"name": "seasonal_naive", "family": "seasonal_naive"}
    selection = {"locked_simple_fallback": {"params": fallback}}

    monkeypatch.setattr(
        hfce_training,
        "evaluate_candidate",
        lambda *_args: (
            {
                "mae_million_php": 0.0,
                "smape_percent": 0.0,
                "mda_percent": 100.0,
                "rmse_million_php": 0.0,
            },
            [],
        ),
    )

    _, comparison, _, _ = release_candidate(
        selected, selection, pd.Series([1.0]), pd.Series([2.0]), pd.Series([3.0])
    )

    assert comparison["fallback"]["mae_deterioration_ratio"] == 0.0


def test_release_gate_skips_unestimable_winner_in_validation_order(monkeypatch) -> None:
    ranked = [
        {"params": {"name": "winner", "family": "sarima"}},
        {"params": {"name": "runner_up", "family": "sarima"}},
        {"params": {"name": "third", "family": "sarima"}},
    ]

    def fake_evaluate(candidate, *_args):
        if candidate["name"] == "winner":
            raise ValueError("winner did not converge")
        return {"mae_million_php": 1.0}, []

    monkeypatch.setattr(hfce_training, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(hfce_training, "fit_candidate", lambda *_args: "model")
    monkeypatch.setattr(hfce_training, "forecast_candidate", lambda *_args: None)

    released, model, _, rejections = apply_release_gate(
        {"sarima_ranked": ranked},
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0]),
        pd.Series([2.0]),
        pd.Series([3.0]),
    )

    assert released is ranked[1]
    assert model == "model"
    assert rejections == [{"name": "winner", "reason": "winner did not converge"}]
