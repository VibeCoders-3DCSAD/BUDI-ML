"""Behavioral checks for synchronized, causal HFCE pooled forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import training.scripts.modeling.compare_hfce_global_models as comparison
from sklearn.preprocessing import StandardScaler


def panel(periods: int = 80) -> pd.DataFrame:
    time = np.arange(periods)
    base = 100 * np.exp(0.02 * time) * np.array([1, 1.1, 0.9, 1.3])[time % 4]
    return pd.DataFrame(
        {"food": base, "health": base * 0.2},
        index=pd.period_range("2000Q1", periods=periods, freq="Q"),
    )


def test_labeled_samples_do_not_depend_on_their_target_or_future() -> None:
    history = panel(30)
    x, y = comparison.make_samples(history, 8)
    changed = history.copy()
    changed.iloc[12:] *= 7
    changed_x, changed_y = comparison.make_samples(changed, 8)

    assert x.shape == (36, 8, 9)
    np.testing.assert_allclose(x[:2], changed_x[:2])
    np.testing.assert_allclose(y[:2], np.repeat(0.08, 2), atol=1e-6)
    assert not np.allclose(y[:2], changed_y[:2])


def test_features_encode_category_and_remove_absolute_scale() -> None:
    history = panel()
    x = comparison.make_features(history, 8)
    scaled = comparison.make_features(history * 100, 8)

    np.testing.assert_allclose(x, scaled, atol=1e-6)
    np.testing.assert_allclose(x[0, :, :7], x[1, :, :7], atol=1e-6)
    np.testing.assert_array_equal(x[0, :, 7:], np.tile([1, 0], (8, 1)))
    np.testing.assert_array_equal(x[1, :, 7:], np.tile([0, 1], (8, 1)))


def test_walk_forward_expands_and_truncates_at_boundary() -> None:
    folds = comparison.origins(68, 96)

    assert len(folds) == 28
    assert folds[:2] == [(68, 4), (69, 4)]
    assert folds[-4:] == [(92, 4), (93, 3), (94, 2), (95, 1)]
    assert sum(horizon for _, horizon in folds) == 106


def test_evaluation_never_exposes_future_categories_to_fit(monkeypatch) -> None:
    history = panel()
    observed_cutoffs = []

    def fit(train, config, seed):
        observed_cutoffs.append(len(train))
        pd.testing.assert_frame_equal(train, history.iloc[: len(train)])
        return object()

    def forecast(fitted, train, horizon):
        return np.tile(train.iloc[-1].to_numpy(), (horizon, 1))

    monkeypatch.setattr(comparison, "fit_global", fit)
    monkeypatch.setattr(comparison, "forecast_global", forecast)
    original = comparison.evaluate(history, 68, 72, {"family": "lstm"}, 42)
    changed = history.copy()
    changed.iloc[68:] *= 10
    # At the first origin the held-out targets may change, but forecasts cannot.
    changed_records = comparison.evaluate(changed, 68, 69, {"family": "seasonal_naive"}, 42)
    naive = comparison.evaluate(history, 68, 69, {"family": "seasonal_naive"}, 42)

    assert observed_cutoffs == [68, 69, 70, 71]
    np.testing.assert_array_equal(changed_records.predicted, naive.predicted)
    assert len(original) == 20
    assert original.groupby("origin").horizon.max().tolist() == [4, 3, 2, 1]


@pytest.mark.parametrize("family", ["random_forest", "lstm"])
def test_real_pooled_fit_and_recursive_forecast_are_finite_and_causal(family, tmp_path) -> None:
    history = panel(40)
    comparison.torch.set_num_threads(1)
    config = {
        "family": family,
        "window": 8,
        "trees": 10,
        "min_samples_leaf": 2,
        "hidden_size": 4,
        "epochs": 3,
    }
    fitted = comparison.fit_global(history, config, 42)
    forecast = comparison.forecast_global(fitted, history, 4)
    first = comparison.forecast_global(fitted, history, 1)

    assert forecast.shape == (4, 2)
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()
    np.testing.assert_allclose(first[0], forecast[0])
    # After appending the first prediction, the next prediction must match step 2.
    extended = history.copy()
    extended.loc[extended.index[-1] + 1] = forecast[0]
    second = comparison.forecast_global(fitted, extended, 1)
    np.testing.assert_allclose(second[0], forecast[1])
    # The scaler mean is determined by labeled training windows only.
    x, _ = comparison.make_samples(history, 8)
    expected = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
    np.testing.assert_allclose(fitted.scaler.mean_, expected.mean_)
    with pytest.raises(ValueError, match="category order"):
        comparison.forecast_global(fitted, history.iloc[:, ::-1], 1)
    artifact = tmp_path / f"{family}.joblib"
    comparison.save_global(fitted, artifact, str(history.index[-1]), "test-source-hash")
    restored = comparison.load_global(artifact)
    np.testing.assert_allclose(comparison.forecast_global(restored, history, 4), forecast)
    assert list(tmp_path.iterdir()) == [artifact]


def test_invalid_log_inputs_are_rejected() -> None:
    history = panel(20)
    history.iloc[1, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        comparison.make_features(history, 8)


def test_metric_summary_keeps_one_step_and_multi_horizon_separate() -> None:
    records = comparison.evaluate(panel(), 68, 72, {"family": "seasonal_naive"}, 42)
    scores = comparison.all_summaries(records)

    assert scores["all_horizons"]["forecast_count"] == 20
    assert scores["by_horizon"]["1"]["forecast_count"] == 8
    assert scores["by_horizon"]["4"]["forecast_count"] == 2
    assert scores["all_horizons"]["nonconverged_origin_category_count"] == 0


def test_nonconvergence_sensitivity_excludes_identical_forecasts_from_all_models(tmp_path) -> None:
    records = comparison.evaluate(panel(), 68, 72, {"family": "seasonal_naive"}, 42)
    for family in ("seasonal_naive", "sarima", "random_forest", "lstm"):
        rows = records.copy()
        if family == "sarima":
            rows.loc[rows.origin.eq("2016Q4") & rows.category.eq("food"), "converged"] = False
        rows.to_csv(tmp_path / f"{family}_test_predictions.csv", index=False)

    result = comparison.convergence_sensitivity(tmp_path)

    assert result["excluded_origin_categories"] == [{"origin": "2016Q4", "category": "food"}]
    for scores in result["test"].values():
        assert scores["all_horizons"]["forecast_count"] == 16
        assert scores["by_horizon"]["1"]["forecast_count"] == 7
