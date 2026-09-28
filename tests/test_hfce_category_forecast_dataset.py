from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from training.scripts.data.build_hfce_category_forecast_dataset_v4 import (
    build_dataset,
    validate_dataset,
)


def test_build_dataset_creates_monthly_category_targets(tmp_path: Path):
    source = tmp_path / "hfce.csv"
    source.write_text(
        '"Household Final Consumption Expenditure by Purpose"\n'
        '"Purpose","At Current Prices 2022 Q1","At Constant 2018 Prices 2022 Q1","At Current Prices 2026 Q2","At Constant 2018 Prices 2026 Q2"\n'
        '"..Food and non-alcoholic beverages",300,200,600,300\n'
        '"..Health",30,20,60,30\n'
        '"Household Final Consumption Expenditure",330,220,660,330\n'
    )

    result = build_dataset(source)

    assert set(result["category"]) == {"food", "health"}
    assert result["year_quarter"].tolist() == ["2022-Q1", "2026-Q2"] * 2
    assert result.loc[result["category"] == "food", "current_price_million_php"].tolist() == [
        300.0,
        600.0,
    ]
    assert result.loc[result["category"] == "food", "constant_2018_price_million_php"].tolist() == [
        200.0,
        300.0,
    ]
    assert result.groupby("year_quarter")["current_price_category_share"].sum().tolist() == [
        1.0,
        1.0,
    ]
    assert result["is_complete_year"].tolist() == [False] * 4
    validate_dataset(result)


def test_build_dataset_rejects_reversed_year_range(tmp_path: Path):
    with pytest.raises(ValueError, match="start_year"):
        build_dataset(tmp_path / "missing.csv", 2026, 2022)


def test_validate_dataset_rejects_missing_category_quarter() -> None:
    dataset = build_dataset_fixture()
    invalid = dataset.loc[~((dataset["category"] == "food") & (dataset["quarter"] == 2))]
    invalid.loc[
        (invalid["category"] == "health") & (invalid["quarter"] == 2),
        "current_price_category_share",
    ] = 1.0

    with pytest.raises(ValueError, match="identical quarterly timeline"):
        validate_dataset(invalid)


def test_validate_dataset_rejects_incomplete_nonfinal_year() -> None:
    dataset = build_dataset_fixture()
    next_year = dataset.loc[dataset["quarter"] == 1].copy()
    next_year["year"] = 2022
    next_year["year_quarter"] = "2022-Q1"
    next_year["is_complete_year"] = False
    dataset = pd.concat([dataset, next_year], ignore_index=True)
    invalid = dataset.loc[dataset["quarter"] != 4]
    invalid.loc[invalid["year"] == 2021, "is_complete_year"] = False

    with pytest.raises(ValueError, match="incomplete non-final year"):
        validate_dataset(invalid, require_continuous_periods=True)


def build_dataset_fixture() -> pd.DataFrame:
    rows = []
    for category in ("food", "health"):
        for quarter in range(1, 5):
            rows.append(
                {
                    "year_quarter": f"2021-Q{quarter}",
                    "year": 2021,
                    "quarter": quarter,
                    "category": category,
                    "source_frequency": "quarterly_observed",
                    "current_price_million_php": 100.0,
                    "constant_2018_price_million_php": 80.0,
                    "current_price_category_share": 0.5,
                    "is_complete_year": True,
                }
            )
    return pd.DataFrame(rows)
