"""Allocate annual FIES household category totals across observed HFCE quarters.

The output is temporally disaggregated, not observed quarterly household spending.
Its quarterly rows reconcile exactly to each annual FIES household-category amount.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training.scripts.data.build_hfce_category_forecast_dataset_v4 import build_dataset
from training.scripts.data.clean_raw_fies import CATEGORIES


OUTPUT_COLUMNS = (
    "household_id",
    "anchor_survey_year",
    "year",
    "quarter",
    "year_quarter",
    "observation_status",
    "allocation_method",
    *CATEGORIES,
)
ALLOCATION_METHOD = "within_year_hfce_current_price_share"


def quarterly_weights(hfce_source: str | Path, years: set[int]) -> pd.DataFrame:
    """Return current-price HFCE shares for every required category and quarter."""
    hfce = build_dataset(hfce_source, min(years), max(years))
    hfce = hfce.loc[
        hfce["year"].isin(years),
        ["year", "quarter", "year_quarter", "category", "current_price_million_php"],
    ].copy()
    expected = pd.MultiIndex.from_product(
        [sorted(years), range(1, 5), CATEGORIES], names=["year", "quarter", "category"]
    )
    available = pd.MultiIndex.from_frame(hfce[["year", "quarter", "category"]])
    missing = expected.difference(available)
    if not missing.empty:
        sample = ", ".join(
            f"{year}-Q{quarter} {category}" for year, quarter, category in missing[:5]
        )
        raise ValueError(
            f"HFCE is missing category-quarter values required for allocation: {sample}"
        )
    hfce["hfce_quarter_weight"] = hfce["current_price_million_php"] / hfce.groupby(
        ["year", "category"]
    )["current_price_million_php"].transform("sum")
    return hfce.drop(columns="current_price_million_php")


def disaggregate_households(households: pd.DataFrame, hfce_source: str | Path) -> pd.DataFrame:
    """Allocate each annual household category amount to four HFCE-weighted quarters."""
    required = {"household_id", "year", *CATEGORIES}
    missing = required - set(households.columns)
    if missing:
        raise ValueError(f"Household categories are missing columns: {sorted(missing)}")
    annual = households.loc[:, ["household_id", "year", *CATEGORIES]].copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    if annual["year"].isna().any() or annual["household_id"].isna().any():
        raise ValueError("Household categories contain a missing household_id or year")
    annual["year"] = annual["year"].astype("int64")
    if annual.duplicated(["household_id", "year"]).any():
        raise ValueError("Household categories repeat a household_id and year")
    annual[list(CATEGORIES)] = annual[list(CATEGORIES)].apply(pd.to_numeric, errors="raise")
    if annual[list(CATEGORIES)].isna().any().any() or annual[list(CATEGORIES)].lt(0).any().any():
        raise ValueError("Household categories contain missing or negative category amounts")

    long = annual.melt(
        id_vars=["household_id", "year"],
        value_vars=CATEGORIES,
        var_name="category",
        value_name="annual_amount_php",
    )
    weighted = long.merge(
        quarterly_weights(hfce_source, set(annual["year"])),
        on=["year", "category"],
        how="left",
        validate="many_to_many",
    )
    weighted["amount_php"] = weighted["annual_amount_php"] * weighted["hfce_quarter_weight"]
    result = (
        weighted.pivot(
            index=["household_id", "year", "quarter", "year_quarter"],
            columns="category",
            values="amount_php",
        )
        .reset_index()
        .rename(columns={"year": "anchor_survey_year"})
    )
    result["year"] = result["anchor_survey_year"]
    result["observation_status"] = "hfce_temporally_disaggregated"
    result["allocation_method"] = ALLOCATION_METHOD
    return (
        result.loc[:, OUTPUT_COLUMNS]
        .sort_values(["anchor_survey_year", "household_id", "quarter"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--households", type=Path, default=Path("RAW/cleaned/household_categories.csv")
    )
    parser.add_argument("--hfce", type=Path, default=Path("RAW/HFCE/hfce_by_purpose.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("RAW/cleaned/household_categories_quarterly.csv")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    disaggregate_households(pd.read_csv(args.households), args.hfce).to_csv(
        args.output, index=False
    )


if __name__ == "__main__":
    main()
