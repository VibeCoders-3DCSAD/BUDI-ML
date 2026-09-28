"""Build v4 survey-only quarterly category data from PSA HFCE levels."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

PERIOD = re.compile(
    r"^At (?P<price_basis>Current|Constant 2018) Prices (?P<year>20\d{2}) Q(?P<quarter>[1-4])$"
)
PURPOSES = {
    "Food and non-alcoholic beverages": "food",
    "Alcoholic beverages, tobacco": "alcohol_tobacco",
    "Clothing and footwear": "clothing_footwear",
    "Housing, water, electricity, gas and other fuels": "housing_water_utilities",
    "Furnishings, household equipment and routine household maintenance": "furnishings_household",
    "Health": "health",
    "Transport": "transport",
    "Communication": "communication",
    "Recreation and culture": "recreation_culture",
    "Education": "education",
    "Restaurants and hotels": "restaurants_hotels",
    "Miscellaneous goods and services": "miscellaneous_goods_services",
}
LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "year_quarter",
    "year",
    "quarter",
    "category",
    "source_frequency",
    "current_price_million_php",
    "constant_2018_price_million_php",
    "current_price_category_share",
    "is_complete_year",
}


def build_dataset(source: str | Path, start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """Create one observed HFCE row per category and published calendar quarter."""
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")

    LOGGER.info("Reading PSA HFCE source %s for %s-%s", source, start_year, end_year)
    raw = pd.read_csv(source, header=1)
    LOGGER.info("Loaded %s purposes and %s published value columns", len(raw), len(raw.columns) - 1)
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for record in raw.itertuples(index=False):
        purpose = str(record[0]).removeprefix("..").strip()
        category = PURPOSES.get(purpose)
        if category is None:
            continue
        for column, value in zip(raw.columns[1:], record[1:], strict=True):
            match = PERIOD.match(column)
            if match is None:
                continue
            year, quarter = int(match["year"]), int(match["quarter"])
            if not start_year <= year <= end_year or pd.isna(value) or str(value).strip() == ".":
                continue
            quarterly_hfce = pd.to_numeric(value, errors="coerce")
            if pd.isna(quarterly_hfce):
                raise ValueError(
                    f"HFCE contains a non-numeric value for {purpose}, {year} Q{quarter}"
                )
            quarterly_hfce = float(quarterly_hfce)
            if quarterly_hfce < 0:
                raise ValueError(f"HFCE contains a negative value for {purpose}, {year} Q{quarter}")
            year_quarter = f"{year}-Q{quarter}"
            row = rows.setdefault(
                (year_quarter, category),
                {
                    "year_quarter": year_quarter,
                    "year": year,
                    "quarter": quarter,
                    "category": category,
                    "source_frequency": "quarterly_observed",
                },
            )
            value_column = (
                "current_price_million_php"
                if match["price_basis"] == "Current"
                else "constant_2018_price_million_php"
            )
            row[value_column] = quarterly_hfce
    dataset = pd.DataFrame(rows.values())
    if dataset.empty:
        raise ValueError("No HFCE values were found for the requested years")
    LOGGER.info(
        "Mapped %s PSA purposes into %s category-quarter rows",
        len(set(dataset["category"])),
        f"{len(dataset):,}",
    )
    required = {"current_price_million_php", "constant_2018_price_million_php"}
    if required - set(dataset.columns) or dataset[list(required)].isna().any().any():
        raise ValueError("Every retained HFCE category-quarter must contain both price bases")
    dataset = dataset.sort_values(["category", "year", "quarter"]).reset_index(drop=True)
    dataset["current_price_category_share"] = dataset.groupby("year_quarter")[
        "current_price_million_php"
    ].transform(lambda values: values / values.sum())
    dataset["is_complete_year"] = dataset.groupby("year")["quarter"].transform("nunique").eq(4)
    LOGGER.info(
        "Derived current-price category shares for %s published quarters",
        dataset["year_quarter"].nunique(),
    )
    return dataset


def validate_dataset(
    dataset: pd.DataFrame,
    expected_categories: set[str] | None = None,
    require_continuous_periods: bool = False,
) -> None:
    """Validate the quarterly HFCE training-data contract and log its coverage."""
    missing = REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(f"HFCE dataset is missing required columns: {sorted(missing)}")
    if dataset.empty:
        raise ValueError("HFCE dataset is empty")
    LOGGER.info("Checking required schema and observed quarterly values")
    required = list(REQUIRED_COLUMNS)
    if dataset[required].isna().any().any():
        raise ValueError("HFCE dataset contains missing required values")
    if not dataset["source_frequency"].eq("quarterly_observed").all():
        raise ValueError("HFCE dataset contains non-observed or non-quarterly rows")
    if not dataset["quarter"].isin(range(1, 5)).all():
        raise ValueError("HFCE dataset contains a quarter outside Q1-Q4")
    expected_period = (
        dataset["year"].astype(int).astype(str) + "-Q" + dataset["quarter"].astype(int).astype(str)
    )
    if not dataset["year_quarter"].eq(expected_period).all():
        raise ValueError("HFCE dataset has year_quarter values inconsistent with year and quarter")
    if dataset.duplicated(["year", "quarter", "category"]).any():
        raise ValueError("HFCE dataset repeats a category-quarter")
    LOGGER.info("Checking category contract and category-share reconciliation")
    categories = set(dataset["category"])
    if expected_categories is not None and categories != expected_categories:
        raise ValueError(
            "HFCE dataset categories differ from the required category contract: "
            f"missing={sorted(expected_categories - categories)}, "
            f"unexpected={sorted(categories - expected_categories)}"
        )
    if (dataset[["current_price_million_php", "constant_2018_price_million_php"]] < 0).any().any():
        raise ValueError("HFCE dataset contains negative price-basis values")
    shares = dataset.groupby("year_quarter")["current_price_category_share"].sum()
    if not shares.sub(1).abs().le(1e-12).all():
        raise ValueError("HFCE category shares do not sum to one in every quarter")

    timeline = dataset[["year", "quarter", "year_quarter"]].drop_duplicates()
    periods_by_category = dataset.groupby("category")["year_quarter"].agg(frozenset)
    if periods_by_category.nunique() != 1:
        raise ValueError("HFCE categories do not share an identical quarterly timeline")
    complete_year = timeline.groupby("year")["quarter"].nunique().eq(4)
    expected_complete = dataset["year"].map(complete_year)
    if not dataset["is_complete_year"].eq(expected_complete).all():
        raise ValueError("HFCE is_complete_year does not match published quarter coverage")
    if require_continuous_periods:
        LOGGER.info("Checking continuous quarterly coverage for SARIMA")
        years = sorted(timeline["year"].unique())
        if years != list(range(years[0], years[-1] + 1)):
            raise ValueError("HFCE dataset has missing calendar years")
        for year in years[:-1]:
            quarters = sorted(timeline.loc[timeline["year"].eq(year), "quarter"].unique())
            if quarters != [1, 2, 3, 4]:
                raise ValueError(f"HFCE dataset has an incomplete non-final year: {year}")
        final_quarters = sorted(timeline.loc[timeline["year"].eq(years[-1]), "quarter"].unique())
        if final_quarters != list(range(1, len(final_quarters) + 1)):
            raise ValueError(
                "HFCE dataset final year must contain sequential quarters starting at Q1"
            )

    LOGGER.info(
        "Validated %s observed category-quarter rows across %s categories from %s to %s",
        f"{len(dataset):,}",
        len(categories),
        timeline["year_quarter"].iloc[0],
        timeline["year_quarter"].iloc[-1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 quarterly category-level HFCE data.")
    parser.add_argument("--input", default="RAW/HFCE/hfce_by_purpose.csv")
    parser.add_argument(
        "--output", default="training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv"
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.input, args.start_year, args.end_year)
    validate_dataset(
        dataset,
        expected_categories=set(PURPOSES.values()),
        require_continuous_periods=True,
    )
    dataset.to_csv(output, index=False)
    LOGGER.info("Wrote validated HFCE training dataset to %s", output)


if __name__ == "__main__":
    main()
