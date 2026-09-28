from __future__ import annotations

from pathlib import Path

import pandas as pd
from training.scripts.data.clean_raw_fies import (
    NCR_REGION_CODE,
    SPECS,
    append_csv,
    canonical_region_code,
    category_values,
)


def test_retained_rows_do_not_realign_against_excluded_regions() -> None:
    """Pandas outer-aligns Series by index, so excluded rows must not stay in the residual."""
    region = pd.Series(["01", "13", "05", "13"])
    row_invalid = pd.Series([False, False, False, True])
    ncr = canonical_region_code(region).eq(NCR_REGION_CODE).fillna(False)
    retain = ncr & ~row_invalid
    income = pd.Series([10.0, 20.0, 30.0, 40.0]).loc[retain].reset_index(drop=True)
    residual = pd.Series([1.0, 2.0, 3.0, 4.0]).loc[retain].reset_index(drop=True)
    households = pd.DataFrame(
        {
            "annual_income_php": income,
            "region_code": NCR_REGION_CODE,
            "unmapped_expense_residual_php": residual,
        }
    )

    assert len(households) == 1
    assert households["annual_income_php"].tolist() == [20.0]
    assert households["unmapped_expense_residual_php"].tolist() == [2.0]


def test_append_csv_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "households.csv"
    append_csv(pd.DataFrame({"region_code": ["13"]}), path)
    append_csv(pd.DataFrame({"region_code": ["13"]}), path)

    assert path.read_text().splitlines() == ["region_code", "13", "13"]


def test_canonical_region_code_normalizes_ncr_forms() -> None:
    codes = canonical_region_code(pd.Series(["13", "013", "13.0", "01", "1", None, ""]))

    assert codes.tolist()[:5] == ["13", "13", "13", "01", "01"]
    assert codes.iloc[:3].eq(NCR_REGION_CODE).all()
    assert not codes.iloc[3:].eq(NCR_REGION_CODE).fillna(False).any()


def test_category_values_preserves_total_food_including_outside_food() -> None:
    spec = SPECS[2012]
    fields = {field for category_fields in spec.base_fields.values() for field in category_fields}
    frame = pd.DataFrame({field: [0.0] for field in fields})
    frame["TFOOD"] = 12_000.0
    frame["TFOODOUTSIDE"] = 3_000.0
    frame["TRESTAURANT"] = 5_000.0

    values, _ = category_values(frame, spec)

    assert values["food"].tolist() == [12_000.0]
    assert values["restaurants_hotels"].tolist() == [5_000.0]
