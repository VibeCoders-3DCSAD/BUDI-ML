from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from training.scripts.data.clean_raw_fies import CATEGORIES
from training.scripts.data.minimize_fies_anchors import (
    WIDE_COLUMNS,
    absorb_anchor_batch,
    minimize_anchors,
)


def _long_anchors(*households: tuple[str, float]) -> pd.DataFrame:
    rows = []
    for category_index, category in enumerate(CATEGORIES):
        for household_id, base in households:
            rows.append(
                {
                    "household_id": household_id,
                    "category": category,
                    "amount_php": base + category_index,
                    "survey_year": 2021,
                }
            )
    return pd.DataFrame(rows)


def test_absorb_anchor_batch_waits_until_every_category_arrives() -> None:
    pending: dict[tuple[str, int], dict[str, float]] = {}
    first = pd.DataFrame(
        {"household_id": ["fies_a"], "survey_year": [2021], "category": ["food"], "amount_php": [10.0]}
    )
    assert absorb_anchor_batch(pending, first).empty
    assert list(pending) == [("fies_a", 2021)]

    rest = pd.DataFrame(
        {
            "household_id": ["fies_a"] * (len(CATEGORIES) - 1),
            "survey_year": [2021] * (len(CATEGORIES) - 1),
            "category": list(CATEGORIES[1:]),
            "amount_php": list(range(1, len(CATEGORIES))),
        }
    )
    wide = absorb_anchor_batch(pending, rest)

    assert pending == {}
    assert wide["household_id"].tolist() == ["fies_a"]
    assert wide["food"].tolist() == [10.0]
    assert list(wide.columns) == list(WIDE_COLUMNS)


def test_minimize_anchors_streams_row_groups_into_one_row_per_household(tmp_path: Path) -> None:
    source = tmp_path / "annual_anchors.parquet"
    first = _long_anchors(("fies_a", 100.0))
    second = _long_anchors(("fies_b", 200.0))
    writer = pq.ParquetWriter(
        source, pa.Table.from_pandas(first, preserve_index=False).schema, compression="snappy"
    )
    writer.write_table(pa.Table.from_pandas(first, preserve_index=False))
    writer.write_table(pa.Table.from_pandas(second, preserve_index=False))
    writer.close()

    counts = minimize_anchors(source, tmp_path, batch_size=2)
    wide = pd.read_parquet(tmp_path / "household_categories.parquet")
    csv_rows = pd.read_csv(tmp_path / "household_categories.csv")

    assert counts == {"anchor_rows": len(CATEGORIES) * 2, "households": 2}
    assert list(wide.columns) == list(WIDE_COLUMNS)
    assert wide["household_id"].tolist() == ["fies_a", "fies_b"]
    assert wide["year"].tolist() == [2021, 2021]
    assert wide["food"].tolist() == [100.0, 200.0]
    assert "housing_water_utilities" in wide.columns
    assert "survey_year" not in wide.columns
    assert csv_rows["year"].tolist() == [2021, 2021]
    assert csv_rows["household_id"].tolist() == ["fies_a", "fies_b"]


def test_minimize_anchors_rejects_a_household_missing_a_category(tmp_path: Path) -> None:
    source = tmp_path / "annual_anchors.parquet"
    frame = _long_anchors(("fies_a", 1.0))
    frame = frame.loc[frame["category"] != "transport"]
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="fies_a"):
        minimize_anchors(source, tmp_path, batch_size=3)

    assert not (tmp_path / "household_categories.parquet").exists()
    assert not (tmp_path / "household_categories.csv").exists()
