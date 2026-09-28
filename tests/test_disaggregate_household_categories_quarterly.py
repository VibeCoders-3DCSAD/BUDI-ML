from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from training.scripts.data.clean_raw_fies import CATEGORIES
from training.scripts.data.disaggregate_household_categories_quarterly import (
    ALLOCATION_METHOD,
    disaggregate_households,
)


def test_disaggregate_households_uses_category_specific_hfce_quarter_weights(
    tmp_path: Path,
) -> None:
    hfce = tmp_path / "hfce.csv"
    periods = ",".join(
        f'"At {basis} Prices 2021 Q{quarter}"'
        for quarter in range(1, 5)
        for basis in ("Current", "Constant 2018")
    )
    purposes = {
        "food": "Food and non-alcoholic beverages",
        "alcohol_tobacco": "Alcoholic beverages, tobacco",
        "clothing_footwear": "Clothing and footwear",
        "housing_water_utilities": "Housing, water, electricity, gas and other fuels",
        "furnishings_household": "Furnishings, household equipment and routine household maintenance",
        "health": "Health",
        "transport": "Transport",
        "communication": "Communication",
        "recreation_culture": "Recreation and culture",
        "education": "Education",
        "restaurants_hotels": "Restaurants and hotels",
        "miscellaneous_goods_services": "Miscellaneous goods and services",
    }
    rows = []
    for category in CATEGORIES:
        levels = ",".join(str(value) for quarter in range(1, 5) for value in (quarter, quarter))
        rows.append(f'"..{purposes[category]}",{levels}')
    hfce.write_text(
        '"Household Final Consumption Expenditure by Purpose"\n'
        f'"Purpose",{periods}\n' + "\n".join(rows) + "\n"
    )
    households = {"household_id": ["fies_a"], "year": [2021]}
    households.update({category: [100.0] for category in CATEGORIES})

    result = disaggregate_households(pd.DataFrame(households), hfce)

    assert result["quarter"].tolist() == [1, 2, 3, 4]
    assert result["food"].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert result[list(CATEGORIES)].sum().tolist() == pytest.approx([100.0] * len(CATEGORIES))
    assert set(result["observation_status"]) == {"hfce_temporally_disaggregated"}
    assert set(result["allocation_method"]) == {ALLOCATION_METHOD}
