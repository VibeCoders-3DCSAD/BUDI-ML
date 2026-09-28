from __future__ import annotations

import json
from pathlib import Path

from training.scripts.data.build_fies_category_anchors_v4 import build_anchors


def test_build_anchors_maps_categories_and_income(tmp_path: Path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "categories": {
                    "food": {"fies_columns": ["FOOD"]},
                    "alcohol_tobacco": {"fies_columns": ["ALCOHOL", "TOBACCO"]},
                }
            }
        )
    )
    source = tmp_path / "fies_2023.csv"
    source.write_text("SEQ_NO,TOTEX,TOINC,FOOD,ALCOHOL,TOBACCO\n1,200,1200,100,20,10\n")

    result = build_anchors(source, mapping)

    assert result["category"].tolist() == ["alcohol_tobacco", "food"]
    assert result["annual_expense_php"].tolist() == [30, 100]
    assert result["annual_income_php"].tolist() == [1200, 1200]
    assert result["monthly_income_proxy_php"].tolist() == [100.0, 100.0]
    assert result["unmapped_expense_residual_php"].tolist() == [70, 70]
