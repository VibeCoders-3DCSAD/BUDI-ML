"""Clean annual FIES household data into v4 HFCE-aligned category anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

YEAR = re.compile(r"20\d{2}")
DEFAULT_MAPPING = (
    Path(__file__).resolve().parents[2] / "config" / "fies_hfce_category_mapping_v4.json"
)


def _survey_year(path: Path) -> int:
    match = YEAR.search(path.name)
    if match is None:
        raise ValueError(f"Unable to infer FIES survey year from filename: {path.name}")
    return int(match.group())


def _household_id(year: int, sequence_number: object) -> str:
    value = f"{year}:{str(sequence_number).strip()}".encode()
    return f"fies_{hashlib.sha256(value).hexdigest()[:24]}"


def build_anchors(source: str | Path, mapping_path: str | Path = DEFAULT_MAPPING) -> pd.DataFrame:
    """Return valid annual household-category anchors from one FIES source file."""
    path = Path(source)
    survey_year = _survey_year(path)
    mapping = json.loads(Path(mapping_path).read_text())
    categories: dict[str, dict[str, object]] = mapping["categories"]
    required = {"SEQ_NO", "TOTEX", "TOINC"}
    required.update(column for item in categories.values() for column in item["fies_columns"])
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required FIES columns: {sorted(missing)}")
    numeric_columns = list(required)
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric_columns).copy()
    if frame["SEQ_NO"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate SEQ_NO values")
    if (frame[["TOTEX", "TOINC"]] < 0).any().any():
        raise ValueError(f"{path.name} contains negative total expenditure or income")

    category_values = pd.DataFrame(
        {category: frame[item["fies_columns"]].sum(axis=1) for category, item in categories.items()}
    )
    if (category_values < 0).any().any():
        raise ValueError(f"{path.name} contains negative category expenditure")
    mapped_total = category_values.sum(axis=1)
    residual = frame["TOTEX"] - mapped_total
    if (residual < -0.005).any():
        raise ValueError(f"{path.name} maps category expenditure above TOTEX")

    anchors = category_values.assign(
        household_id=frame["SEQ_NO"].map(lambda value: _household_id(survey_year, value)),
        survey_year=survey_year,
        annual_income_php=frame["TOINC"],
        monthly_income_proxy_php=frame["TOINC"] / 12,
        annual_total_expense_php=frame["TOTEX"],
        unmapped_expense_residual_php=residual.clip(lower=0),
    ).melt(
        id_vars=[
            "household_id",
            "survey_year",
            "annual_income_php",
            "monthly_income_proxy_php",
            "annual_total_expense_php",
            "unmapped_expense_residual_php",
        ],
        var_name="category",
        value_name="annual_expense_php",
    )
    return anchors.sort_values(["household_id", "category"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v4 HFCE-aligned annual FIES category anchors."
    )
    parser.add_argument("--input-dir", default="training/datasets/fies")
    parser.add_argument(
        "--output", default="training/datasets/fies_hfce_v4/fies_category_anchors.parquet"
    )
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    args = parser.parse_args()
    sources = sorted(Path(args.input_dir).glob("*.csv"))
    if not sources:
        raise ValueError(f"No FIES CSV files found in {args.input_dir}")
    anchors = pd.concat([build_anchors(path, args.mapping) for path in sources], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
