"""Pivot long FIES anchors into one row per household, one column per category."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training.scripts.data.clean_raw_fies import CATEGORIES, append_csv

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 50_000
WIDE_COLUMNS = ("household_id", "year", *CATEGORIES)
CATEGORY_SET = set(CATEGORIES)


def absorb_anchor_batch(
    pending: dict[tuple[str, int], dict[str, float]], frame: pd.DataFrame
) -> pd.DataFrame:
    """Add one anchor batch to partial households and return any newly complete rows."""
    required = {"household_id", "survey_year", "category", "amount_php"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Anchor batch is missing columns: {sorted(missing)}")
    completed: list[dict[str, object]] = []
    columns = frame.loc[:, ["household_id", "survey_year", "category", "amount_php"]]
    for household_id, survey_year, category, amount in columns.itertuples(index=False, name=None):
        if pd.isna(household_id):
            raise ValueError("Anchor row is missing household_id")
        if pd.isna(survey_year):
            raise ValueError(f"{household_id} is missing survey_year")
        if category not in CATEGORY_SET:
            raise ValueError(f"Unknown category: {category}")
        if pd.isna(amount):
            raise ValueError(f"{household_id} has a missing {category} amount")
        household_key = (str(household_id), int(survey_year))
        slot = pending.setdefault(household_key, {})
        if category in slot:
            raise ValueError(f"{household_key} repeats {category}")
        slot[category] = float(amount)
        if len(slot) == len(CATEGORIES):
            completed.append(
                {
                    "household_id": household_key[0],
                    "year": household_key[1],
                    **{name: slot[name] for name in CATEGORIES},
                }
            )
            del pending[household_key]
    if not completed:
        return pd.DataFrame(columns=list(WIDE_COLUMNS))
    wide = pd.DataFrame(completed, columns=list(WIDE_COLUMNS))
    wide["household_id"] = wide["household_id"].astype("string")
    wide["year"] = wide["year"].astype("int64")
    for column in CATEGORIES:
        wide[column] = wide[column].astype("float64")
    return wide


def minimize_anchors(
    source: str | Path, output_dir: str | Path, batch_size: int = BATCH_SIZE
) -> dict[str, int]:
    """Stream long anchors into a wide household table without reading the whole file."""
    anchor_path = Path(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / "household_categories.parquet"
    csv_path = destination / "household_categories.csv"
    partial_parquet = parquet_path.with_suffix(".parquet.partial")
    partial_csv = csv_path.with_suffix(".csv.partial")
    partial_parquet.unlink(missing_ok=True)
    partial_csv.unlink(missing_ok=True)

    parquet = pq.ParquetFile(anchor_path)
    required = {"household_id", "survey_year", "category", "amount_php"}
    if required - set(parquet.schema.names):
        raise ValueError(f"{anchor_path} is missing required anchor columns")

    pending: dict[tuple[str, int], dict[str, float]] = {}
    writer: pq.ParquetWriter | None = None
    rows_read = 0
    households_written = 0
    try:
        for batch in parquet.iter_batches(
            columns=["household_id", "survey_year", "category", "amount_php"], batch_size=batch_size
        ):
            frame = batch.to_pandas()
            rows_read += len(frame)
            wide = absorb_anchor_batch(pending, frame)
            if wide.empty:
                LOGGER.info(
                    "Read %s anchor rows; %s households still partial",
                    f"{rows_read:,}",
                    f"{len(pending):,}",
                )
                continue
            table = pa.Table.from_pandas(wide, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial_parquet, table.schema, compression="snappy")
            writer.write_table(table)
            append_csv(wide, partial_csv)
            households_written += len(wide)
            LOGGER.info(
                "Read %s anchor rows; wrote %s households; %s still partial",
                f"{rows_read:,}",
                f"{households_written:,}",
                f"{len(pending):,}",
            )
        if pending:
            sample_id, slot = next(iter(pending.items()))
            missing = [name for name in CATEGORIES if name not in slot]
            raise ValueError(
                f"{len(pending)} households are missing categories, for example {sample_id}: {missing}"
            )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        partial_parquet.unlink(missing_ok=True)
        partial_csv.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    if writer is None and households_written == 0:
        empty = pd.DataFrame(columns=list(WIDE_COLUMNS))
        empty["household_id"] = empty["household_id"].astype("string")
        empty["year"] = empty["year"].astype("int64")
        for column in CATEGORIES:
            empty[column] = empty[column].astype("float64")
        pq.write_table(
            pa.Table.from_pandas(empty, preserve_index=False), partial_parquet, compression="snappy"
        )
        append_csv(empty, partial_csv)

    partial_parquet.replace(parquet_path)
    partial_csv.replace(csv_path)
    LOGGER.info("Wrote %s households to %s", f"{households_written:,}", destination)
    return {"anchor_rows": rows_read, "households": households_written}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=Path("RAW/cleaned/annual_anchors.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("RAW/cleaned"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    minimize_anchors(args.anchors, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
