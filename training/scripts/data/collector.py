#!/usr/bin/env python3
"""Odin ML — Data Collector.

Converts raw CSV datasets to Parquet format for pipeline consumption.

Usage:
    # Convert single file
    python scripts/collector.py --input datasets/raw/family_income_and_expenditure.csv

    # Convert all CSVs in directory
    python scripts/collector.py --input datasets/raw/ --output datasets/unprocessed/
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: str, output_dir: str) -> str:
    """Convert a single CSV file to Parquet format.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Directory to write the Parquet file.

    Returns:
        Path to the created Parquet file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.suffix.lower() == ".csv":
        raise ValueError(f"Expected a CSV file, got: {csv_path.suffix}")

    parquet_name = csv_path.stem + ".parquet"
    parquet_path = Path(output_dir) / parquet_name

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)

    csv_size = csv_path.stat().st_size / 1e6
    pq_size = parquet_path.stat().st_size / 1e6
    ratio = (1 - pq_size / csv_size) * 100 if csv_size > 0 else 0

    print(f"  {csv_path.name}: {csv_size:.1f}MB CSV -> {pq_size:.1f}MB Parquet ({len(df):,} rows, {ratio:.0f}% smaller)")

    return str(parquet_path)


def main():
    parser = argparse.ArgumentParser(
        description="Odin ML — Data Collector (CSV to Parquet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/unprocessed",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    converted = 0
    errors = []

    if input_path.is_file():
        if not input_path.suffix.lower() == ".csv":
            print(f"Error: Expected a CSV file, got: {input_path.suffix}")
            sys.exit(1)
        try:
            convert_csv_to_parquet(str(input_path), str(output_dir))
            converted = 1
        except Exception as e:
            errors.append((str(input_path), str(e)))

    elif input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in {input_path}")
            sys.exit(0)

        print(f"Found {len(csv_files)} CSV files in {input_path}")
        for csv_file in csv_files:
            try:
                convert_csv_to_parquet(str(csv_file), str(output_dir))
                converted += 1
            except Exception as e:
                errors.append((str(csv_file), str(e)))

    print(f"\nConverted {converted} file(s) to {output_dir}/")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
