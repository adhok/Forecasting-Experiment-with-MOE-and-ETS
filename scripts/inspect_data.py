#!/usr/bin/env python3
"""Inspect the first rows and column types of a CSV dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="m5_ca3_sub.csv")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")

    print("\nFirst 10 rows:")
    print(frame.head(10).to_string(index=False))

    print("\nColumn data types:")
    print(frame.dtypes.rename("dtype").to_string())

    print("\nMissing values by column:")
    print(frame.isna().sum().rename("missing_values").to_string())

    if {"unique_id", "ds"}.issubset(frame.columns):
        dates = pd.to_datetime(frame["ds"], errors="coerce")
        summary = (
            frame.assign(_date=dates)
            .groupby("unique_id", sort=True)
            .agg(
                data_points=("unique_id", "size"),
                first_date=("_date", "min"),
                last_date=("_date", "max"),
            )
        )
        print("\nData points per unique_id:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
