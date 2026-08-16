#!/usr/bin/env python3
"""Drop series that do not have the full observation count."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_dropped{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="m5_ca3_sub.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = Path(args.csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    frame = pd.read_csv(input_path)
    required = {"unique_id", "ds", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    counts = frame.groupby("unique_id").size()
    full_count = counts.max()
    keep_ids = counts[counts == full_count].index
    filtered = frame[frame["unique_id"].isin(keep_ids)].copy()

    destination = args.output or output_path(input_path)
    filtered.to_csv(destination, index=False)

    print(f"Full observation count: {full_count}")
    print(f"Kept {len(keep_ids)} of {len(counts)} unique_ids")
    print(f"Dropped {len(counts) - len(keep_ids)} unique_ids")
    print(f"Wrote {len(filtered)} rows to {destination}")


if __name__ == "__main__":
    main()
