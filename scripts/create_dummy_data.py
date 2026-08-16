#!/usr/bin/env python3
"""Create deterministic dummy panel data for end-to-end testing.

This only generates data; it does not import, train, or execute the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def create_dummy_panel(
    output: str | Path = "dummy_panel.csv",
    n_series: int = 4,
    periods: int = 104,
    start: str = "2023-01-01",
    frequency: str = "D",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate smooth, intermittent, seasonal, and volatile example series."""
    if n_series < 1 or periods < 1:
        raise ValueError("n_series and periods must be positive")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq=frequency)
    t = np.arange(periods, dtype=np.float32)
    rows: list[dict[str, object]] = []
    regimes = ("smooth", "intermittent", "seasonal", "volatile")

    for series_number in range(n_series):
        regime = regimes[series_number % len(regimes)]
        level = 20.0 + 3.0 * series_number
        if regime == "smooth":
            values = level + 0.08 * t + rng.normal(0, 0.35, periods)
        elif regime == "intermittent":
            arrivals = rng.random(periods) < 0.22
            values = arrivals * (level + rng.normal(0, 2.0, periods))
        elif regime == "seasonal":
            values = level + 5.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, periods)
        else:
            values = level + rng.normal(0, 5.0, periods)
        values = np.maximum(values, 0).astype(np.float32)
        rows.extend(
            {"unique_id": f"series_{series_number + 1}", "ds": date, "y": float(value)}
            for date, value in zip(dates, values)
        )

    result = pd.DataFrame(rows)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows across {n_series} series to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dummy_panel.csv")
    parser.add_argument("--n-series", type=int, default=4)
    parser.add_argument("--periods", type=int, default=104)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--frequency", default="D")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    create_dummy_panel(args.output, args.n_series, args.periods, args.start, args.frequency, args.seed)


if __name__ == "__main__":
    main()
