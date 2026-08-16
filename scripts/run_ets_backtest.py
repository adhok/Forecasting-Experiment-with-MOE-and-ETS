#!/usr/bin/env python3
"""Generate a weekly seasonal Holt-Winters/ETS backtest file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="m5_ca3_sub_dropped.csv")
    parser.add_argument("--series-fraction", type=float, default=0.10)
    parser.add_argument("--backtest-cutoff", required=True)
    parser.add_argument("--output", default="m5_ca3_ets_backtest.csv")
    parser.add_argument("--season-period", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=13)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not 0 < args.series_fraction <= 1:
        raise ValueError("--series-fraction must be greater than 0 and at most 1")
    if args.season_period < 2 or args.horizon < 1:
        raise ValueError("--season-period must be at least 2 and horizon must be positive")

    frame = pd.read_csv(args.data, parse_dates=["ds"])
    required = {"unique_id", "ds", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    all_ids = frame["unique_id"].drop_duplicates()
    n_series = max(1, int(len(all_ids) * args.series_fraction))
    selected_ids = all_ids.sample(n=n_series, random_state=args.seed)
    frame = frame[frame["unique_id"].isin(selected_ids)].copy()
    cutoff = pd.Timestamp(args.backtest_cutoff)
    print(f"Using {n_series} of {len(all_ids)} unique_ids ({n_series / len(all_ids):.1%})")

    rows = []
    for uid, group in frame.groupby("unique_id", sort=True):
        group = group.sort_values("ds")
        history = group[group["ds"] <= cutoff]
        future = group[
            (group["ds"] > cutoff)
            & (group["ds"] <= cutoff + pd.Timedelta(days=args.horizon))
        ]
        if len(history) < 2 * args.season_period:
            raise ValueError(f"{uid} needs at least two seasonal cycles before the cutoff")
        if len(future) != args.horizon:
            raise ValueError(f"{uid} does not have {args.horizon} complete future dates after cutoff")

        rows.extend(
            {
                "unique_id": uid,
                "ds": row.ds,
                "actuals": float(row.y),
                "forecasts": float("nan"),
            }
            for row in history.itertuples(index=False)
        )

        model = ExponentialSmoothing(
            history["y"].astype(float).to_numpy(),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=args.season_period,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, remove_bias=False)
        predictions = fitted.forecast(args.horizon)
        for row, prediction in zip(future.itertuples(index=False), predictions):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": row.ds,
                    "actuals": float(row.y),
                    "forecasts": max(0.0, float(prediction)),
                }
            )

    output_path = Path(args.output)
    pd.DataFrame(rows).sort_values(["unique_id", "ds"]).to_csv(output_path, index=False)
    print("Method: additive damped-trend Holt-Winters with weekly seasonality")
    print(f"Backtest cutoff: {cutoff.date()}")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
