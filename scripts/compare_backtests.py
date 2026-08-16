#!/usr/bin/env python3
"""Compare MoE and naive backtest errors by unique_id."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"unique_id", "ds", "actuals", "forecasts"}


def load_forecast(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["ds"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["forecasts"].notna()].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no backtest forecast rows")
    return frame.rename(columns={"actuals": f"actuals_{label}", "forecasts": f"forecasts_{label}"})


def metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    actuals = frame[f"actuals_{label}"]
    forecasts = frame[f"forecasts_{label}"]
    errors = actuals - forecasts
    grouped = frame.assign(
        _abs_error=errors.abs(),
        _error=errors,
        _actuals=actuals,
    ).groupby("unique_id", sort=True)
    result = grouped.agg(
        **{
            f"{label}_abs_error": ("_abs_error", "sum"),
            f"{label}_error": ("_error", "sum"),
            f"{label}_actuals_sum": ("_actuals", "sum"),
            f"{label}_n_forecasts": ("_error", "size"),
        }
    ).reset_index()
    denominator = result[f"{label}_actuals_sum"].replace(0, np.nan)
    result[f"{label}_bias"] = result[f"{label}_error"] / denominator
    result[f"{label}_mape"] = result[f"{label}_abs_error"] / denominator
    result[f"{label}_fq"] = result[f"{label}_bias"].abs() + result[f"{label}_mape"].abs()
    return result.drop(columns=[f"{label}_actuals_sum"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe", default="m5_ca3_backtest.csv")
    parser.add_argument("--naive", default="m5_ca3_naive_backtest.csv")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--output", default="backtest_comparison.csv")
    args = parser.parse_args()

    moe = load_forecast(Path(args.moe), "moe")
    baseline = load_forecast(Path(args.naive), args.baseline_label)
    keys = ["unique_id", "ds"]
    merged = moe.merge(baseline, on=keys, how="inner")
    if merged.empty:
        raise ValueError("the backtest files have no matching unique_id/date forecast rows")
    if not np.allclose(
        merged["actuals_moe"].to_numpy(), merged[f"actuals_{args.baseline_label}"].to_numpy(), equal_nan=True
    ):
        raise ValueError("actuals differ between the two backtest files")

    comparison = metrics(merged, "moe").merge(
        metrics(merged, args.baseline_label), on="unique_id", how="outer"
    )
    comparison["better_by_fq"] = np.where(
        comparison["moe_fq"] < comparison[f"{args.baseline_label}_fq"], "moe",
        np.where(comparison[f"{args.baseline_label}_fq"] < comparison["moe_fq"], args.baseline_label, "tie"),
    )
    comparison.to_csv(args.output, index=False)

    print(f"Compared {len(merged)} matching backtest rows across {len(comparison)} unique_ids")
    print(f"Wrote comparison table to {args.output}")
    print("\nAggregate metrics across all matching backtest rows:")
    for label in ("moe", args.baseline_label):
        actuals = merged[f"actuals_{label}"]
        errors = actuals - merged[f"forecasts_{label}"]
        abs_error = errors.abs().sum()
        error = errors.sum()
        denominator = actuals.sum()
        bias = error / denominator if denominator else np.nan
        mape = abs_error / denominator if denominator else np.nan
        fq = abs(bias) + abs(mape)
        print(
            f"{label}: ABS_ERROR={abs_error:.4f} ERROR={error:.4f} "
            f"BIAS={bias:.6f} MAPE={mape:.6f} FQ={fq:.6f}"
        )


if __name__ == "__main__":
    main()
