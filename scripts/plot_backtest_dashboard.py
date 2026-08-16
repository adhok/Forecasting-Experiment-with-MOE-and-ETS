#!/usr/bin/env python3
"""Create an interactive Plotly dashboard for backtest forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot


REQUIRED_COLUMNS = {"unique_id", "ds", "actuals", "forecasts"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="m5_ca3_backtest.csv")
    parser.add_argument("--output", default="backtest_dashboard.html")
    parser.add_argument("--open", action="store_true", help="open the dashboard after creating it")
    args = parser.parse_args()

    input_path = Path(args.csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    frame = pd.read_csv(input_path, parse_dates=["ds"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.sort_values(["unique_id", "ds"])
    unique_ids = frame["unique_id"].drop_duplicates().tolist()
    if not unique_ids:
        raise ValueError("the CSV contains no unique_id values")

    figure = go.Figure()
    visibility = []
    for series_number, unique_id in enumerate(unique_ids):
        series = frame[frame["unique_id"] == unique_id]
        figure.add_trace(
            go.Scatter(
                x=series["ds"],
                y=series["actuals"],
                mode="lines",
                name="Actuals",
                visible=series_number == 0,
                line={"color": "#1f77b4"},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=series["ds"],
                y=series["forecasts"],
                mode="lines+markers",
                name="Forecasts",
                visible=series_number == 0,
                line={"color": "#d62728", "dash": "dash"},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
            )
        )
        visibility.extend([series_number == 0, series_number == 0])

    buttons = []
    for series_number, unique_id in enumerate(unique_ids):
        selected = [False] * len(visibility)
        selected[2 * series_number : 2 * series_number + 2] = [True, True]
        buttons.append(
            {
                "label": str(unique_id),
                "method": "update",
                "args": [
                    {"visible": selected},
                    {"title": f"Backtest forecast: {unique_id}"},
                ],
            }
        )

    figure.update_layout(
        title=f"Backtest forecast: {unique_ids[0]}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        margin={"t": 110},
    )

    output_path = Path(args.output)
    plot(figure, filename=str(output_path), auto_open=args.open, include_plotlyjs=True)
    print(f"Wrote dashboard for {len(unique_ids)} unique_ids to {output_path}")


if __name__ == "__main__":
    main()
