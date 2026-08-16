#!/usr/bin/env python3
"""Run a small local training smoke test on generated panel data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Allow `python scripts/train_smoke.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PanelWindowDataset
from src.models import ForecasterConfig, MoEForecaster
from src.training.losses import total_loss


def tensor_collate(samples):
    """Batch only tensors; timestamp and ID metadata are not needed for training."""
    return {
        "y": torch.stack([sample["y"] for sample in samples]),
        "future": torch.stack([sample["future"] for sample in samples]),
        "time_features": torch.stack([sample["time_features"] for sample in samples]),
        "future_time_features": torch.stack([sample["future_time_features"] for sample in samples]),
        "series_index": torch.tensor([sample["series_index"] for sample in samples], dtype=torch.long),
    }


def forecast_collate(samples):
    batch = tensor_collate(samples)
    batch["unique_id"] = [sample["unique_id"] for sample in samples]
    batch["cutoff_ds"] = [sample["cutoff_ds"] for sample in samples]
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dummy_panel.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--lags", default="7,14,28", help="comma-separated target lags (default: 7,14,28)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--forecast-output",
        default="final_forecasts.csv",
        help="CSV path for final forecasts",
    )
    parser.add_argument(
        "--backtest-cutoff",
        help="historical cutoff date; output history plus the following 13-day backtest",
    )
    parser.add_argument(
        "--series-fraction",
        type=float,
        default=0.10,
        help="fraction of unique_id series to use (default: 0.10)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--amp", action="store_true", help="use CUDA mixed precision")
    parser.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    args = parser.parse_args()

    if not 0 < args.series_fraction <= 1:
        raise ValueError("--series-fraction must be greater than 0 and at most 1")
    if args.lookback < 1:
        raise ValueError("--lookback must be positive")
    try:
        lags = tuple(int(value) for value in args.lags.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--lags must be a comma-separated list of integers") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = pd.read_csv(args.data, parse_dates=["ds"])
    unique_ids = frame["unique_id"].drop_duplicates()
    n_series = max(1, int(len(unique_ids) * args.series_fraction))
    selected_ids = unique_ids.sample(n=n_series, random_state=args.seed)
    frame = frame[frame["unique_id"].isin(selected_ids)].copy()
    print(f"Using {n_series} of {len(unique_ids)} unique_ids ({n_series / len(unique_ids):.1%})")

    dataset = PanelWindowDataset(frame, lookback=args.lookback, horizon=13, frequency="D", lags=lags)
    if not len(dataset):
        raise ValueError("dummy data does not contain any complete lookback/future windows")

    config = ForecasterConfig(lookback=args.lookback, horizon=13)
    backtest_cutoff = pd.Timestamp(args.backtest_cutoff) if args.backtest_cutoff else None
    if backtest_cutoff is None:
        training_dataset = dataset
        forecast_dataset = dataset
    else:
        training_indices = []
        forecast_indices = []
        for index, window in enumerate(dataset.windows):
            window_cutoff = pd.Timestamp(window.cutoff_ds)
            target_end = window_cutoff + pd.Timedelta(days=config.horizon)
            if target_end <= backtest_cutoff:
                training_indices.append(index)
            if window_cutoff == backtest_cutoff:
                forecast_indices.append(index)
        if not training_indices:
            raise ValueError("backtest cutoff leaves no training windows")
        if len(forecast_indices) != frame["unique_id"].nunique():
            raise ValueError("backtest cutoff must exist for every selected unique_id")
        training_dataset = Subset(dataset, training_indices)
        forecast_dataset = Subset(dataset, forecast_indices)
        print(f"Backtest cutoff: {backtest_cutoff.date()}")
        print(f"Training windows: {len(training_dataset)}")
        print(f"Backtest windows: {len(forecast_dataset)}")

    loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=tensor_collate)
    sample = dataset[0]
    model = MoEForecaster(
        time_feature_dim=sample["time_features"].shape[-1],
        config=config,
        n_series=len(dataset.series_to_index),
        future_feature_dim=sample["future_time_features"].shape[-1],
    ).to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a newer PyTorch version")
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_total = epoch_forecast = epoch_balance = 0.0
        for batch in loader:
            optimizer.zero_grad()
            y = batch["y"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            features = batch["time_features"].to(device, non_blocking=True)
            future_features = batch["future_time_features"].to(device, non_blocking=True)
            series_index = batch["series_index"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction, diagnostics = model(y, features, series_index, future_features=future_features)
                losses = total_loss(prediction, future, diagnostics, kind="huber")
            scaler.scale(losses["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_total += losses["total_loss"].item()
            epoch_forecast += losses["forecast_loss"].item()
            epoch_balance += losses["routing_balance_loss"].item()

        count = len(loader)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"total_loss={epoch_total / count:.5f} "
            f"forecast_loss={epoch_forecast / count:.5f} "
            f"routing_balance_loss={epoch_balance / count:.5f}"
        )

    model.eval()
    forecast_loader = DataLoader(
        forecast_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=forecast_collate
    )
    rows = []
    if backtest_cutoff is not None:
        historical = frame[frame["ds"] <= backtest_cutoff]
        rows.extend(
            {
                "unique_id": row.unique_id,
                "ds": row.ds,
                "actuals": float(row.y),
                "forecasts": float("nan"),
            }
            for row in historical.itertuples(index=False)
        )
    with torch.inference_mode():
        for batch in forecast_loader:
            y = batch["y"].to(device, non_blocking=True)
            features = batch["time_features"].to(device, non_blocking=True)
            future_features = batch["future_time_features"].to(device, non_blocking=True)
            series_index = batch["series_index"].to(device, non_blocking=True)
            prediction = model(y, features, series_index, return_diagnostics=False, future_features=future_features)
            prediction = prediction.cpu()
            actual = batch["future"]
            for i, (uid, cutoff) in enumerate(zip(batch["unique_id"], batch["cutoff_ds"])):
                dates = pd.date_range(
                    start=pd.Timestamp(cutoff) + pd.Timedelta(days=1),
                    periods=prediction.shape[1],
                    freq="D",
                )
                for step, date in enumerate(dates):
                    if backtest_cutoff is not None:
                        rows.append(
                            {
                                "unique_id": uid,
                                "ds": date,
                                "actuals": float(actual[i, step]),
                                "forecasts": float(prediction[i, step]),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "unique_id": uid,
                                "cutoff_ds": cutoff,
                                "ds": date,
                                "horizon_step": step + 1,
                                "y_actual": float(actual[i, step]),
                                "y_pred": float(prediction[i, step]),
                            }
                        )

    forecast_output = Path(args.forecast_output)
    output_frame = pd.DataFrame(rows).sort_values(["unique_id", "ds"])
    output_frame.to_csv(forecast_output, index=False)
    print(f"Wrote {len(rows)} forecasts to {forecast_output}")


if __name__ == "__main__":
    main()
