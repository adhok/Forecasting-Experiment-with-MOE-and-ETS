from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import time_features


@dataclass(frozen=True)
class Window:
    unique_id: object
    series_index: int
    cutoff_ds: object
    y: torch.Tensor
    future: torch.Tensor
    time_features: torch.Tensor
    future_time_features: torch.Tensor


class PanelWindowDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        lookback: int = 52,
        horizon: int = 13,
        frequency: str | None = None,
        lags: tuple[int, ...] = (),
    ):
        required = {"unique_id", "ds", "y"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        self.lookback, self.horizon = lookback, horizon
        if any(lag < 1 for lag in lags):
            raise ValueError("lags must contain positive integers")
        self.lags = lags
        self.windows: list[Window] = []
        self.series_to_index: dict[object, int] = {}
        for series_index, (uid, group) in enumerate(
            frame.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False)
        ):
            self.series_to_index[uid] = series_index
            group = group.reset_index(drop=True)
            values = group["y"].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(group["ds"])
            all_time_features = time_features(dates, frequency)
            all_lag_values = []
            for lag in lags:
                shifted = np.empty_like(values)
                shifted[:lag] = values[0]
                shifted[lag:] = values[:-lag]
                all_lag_values.append(shifted)
            all_lag_values = np.column_stack(all_lag_values) if lags else None
            for end in range(lookback - 1, len(group) - horizon):
                start = end - lookback + 1
                window_values = values[start:end + 1]
                mu = window_values.mean()
                sigma = max(window_values.std(), 1e-6)
                window_features = all_time_features[start:end + 1]
                future_features = time_features(dates.iloc[end + 1:end + horizon + 1], frequency)
                if all_lag_values is not None:
                    normalized_lags = (all_lag_values[start:end + 1] - mu) / sigma
                    window_features = np.column_stack([window_features, normalized_lags])
                self.windows.append(Window(uid, series_index, dates.iloc[end], torch.from_numpy(values[start:end + 1]),
                                           torch.from_numpy(values[end + 1:end + horizon + 1]),
                                           torch.from_numpy(window_features.astype(np.float32)),
                                           torch.from_numpy(future_features.astype(np.float32))))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        w = self.windows[index]
        return {"unique_id": w.unique_id, "series_index": w.series_index, "cutoff_ds": w.cutoff_ds, "y": w.y, "future": w.future, "time_features": w.time_features, "future_time_features": w.future_time_features}
