from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(ds: pd.Series, frequency: str | None = None) -> np.ndarray:
    """Return deterministic cyclical features appropriate for the observed dates."""
    dates = pd.to_datetime(ds)
    out: list[np.ndarray] = []
    # These features are useful at any frequency and do not use the target.
    if frequency is None or frequency in {"D", "B", "H", "W", "M", "MS"}:
        dow = dates.dt.dayofweek.to_numpy(float)
        out += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
    if frequency is None or frequency in {"D", "B", "W", "M", "MS"}:
        week = dates.dt.isocalendar().week.to_numpy(float)
        out += [np.sin(2 * np.pi * (week - 1) / 52), np.cos(2 * np.pi * (week - 1) / 52)]
        month = dates.dt.month.to_numpy(float)
        out += [np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)]
    if frequency is None or frequency in {"D", "B"}:
        day_of_year = dates.dt.dayofyear.to_numpy(float)
        out += [
            np.sin(2 * np.pi * (day_of_year - 1) / 365.25),
            np.cos(2 * np.pi * (day_of_year - 1) / 365.25),
        ]
    if frequency is None or frequency in {"M", "MS"}:
        # Month-of-year is already included; no extra target-derived feature needed.
        pass
    n = max(len(dates) - 1, 1)
    out.append(np.arange(len(dates), dtype=float) / n)
    return np.column_stack(out).astype(np.float32)
