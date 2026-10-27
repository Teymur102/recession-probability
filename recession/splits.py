"""Expanding-window, annually refit pseudo-out-of-sample splitter.

At each refit origin (January of each year) the forecaster is assumed to know USREC
through the previous month minus an optional publication lag. A training row t is
usable only if its whole label window t+1..t+h is inside that known range:

    t + h <= origin - 1 month - label_lag months

The model fitted at the origin then predicts the 12 months origin .. origin+11 using
features observed in each of those months. Every out-of-sample month is predicted
exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    origin: pd.Timestamp
    info_end: pd.Timestamp  # last USREC month the forecaster may use
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def expanding_window_splits(
    index: pd.DatetimeIndex,
    oos_start: str | pd.Timestamp,
    oos_end: str | pd.Timestamp | None = None,
    horizon: int = 12,
    label_lag: int = 0,
    refit_months: int = 12,
) -> list[Split]:
    index = pd.DatetimeIndex(index).sort_values()
    oos_start = pd.Timestamp(oos_start)
    oos_end = pd.Timestamp(oos_end) if oos_end is not None else index[-1]
    splits: list[Split] = []
    origin = oos_start
    while origin <= oos_end:
        info_end = origin - pd.DateOffset(months=1 + label_lag)
        last_train_t = info_end - pd.DateOffset(months=horizon)
        train = index[index <= last_train_t]
        nxt = origin + pd.DateOffset(months=refit_months)
        test = index[(index >= origin) & (index < nxt) & (index <= oos_end)]
        if len(test):
            splits.append(Split(origin, info_end, train, test))
        origin = nxt
    return splits
