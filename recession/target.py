"""Target construction.

Definition (documented in the README):
    y_t = 1 if USREC equals 1 in ANY of the months t+1, ..., t+h   (h = 12)
    y_t = 0 if USREC equals 0 in all of those months
    y_t = NaN if any of t+1..t+h is not yet observed (the label is not known yet)

Models are trained and scored only on months where the economy is in expansion
(USREC_t = 0). For those months y_t = 1 means exactly "a recession starts within the
next 12 months". Months already in recession are excluded: forecasting that a
recession that is under way will still be under way next month is not informative.

USREC_t itself is never used in y_t, and nothing later than t+h is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_target(usrec: pd.Series, horizon: int = 12) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fut = pd.concat([usrec.shift(-k) for k in range(1, horizon + 1)], axis=1)
    complete = fut.notna().all(axis=1)
    y = fut.max(axis=1).where(complete, np.nan)
    y.name = f"rec_next_{horizon}m"
    return y


def recession_starts(usrec: pd.Series) -> list[pd.Timestamp]:
    """First month of each recession (USREC switches 0 -> 1)."""
    u = usrec.dropna().astype(int)
    switch = (u == 1) & (u.shift(1) == 0)
    return list(u.index[switch])


def label_known_through(t: pd.Timestamp, horizon: int = 12) -> pd.Timestamp:
    """Last USREC month needed to know y_t."""
    return t + pd.DateOffset(months=horizon)
