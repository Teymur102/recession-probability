"""Load FRED series (cached in data/) and build the monthly modelling panel.

Every series is fetched from the public FRED CSV endpoint, which needs no API key:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
Downloaded files are kept in data/ and reused, so results reproduce from the cache.
Pass refresh=True (or `python run.py --refresh`) to re-download.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Series used by the models (monthly) plus T10Y3M / BAA10Y (daily) for cross-checks only.
SERIES = {
    "GS10": "10-year Treasury constant maturity yield, monthly avg (%)",
    "TB3MS": "3-month Treasury bill, secondary market, monthly avg (%)",
    "BAA": "Moody's seasoned Baa corporate bond yield, monthly avg (%)",
    "UNRATE": "Civilian unemployment rate, SA (%)",
    "USREC": "NBER-based recession indicator (1 = recession month)",
    "T10Y3M": "10y minus 3m Treasury spread, daily (cross-check only)",
    "BAA10Y": "Baa minus 10y Treasury spread, daily (cross-check only)",
}


def load_series(sid: str, refresh: bool = False, data_dir: Path = DATA_DIR) -> pd.Series:
    path = data_dir / f"{sid}.csv"
    if refresh or not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FRED_URL.format(sid=sid), timeout=60) as r:
            raw = r.read()
        path.write_bytes(raw)
    df = pd.read_csv(path)
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.to_numeric(df[val_col], errors="coerce")  # FRED uses "." for missing
    s.index = pd.to_datetime(df[date_col])
    s.name = sid
    return s.dropna()


def to_monthly(s: pd.Series) -> pd.Series:
    """Monthly average, indexed by month start."""
    return s.resample("MS").mean()


def sahm_gap(unrate: pd.Series) -> pd.Series:
    """Sahm-rule-style labour signal.

    3-month average unemployment rate minus its minimum over the previous 12 months
    (the window ends the month before t, so only past and current data are used).
    """
    ma3 = unrate.rolling(3).mean()
    prior_min = ma3.shift(1).rolling(12).min()
    return ma3 - prior_min


def build_panel(refresh: bool = False) -> pd.DataFrame:
    """Monthly panel: features observable at the end of month t, plus USREC."""
    gs10 = load_series("GS10", refresh)
    tb3 = load_series("TB3MS", refresh)
    baa = load_series("BAA", refresh)
    unrate = load_series("UNRATE", refresh)
    usrec = load_series("USREC", refresh)

    # UNRATE for 2025-10 was never published (federal government shutdown). Carry the
    # previous month forward (uses only past data; no look-ahead) and record the fill.
    full_idx = pd.date_range(unrate.index[0], unrate.index[-1], freq="MS")
    missing = full_idx.difference(unrate.index)
    unrate = unrate.reindex(full_idx).ffill(limit=1)
    p = pd.concat([gs10, tb3, baa, unrate, usrec], axis=1)
    p["unrate_filled"] = p.index.isin(missing)
    p = p.loc[p.index >= "1953-04-01"]  # GS10 starts April 1953
    p["term_spread"] = p["GS10"] - p["TB3MS"]
    p["credit_spread"] = p["BAA"] - p["GS10"]
    p["sahm_gap"] = sahm_gap(p["UNRATE"])
    p.index.name = "date"
    return p


def data_checks(panel: pd.DataFrame, refresh: bool = False) -> dict:
    """Cross-check the monthly long-history constructs against FRED's own daily spreads."""
    t10y3m = to_monthly(load_series("T10Y3M", refresh))
    baa10y = to_monthly(load_series("BAA10Y", refresh))
    j1 = pd.concat([panel["term_spread"], t10y3m], axis=1).dropna()
    j2 = pd.concat([panel["credit_spread"], baa10y], axis=1).dropna()
    last_daily = load_series("T10Y3M", refresh)
    return {
        "term_spread_vs_T10Y3M_corr": round(float(j1.corr().iloc[0, 1]), 4),
        "term_spread_vs_T10Y3M_mean_diff_pp": round(float((j1.iloc[:, 0] - j1.iloc[:, 1]).mean()), 3),
        "term_spread_vs_T10Y3M_overlap": f"{j1.index[0]:%Y-%m} to {j1.index[-1]:%Y-%m}",
        "credit_spread_vs_BAA10Y_corr": round(float(j2.corr().iloc[0, 1]), 4),
        "credit_spread_vs_BAA10Y_overlap": f"{j2.index[0]:%Y-%m} to {j2.index[-1]:%Y-%m}",
        "latest_daily_T10Y3M_date": f"{last_daily.index[-1]:%Y-%m-%d}",
        "latest_daily_T10Y3M_value": float(last_daily.iloc[-1]),
        "panel_start": f"{panel.index[0]:%Y-%m}",
        "usrec_last_month": f"{panel['USREC'].dropna().index[-1]:%Y-%m}",
        "features_last_month": f"{panel[['term_spread', 'credit_spread', 'sahm_gap']].dropna().index[-1]:%Y-%m}",
        "unrate_months_forward_filled": [f"{d:%Y-%m}" for d in panel.index[panel["unrate_filled"]]],
        "nan_check":bool(np.isfinite(panel[['term_spread', 'credit_spread']].dropna().values).all()),
    }
