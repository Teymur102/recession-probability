"""Pseudo-out-of-sample backtest: expanding window, annual refit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import model_specs
from .splits import expanding_window_splits
from .target import make_target

FEATURES = ["term_spread", "credit_spread", "sahm_gap"]


def modelling_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows with all features observed. USREC may be missing at the end (not yet published)."""
    return panel.dropna(subset=FEATURES).copy()


def run_backtest(
    panel: pd.DataFrame,
    oos_start: str,
    horizon: int = 12,
    label_lag: int = 0,
    specs: dict | None = None,
) -> pd.DataFrame:
    """Return one row per out-of-sample month with each model's probability.

    For every refit origin the training labels are rebuilt from USREC truncated at the
    forecaster's information date, so no future recession information can enter the fit.
    """
    specs = specs or model_specs()
    df = modelling_frame(panel)
    usrec = panel["USREC"].dropna()
    rows = []
    for sp in expanding_window_splits(df.index, oos_start, horizon=horizon, label_lag=label_lag):
        known_usrec = usrec.loc[: sp.info_end]
        y_known = make_target(known_usrec, horizon)
        in_rec_known = known_usrec.reindex(sp.train_idx)
        train = sp.train_idx[(in_rec_known == 0).to_numpy() & y_known.reindex(sp.train_idx).notna().to_numpy()]
        Xtr, ytr = df.loc[train, FEATURES], y_known.loc[train]
        Xte = df.loc[sp.test_idx, FEATURES]
        out = pd.DataFrame(index=sp.test_idx)
        out["origin"] = sp.origin
        out["n_train"] = len(train)
        out["n_train_pos"] = int(ytr.sum())
        for key, make in specs.items():
            out[key] = make().fit(Xtr, ytr).predict(Xte)
        rows.append(out)
    res = pd.concat(rows)
    # Attach realised outcomes (full, final USREC) for scoring only.
    y_final = make_target(usrec, horizon)
    res["in_recession"] = usrec.reindex(res.index)
    res["y"] = y_final.reindex(res.index)
    res[FEATURES] = df.loc[res.index, FEATURES]
    res.index.name = "date"
    return res


def scored(res: pd.DataFrame) -> pd.DataFrame:
    """Months that count for evaluation: in expansion and with a known 12-month outcome."""
    return res[(res["in_recession"] == 0) & res["y"].notna()]
