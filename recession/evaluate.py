"""Scoring: Brier, Brier skill vs base rate, AUC, log loss, calibration, per-recession table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def metrics_table(sc: pd.DataFrame, models: list[str], ref: str = "base_rate") -> pd.DataFrame:
    y = sc["y"].to_numpy()
    b_ref = brier(y, sc[ref])
    rows = []
    for m in models:
        p = sc[m].to_numpy()
        b = brier(y, p)
        auc = roc_auc_score(y, p) if np.unique(p).size > 1 else np.nan
        rows.append({
            "model": m,
            "brier": b,
            "brier_skill_vs_base": 1 - b / b_ref,
            "auc": auc,
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "hit_rate_at_50": float((p[y == 1] >= 0.5).mean()),
            "false_alarm_rate_at_50": float((p[y == 0] >= 0.5).mean()),
        })
    return pd.DataFrame(rows).set_index("model")


def block_bootstrap_bss(sc: pd.DataFrame, model: str, ref: str = "base_rate",
                        block: int = 24, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """90% moving-block bootstrap interval for the Brier skill score.

    Overlapping 12-month targets make neighbouring months strongly dependent, so
    contiguous 24-month blocks are resampled instead of single months.
    """
    rng = np.random.default_rng(seed)
    y = sc["y"].to_numpy()
    e_m = (sc[model].to_numpy() - y) ** 2
    e_r = (sc[ref].to_numpy() - y) ** 2
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[i] = 1 - e_m[idx].mean() / e_r[idx].mean()
    lo, hi = np.quantile(out, [0.05, 0.95])
    return float(lo), float(hi)


def calibration_table(sc: pd.DataFrame, models: list[str],
                      bins=(0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)) -> pd.DataFrame:
    rows = []
    for m in models:
        cut = pd.cut(sc[m], bins=list(bins), include_lowest=True)
        g = sc.groupby(cut, observed=True).agg(mean_pred=(m, "mean"), obs_freq=("y", "mean"), n=("y", "size"))
        for interval, r in g.iterrows():
            rows.append({"model": m, "bin": str(interval), **r.to_dict()})
    t = pd.DataFrame(rows)
    t["n"] = t["n"].astype(int)
    return t


def recession_table(res: pd.DataFrame, starts: list[pd.Timestamp], models: list[str]) -> pd.DataFrame:
    """Probability 6-12 months before each recession start (months start-12 .. start-6)."""
    rows = []
    for s in starts:
        win = pd.date_range(s - pd.DateOffset(months=12), s - pd.DateOffset(months=6), freq="MS")
        if not set(win).issubset(res.index):
            continue
        r = {"recession_start": f"{s:%Y-%m}"}
        for m in models:
            r[f"{m}_mean"] = float(res.loc[win, m].mean())
            r[f"{m}_max"] = float(res.loc[win, m].max())
        r["term_spread_min"] = float(res.loc[win, "term_spread"].min())
        rows.append(r)
    return pd.DataFrame(rows)


def signal_episodes(res: pd.DataFrame, model: str, starts: list[pd.Timestamp],
                    threshold: float = 0.5, horizon: int = 12) -> pd.DataFrame:
    """Contiguous runs of expansion months with probability >= threshold.

    An episode is a 'hit' if a recession started within `horizon` months after any
    month of the run (i.e. at least one month of the run has y = 1), otherwise a
    'false alarm' -- or 'unresolved' if the outcome window is not yet observed.
    """
    r = res[res["in_recession"].fillna(0) == 0]
    flag = r[model] >= threshold
    rows, cur = [], []
    prev = None
    for d, f in flag.items():
        contiguous = prev is not None and d == prev + pd.DateOffset(months=1)
        if f and (cur and contiguous):
            cur.append(d)
        elif f:
            if cur:
                rows.append(cur)
            cur = [d]
        else:
            if cur:
                rows.append(cur)
            cur = []
        prev = d
    if cur:
        rows.append(cur)
    out = []
    for run in rows:
        ys = r.loc[run, "y"]
        if (ys == 1).any():
            verdict = "hit"
            nxt = [s for s in starts if s > run[0]]
            lead = (nxt[0].year - run[0].year) * 12 + nxt[0].month - run[0].month if nxt else np.nan
        elif ys.isna().any():
            verdict, lead = "unresolved", np.nan
        else:
            verdict, lead = "false alarm", np.nan
        out.append({
            "first_month": f"{run[0]:%Y-%m}", "last_month": f"{run[-1]:%Y-%m}", "months": len(run),
            "peak_prob": float(r.loc[run, model].max()), "verdict": verdict,
            "months_from_first_signal_to_recession": lead,
        })
    return pd.DataFrame(out)
