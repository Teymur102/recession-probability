"""Figures written to figures/."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK = "#1f2933"
MUTED = "#7b8794"
SHADE = "#d9dee4"
C_SPREAD = "#2f6fb0"
C_FULL = "#c0392b"
C_BASE = "#7b8794"
C_GB = "#8e6bbf"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150,
})


def _shade(ax, usrec: pd.Series, start=None, end=None):
    u = usrec.dropna()
    if start is not None:
        u = u[u.index >= start]
    if end is not None:
        u = u[u.index <= end]
    on = None
    for d, v in u.items():
        if v == 1 and on is None:
            on = d
        if v == 0 and on is not None:
            ax.axvspan(on, d, color=SHADE, lw=0, zorder=0)
            on = None
    if on is not None:
        ax.axvspan(on, u.index[-1] + pd.DateOffset(months=1), color=SHADE, lw=0, zorder=0)


def plot_term_spread(panel: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    _shade(ax, panel["USREC"])
    ax.plot(panel.index, panel["term_spread"], color=C_SPREAD, lw=1.1)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("10y minus 3m Treasury, pp")
    ax.set_title("Term spread (GS10 - TB3MS, monthly), NBER recessions shaded")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_oos_probability(res: pd.DataFrame, panel: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    _shade(ax, panel["USREC"], res.index[0], res.index[-1])
    ax.plot(res.index, res["base_rate"], color=C_BASE, lw=1, ls="--", label="Base rate")
    ax.plot(res.index, res["probit_spread"], color=C_SPREAD, lw=1.4, label="Probit: term spread")
    ax.plot(res.index, res["probit_full"], color=C_FULL, lw=1.4, label="Probit: term + credit + Sahm gap")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(recession starts within 12 months)")
    ax.set_title("Out-of-sample probability (expanding window, refit each January); NBER recessions shaded")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(cal: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", label="Perfect calibration")
    styles = {"probit_spread": (C_SPREAD, "Probit: term spread"),
              "probit_full": (C_FULL, "Probit: term + credit + Sahm"),
              "boosting": (C_GB, "Gradient boosting")}
    for m, (c, lab) in styles.items():
        t = cal[cal["model"] == m]
        ax.plot(t["mean_pred"], t["obs_freq"], marker="o", color=c, lw=1.2, label=lab)
        for _, r in t.iterrows():
            ax.annotate(str(int(r["n"])), (r["mean_pred"], r["obs_freq"]), fontsize=6.5,
                        color=c, xytext=(3, -8), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, out-of-sample\n(labels = months in bin)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
