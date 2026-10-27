"""Reproduce every result, table and figure in this repo.

    python run.py            # uses cached CSVs in data/
    python run.py --refresh  # re-download from FRED first
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from recession.backtest import FEATURES, modelling_frame, run_backtest, scored
from recession.data import build_panel, data_checks
from recession.evaluate import (
    block_bootstrap_bss, calibration_table, metrics_table, recession_table, signal_episodes,
)
from recession.models import LABELS, Probit, SPREAD, FULL
from recession.plots import plot_calibration, plot_oos_probability, plot_term_spread
from recession.target import make_target, recession_starts

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
HORIZON = 12
OOS_START = "1985-01-01"
MODELS = ["base_rate", "probit_spread", "probit_full", "boosting"]


def fmt_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Small markdown writer (avoids the optional tabulate dependency)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def headline(met, met_pre, current, rt, inv) -> dict:
    ps, pf = current["probit_spread"], current["probit_full"]
    h = {
        "Current month": current["as_of_month"],
        "Current P(spread probit)": f"{pct(ps['prob'])} (parameter 90% band {pct(ps['param_90_low'])}-{pct(ps['param_90_high'])})",
        "Current P(full probit)": f"{pct(pf['prob'])} (parameter 90% band {pct(pf['param_90_low'])}-{pct(pf['param_90_high'])})",
        "Full-sample base rate": pct(current["base_rate_full_sample"]),
        "Current term spread (pp)": f"{current['features']['term_spread']:.2f}",
        "Current credit spread (pp)": f"{current['features']['credit_spread']:.2f}",
        "Current Sahm gap (pp)": f"{current['features']['sahm_gap']:.2f}",
    }
    for m in ["probit_spread", "probit_full", "boosting"]:
        h[f"OOS 1985-2025 {m}: AUC / BSS / BSS 90% CI"] = (
            f"{met.loc[m, 'auc']:.3f} / {met.loc[m, 'brier_skill_vs_base']:.3f} / "
            f"[{met.loc[m, 'bss_90ci_low']:.2f}, {met.loc[m, 'bss_90ci_high']:.2f}]")
        h[f"OOS scored 1985-2021 {m}: AUC / BSS"] = (
            f"{met_pre.loc[m, 'auc']:.3f} / {met_pre.loc[m, 'brier_skill_vs_base']:.3f}")
    for _, r in rt.iterrows():
        h[f"{r['recession_start']} start: spread probit mean/max P 6-12m before"] = (
            f"{pct(r['probit_spread_mean'])} / {pct(r['probit_spread_max'])}")
    h["2022-24 inversion: spread probit peak P"] = f"{pct(inv['probit_spread_peak_prob'])} in {inv['probit_spread_peak_month']}"
    return h


def main(refresh: bool = False):
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    panel = build_panel(refresh)
    checks = data_checks(panel, refresh)
    starts = recession_starts(panel["USREC"])

    # ---- 1. Pseudo-out-of-sample backtest -------------------------------------------
    res = run_backtest(panel, OOS_START, HORIZON, label_lag=0)
    res.to_csv(RES / "oos_predictions.csv", float_format="%.6f")
    sc = scored(res)
    met = metrics_table(sc, MODELS)
    for m in MODELS[1:]:
        lo, hi = block_bootstrap_bss(sc, m)
        met.loc[m, "bss_90ci_low"], met.loc[m, "bss_90ci_high"] = lo, hi
    met["n_months"] = len(sc)
    met["n_positive"] = int(sc["y"].sum())
    met.loc["base_rate", "auc"] = float("nan")  # a near-constant forecast has no meaningful ranking
    met.to_csv(RES / "metrics.csv", float_format="%.4f")

    # Same out-of-sample predictions, scored only on months before 2022 (no refit):
    # shows how much of the full-period score is the 2022-2024 inversion.
    sc_pre = sc.loc[:"2021-12-01"]
    met_pre = metrics_table(sc_pre, MODELS)
    met_pre.loc["base_rate", "auc"] = float("nan")
    met_pre["n_months"] = len(sc_pre)
    met_pre["n_positive"] = int(sc_pre["y"].sum())
    met_pre.to_csv(RES / "metrics_scored_1985_2021.csv", float_format="%.4f")

    # ---- 2. Robustness: NBER dates known only 12 months later -------------------------
    res_lag = run_backtest(panel, OOS_START, HORIZON, label_lag=12)
    met_lag = metrics_table(scored(res_lag), MODELS)
    met_lag.loc["base_rate", "auc"] = float("nan")
    met_lag.to_csv(RES / "metrics_label_lag12.csv", float_format="%.4f")

    # ---- 3. Calibration, per-recession table, signal episodes -------------------------
    cal = calibration_table(sc, MODELS[1:])
    cal.to_csv(RES / "calibration.csv", index=False, float_format="%.4f")
    oos_starts = [s for s in starts if s >= pd.Timestamp(OOS_START)]
    rt = recession_table(res, oos_starts, MODELS)
    rt.to_csv(RES / "recession_table.csv", index=False, float_format="%.4f")
    eps = pd.concat([signal_episodes(res, m, starts).assign(model=m) for m in ["probit_spread", "probit_full", "boosting"]])
    eps.to_csv(RES / "signal_episodes.csv", index=False, float_format="%.4f")

    # 2022-2024 inversion: what the models said, month by month summary.
    w = res.loc["2022-01-01":"2024-12-01"]
    inv = {
        "months_term_spread_negative_2022_2024": int((w["term_spread"] < 0).sum()),
        "first_negative_month": f"{w.index[(w['term_spread'] < 0).to_numpy()][0]:%Y-%m}",
        "last_negative_month": f"{w.index[(w['term_spread'] < 0).to_numpy()][-1]:%Y-%m}",
        "min_term_spread": float(w["term_spread"].min()),
        "min_term_spread_month": f"{w['term_spread'].idxmin():%Y-%m}",
    }
    for m in ["probit_spread", "probit_full", "boosting"]:
        inv[f"{m}_peak_prob"] = float(w[m].max())
        inv[f"{m}_peak_month"] = f"{w[m].idxmax():%Y-%m}"
        inv[f"{m}_months_at_or_above_50pct"] = int((w[m] >= 0.5).sum())
    inv["labels_in_window"] = {str(k): int(v) for k, v in w["y"].value_counts(dropna=False).items()}
    (RES / "inversion_2022_2024.json").write_text(json.dumps(inv, indent=2))

    # ---- 4. Full-sample champion fit + current reading --------------------------------
    df = modelling_frame(panel)
    usrec = panel["USREC"].dropna()
    y = make_target(usrec, HORIZON)
    train = df.index[(usrec.reindex(df.index) == 0).to_numpy() & y.reindex(df.index).notna().to_numpy()]
    coef_rows = []
    current = {"as_of_month": f"{df.index[-1]:%Y-%m}", "usrec_known_through": f"{usrec.index[-1]:%Y-%m}",
               "last_labelled_month": f"{train[-1]:%Y-%m}", "n_train": len(train),
               "n_train_pos": int(y.loc[train].sum()), "base_rate_full_sample": float(y.loc[train].mean())}
    latest = df.iloc[[-1]]
    current["features"] = {f: float(latest[f].iloc[0]) for f in FEATURES}
    recent = pd.DataFrame(index=df.index[-12:])
    for key, feats in [("probit_spread", SPREAD), ("probit_full", FULL)]:
        mdl = Probit(feats, key).fit(df.loc[train], y.loc[train])
        p = float(mdl.predict(latest)[0])
        lo, hi = mdl.predict_interval(latest)
        current[key] = {"prob": p, "param_90_low": float(lo[0]), "param_90_high": float(hi[0]),
                        "pseudo_r2": float(mdl.res_.prsquared)}
        recent[key] = mdl.predict(df.loc[recent.index])
        r = mdl.res_
        me = r.get_margeff(at="mean").summary_frame()
        for name in r.params.index:
            coef_rows.append({"model": key, "variable": name, "coef": r.params[name], "std_err": r.bse[name],
                              "z": r.tvalues[name], "p_value": r.pvalues[name],
                              "marginal_effect_at_mean": me["dy/dx"].get(name, float("nan"))})
    current["oos_backtest_latest"] = {m: float(res[m].iloc[-1]) for m in MODELS}
    current["oos_backtest_latest_month"] = f"{res.index[-1]:%Y-%m}"
    pd.DataFrame(coef_rows).to_csv(RES / "probit_coefficients_full_sample.csv", index=False, float_format="%.4f")
    recent.index.name = "date"
    recent.to_csv(RES / "recent_12m_full_sample_fit.csv", float_format="%.4f")
    (RES / "current_reading.json").write_text(json.dumps(current, indent=2))
    (RES / "data_checks.json").write_text(json.dumps(checks, indent=2))

    # ---- 5. Figures -----------------------------------------------------------------
    plot_term_spread(panel.loc[panel["term_spread"].notna()], FIG / "term_spread.png")
    plot_oos_probability(res, panel, FIG / "oos_probability.png")
    plot_calibration(cal, FIG / "calibration.png")

    # ---- 6. Human-readable summary ----------------------------------------------------
    m2 = met.copy()
    m2.index = [LABELS[i] for i in m2.index]
    m2.index.name = "model"
    rt2 = rt.set_index("recession_start")
    s = [f"# Results summary (generated by run.py)\n",
         f"Out-of-sample window: {sc.index[0]:%Y-%m} to {sc.index[-1]:%Y-%m}, "
         f"{len(sc)} scored expansion months, {int(sc['y'].sum())} of them followed by a recession within 12 months.\n",
         "## Headline numbers (formatted for the README)\n",
         *[f"- {k}: {v}" for k, v in headline(met, met_pre, current, rt, inv).items()],
         "\n## Metrics\n", fmt_table(m2.drop(columns=["n_months", "n_positive"]), "{:.3f}"),
         f"\n## Same predictions scored on {sc_pre.index[0]:%Y-%m} to {sc_pre.index[-1]:%Y-%m} only "
         f"({len(sc_pre)} months, {int(sc_pre['y'].sum())} positive)\n",
         fmt_table(met_pre.drop(columns=["n_months", "n_positive"]).rename(index=LABELS), "{:.3f}"),
         "\n## Robustness: NBER dates assumed known only 12 months later\n",
         fmt_table(met_lag.rename(index=LABELS), "{:.3f}"),
         "\n## Probability 6-12 months before each recession start\n", fmt_table(rt2, "{:.3f}"),
         "\n## Signal episodes (probability >= 50%)\n", fmt_table(eps.set_index("model"), "{:.3f}"),
         "\n## 2022-2024 inversion\n```json\n" + json.dumps(inv, indent=2) + "\n```",
         "\n## Current reading\n```json\n" + json.dumps(current, indent=2) + "\n```",
         "\n## Data checks\n```json\n" + json.dumps(checks, indent=2) + "\n```\n"]
    (RES / "summary.md").write_text("\n".join(s))
    print("\n".join(s))
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download FRED CSVs into data/")
    main(ap.parse_args().refresh)
