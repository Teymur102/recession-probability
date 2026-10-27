# Recession Probability from the Yield Curve

Can the 10y-3m Treasury spread predict a US recession within 12 months, out of sample? It ranks months well (AUC 0.874, 1985-2025) but does not beat the base rate on Brier score because of the 2022-24 false alarm; current reading for 2026-09: **11.8%**.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg) ![pandas](https://img.shields.io/badge/pandas-3.1.0-150458.svg) ![statsmodels](https://img.shields.io/badge/statsmodels-0.15.0-4051b5.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-f7931e.svg) ![tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)

<table>
<tr>
<td width="62%"><img src="figures/oos_probability.png" alt="Out-of-sample recession probability"></td>
<td width="38%"><img src="figures/calibration.png" alt="Calibration"></td>
</tr>
</table>

## Results

Term-spread probit, pseudo-out-of-sample, refit every January ([`results/summary.md`](results/summary.md)):

| Metric | Value |
|---|---|
| AUC, 1985-2025 (453 expansion months, 48 positive) | 0.874 |
| Brier skill vs base rate, 1985-2025 (90% block-bootstrap CI) | -0.014 [-0.78, 0.50] |
| Brier skill vs base rate, scored 1985-2021 only | 0.453 |
| Peak probability during the 2022-24 inversion (no recession dated) | 99.2% in 2023-05 |
| Current P(recession within 12 months), 2026-09 | 11.8% (90% band 9.5%-14.4%) |
| Historical base rate (expansion months) | 15.9% |

## Proof

- **14 tests** (`pytest`): target construction, no leakage past each refit date (changing future `USREC` leaves forecasts unchanged), expanding-window splitter.
- **One command:** `make` rebuilds every table and figure from cached data in about 6 seconds.
- **Data:** FRED [GS10](https://fred.stlouisfed.org/series/GS10), [TB3MS](https://fred.stlouisfed.org/series/TB3MS), [BAA](https://fred.stlouisfed.org/series/BAA), [UNRATE](https://fred.stlouisfed.org/series/UNRATE), [USREC](https://fred.stlouisfed.org/series/USREC), cached in `data/`; `run.py --refresh` re-downloads.
- **Deterministic:** rerunning on cached data reproduces `results/` and `figures/` byte for byte (clean `git status`).
- **Out-of-sample protocol:** expanding window from 1954, annual refits 1985-2026; a training month is used only once its full 12-month outcome window is observed. Robustness run with a further 12-month NBER label lag in `results/metrics_label_lag12.csv`.

## Stack

Python 3.14 · numpy 2.5.4 · pandas 3.1.0 · scipy 1.18.1 · statsmodels 0.15.0 · scikit-learn 1.9.1 · matplotlib 3.11.2 · pytest 9.2.0

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py && .venv/bin/python -m pytest
```

## Structure

```
recession/   data.py, target.py, splits.py, models.py, backtest.py, evaluate.py, plots.py
tests/       target construction, splitter and leakage tests
data/        cached FRED CSVs
results/     metrics, calibration, current reading, summary.md (generated)
figures/     PNG charts (generated)
run.py       reproduces everything
```

---

Full method, limitations and interview-level detail: [docs/DETAILS.md](docs/DETAILS.md) · MIT License
