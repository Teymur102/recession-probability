import numpy as np
import pandas as pd
import pytest

from recession.backtest import run_backtest
from recession.models import BaseRate, Probit
from recession.splits import expanding_window_splits

IDX = pd.date_range("1960-01-01", "2010-06-01", freq="MS")


@pytest.mark.parametrize("label_lag", [0, 12])
def test_training_labels_fully_observed_at_origin(label_lag):
    for sp in expanding_window_splits(IDX, "1985-01-01", horizon=12, label_lag=label_lag):
        assert sp.info_end == sp.origin - pd.DateOffset(months=1 + label_lag)
        # every training month's label window t+1..t+12 ends by the information date
        assert sp.train_idx.max() + pd.DateOffset(months=12) <= sp.info_end
        assert sp.train_idx.max() < sp.test_idx.min()


def test_expanding_and_annual():
    splits = expanding_window_splits(IDX, "1985-01-01", horizon=12)
    assert all(s.origin.month == 1 for s in splits)
    sizes = [len(s.train_idx) for s in splits]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert all(s.train_idx[0] == IDX[0] for s in splits)  # window anchored at the start


def test_every_oos_month_predicted_exactly_once():
    splits = expanding_window_splits(IDX, "1985-01-01", horizon=12)
    test = pd.DatetimeIndex(np.concatenate([s.test_idx.values for s in splits]))
    assert test.is_unique
    assert test.equals(IDX[IDX >= "1985-01-01"])
    assert len(splits[-1].test_idx) == 6  # partial final year (Jan-Jun 2010)


def _synthetic_panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-01", "2005-12-01", freq="MS")
    n = len(idx)
    usrec = np.zeros(n)
    for s in rng.choice(np.arange(24, n - 12), 8, replace=False):
        usrec[s:s + 8] = 1
    spread = rng.normal(1, 1, n) - 2 * pd.Series(usrec).shift(-9).fillna(0).to_numpy()
    return pd.DataFrame({"term_spread": spread, "credit_spread": rng.normal(2, 0.5, n),
                         "sahm_gap": rng.normal(0, 0.2, n), "USREC": usrec}, index=idx)


def test_backtest_predictions_ignore_future_recessions():
    """Flip every USREC value after an origin's information date: that origin's predictions must not move."""
    panel = _synthetic_panel()
    specs = {"base_rate": lambda: BaseRate(), "probit": lambda: Probit(["term_spread"], "p")}
    res = run_backtest(panel, "1990-01-01", specs=specs)
    for origin in pd.to_datetime(["1990-01-01", "1995-01-01", "2000-01-01"]):
        info_end = origin - pd.DateOffset(months=1)
        alt = panel.copy()
        alt.loc[alt.index > info_end, "USREC"] = 1 - alt.loc[alt.index > info_end, "USREC"]
        res_alt = run_backtest(alt, "1990-01-01", specs=specs)
        a = res.loc[res["origin"] == origin, ["base_rate", "probit"]]
        b = res_alt.loc[res_alt["origin"] == origin, ["base_rate", "probit"]]
        pd.testing.assert_frame_equal(a, b)


def test_backtest_predictions_ignore_future_features():
    """Changing features after the test year must not change that year's predictions."""
    panel = _synthetic_panel(1)
    specs = {"probit": lambda: Probit(["term_spread"], "p")}
    res = run_backtest(panel, "1990-01-01", specs=specs)
    alt = panel.copy()
    alt.loc["1996-01-01":, "term_spread"] += 5
    res_alt = run_backtest(alt, "1990-01-01", specs=specs)
    pd.testing.assert_series_equal(res.loc[:"1995-12-01", "probit"], res_alt.loc[:"1995-12-01", "probit"])
