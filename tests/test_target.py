import numpy as np
import pandas as pd
import pytest

from recession.target import make_target, recession_starts


def _usrec(values, start="2000-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_hand_example_horizon_3():
    #            0  1  2  3  4  5  6  7  8  9
    u = _usrec([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    y = make_target(u, horizon=3)
    # y_t = max(u[t+1..t+3])
    expected = [0, 1, 1, 1, 1, 0, 0, np.nan, np.nan, np.nan]
    np.testing.assert_array_equal(y.to_numpy(), np.array(expected, dtype=float))


def test_current_month_is_not_used():
    u1 = _usrec([0, 0, 0, 0, 0, 0])
    u2 = _usrec([1, 0, 0, 0, 0, 0])
    assert make_target(u1, 2).iloc[0] == make_target(u2, 2).iloc[0] == 0


def test_last_h_months_are_unknown():
    u = _usrec([0] * 30)
    y = make_target(u, horizon=12)
    assert y.iloc[-12:].isna().all()
    assert y.iloc[:-12].notna().all()


def test_no_dependence_beyond_horizon():
    """Changing USREC after t+h must not change y_t (no leakage from the far future)."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 60).astype(float)
    h = 12
    y0 = make_target(_usrec(base), h)
    for t in range(0, 60 - h - 1):
        alt = base.copy()
        alt[t + h + 1:] = 1 - alt[t + h + 1:]
        assert make_target(_usrec(alt), h).iloc[t] == y0.iloc[t]


def test_truncating_history_only_removes_labels():
    """Labels built from a truncated USREC agree with the full-history labels wherever known."""
    rng = np.random.default_rng(1)
    u = _usrec(rng.integers(0, 2, 80).astype(float))
    full = make_target(u, 12)
    for cut in (20, 40, 79):
        part = make_target(u.iloc[:cut], 12).dropna()
        pd.testing.assert_series_equal(part, full.loc[part.index])


def test_missing_future_month_makes_label_unknown():
    u = _usrec([0, 0, 0, np.nan, 0, 0])
    y = make_target(u, 2)
    assert np.isnan(y.iloc[1]) and np.isnan(y.iloc[2])
    assert y.iloc[0] == 0


def test_recession_starts():
    u = _usrec([1, 1, 0, 0, 1, 1, 0, 1])
    assert recession_starts(u) == [pd.Timestamp("2000-05-01"), pd.Timestamp("2000-08-01")]


def test_bad_horizon():
    with pytest.raises(ValueError):
        make_target(_usrec([0, 1]), 0)
