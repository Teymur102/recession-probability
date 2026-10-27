"""Forecasting models. Each has fit(X, y) and predict(X) -> probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier


class BaseRate:
    """Unconditional benchmark: share of training months followed by a recession."""

    name = "Base rate"

    def __init__(self, features=()):
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.p_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p_)


class Probit:
    """Probit P(y=1|x) = Phi(a + b'x), estimated by maximum likelihood (statsmodels)."""

    def __init__(self, features, name):
        self.features = list(features)
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xc = sm.add_constant(X[self.features], has_constant="add")
        self.res_ = sm.Probit(y.astype(float), Xc).fit(disp=0, maxiter=200)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xc = sm.add_constant(X[self.features], has_constant="add")
        return np.asarray(self.res_.predict(Xc))

    def predict_interval(self, X: pd.DataFrame, n_draws: int = 5000, seed: int = 0, q=(0.05, 0.95)):
        """Parameter-uncertainty band: draw coefficients from their asymptotic normal."""
        from scipy.stats import norm

        rng = np.random.default_rng(seed)
        Xc = sm.add_constant(X[self.features], has_constant="add").to_numpy()
        draws = rng.multivariate_normal(self.res_.params.to_numpy(), self.res_.cov_params().to_numpy(), n_draws)
        p = norm.cdf(Xc @ draws.T)
        return np.quantile(p, q[0], axis=1), np.quantile(p, q[1], axis=1)


class Boosting:
    """Gradient-boosted trees challenger (shallow, heavily regularised)."""

    def __init__(self, features, name="Gradient boosting (3 signals)"):
        self.features = list(features)
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.m_ = GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=20, random_state=0,
        ).fit(X[self.features].to_numpy(), y.astype(int).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.m_.predict_proba(X[self.features].to_numpy())[:, 1]


SPREAD = ["term_spread"]
FULL = ["term_spread", "credit_spread", "sahm_gap"]


def model_specs():
    """Name -> factory. Order is the order used in tables."""
    return {
        "base_rate": lambda: BaseRate(),
        "probit_spread": lambda: Probit(SPREAD, "Probit: term spread"),
        "probit_full": lambda: Probit(FULL, "Probit: term + credit + Sahm"),
        "boosting": lambda: Boosting(FULL),
    }


LABELS = {
    "base_rate": "Base rate (benchmark)",
    "probit_spread": "Probit: term spread",
    "probit_full": "Probit: term + credit + Sahm gap",
    "boosting": "Gradient boosting (challenger)",
}
