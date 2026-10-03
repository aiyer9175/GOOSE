"""Conditional outcome quantiles from out-of-fold predictions.

Gradient-boosted quantile regression collapses toward the unconditional quantile (often 0 for
playing time), so instead: get out-of-fold mean predictions (folds by season), bin them within
groups (e.g. established regulars vs. everyone else), take the empirical distribution of actual
outcomes per bin, and interpolate linearly between bin centers so ranges vary smoothly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone


class BinnedQuantiles:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), thresholds=(), n_bins: int = 20):
        self.quantiles, self.thresholds, self.n_bins = quantiles, thresholds, n_bins

    def fit(self, model, X: pd.DataFrame, y: pd.Series, folds: pd.Series,
            group: pd.Series | None = None) -> "BinnedQuantiles":
        oof = np.empty(len(y))
        for f in folds.unique():
            te = (folds == f).to_numpy()
            oof[te] = clone(model).fit(X[~te], y[~te]).predict(X[te])
        group = pd.Series(0, index=y.index) if group is None else group
        self.tables = {}
        for g in group.unique():
            sel = (group == g).to_numpy()
            p, yy = oof[sel], y.to_numpy()[sel]
            n_bins = max(4, min(self.n_bins, sel.sum() // 150))
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            df = pd.DataFrame({"b": b, "y": yy, "p": p})
            t = df.groupby("b").y.quantile(list(self.quantiles)).unstack()
            for th in self.thresholds:
                t[f"P_{th}"] = df.groupby("b").y.apply(lambda s: (s >= th).mean())
            t = t.cummax()  # monotone in the predicted mean
            t["center"] = df.groupby("b").p.median()
            self.tables[g] = t.sort_values("center")
        return self

    def fit_oof(self, oof: np.ndarray, y: pd.Series, group: pd.Series | None = None) -> "BinnedQuantiles":
        """Same as fit, from precomputed out-of-fold predictions."""
        group = pd.Series(0, index=y.index) if group is None else group
        self.tables = {}
        for g in group.unique():
            sel = (group == g).to_numpy()
            p, yy = oof[sel], y.to_numpy()[sel]
            n_bins = max(4, min(self.n_bins, sel.sum() // 150))
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            df = pd.DataFrame({"b": b, "y": yy, "p": p})
            t = df.groupby("b").y.quantile(list(self.quantiles)).unstack()
            for th in self.thresholds:
                t[f"P_{th}"] = df.groupby("b").y.apply(lambda s: (s >= th).mean())
            t["center"] = df.groupby("b").p.median()
            t = t.sort_values("center")
            # Smooth monotone in the projection (isotonic, weighted by bin size).
            from sklearn.isotonic import IsotonicRegression
            n = df.groupby("b").size().reindex(t.index)
            for c in [c for c in t.columns if c != "center"]:
                t[c] = IsotonicRegression().fit(t.center, t[c], sample_weight=n).predict(t.center)
            self.tables[g] = t
        return self

    def predict(self, mean_pred: np.ndarray, group: np.ndarray | None = None) -> pd.DataFrame:
        group = np.zeros(len(mean_pred)) if group is None else np.asarray(group)
        cols = [c for c in next(iter(self.tables.values())).columns if c != "center"]
        out = pd.DataFrame(index=range(len(mean_pred)), columns=cols, dtype=float)
        for g, t in self.tables.items():
            sel = group == g
            for c in cols:
                out.loc[sel, c] = np.interp(mean_pred[sel], t.center.to_numpy(), t[c].to_numpy())
        return out


class HealthyBlend:
    """Single playing-time number between the injury-adjusted mean (E) and the
    healthy-and-in-MLB mean (H): PT = E + w * (H - E), w = sigmoid(Z @ b).

    w is fit on out-of-fold predictions by least squares, with Z = talent and track record,
    so better, more established players lean toward their healthy workload (teams play
    them whenever they can), as backtests showed.
    """

    def fit(self, E, H, y, Z):
        from scipy.optimize import minimize
        from scipy.special import expit
        Z = np.column_stack([np.ones(len(E)), Z])

        def loss(b):
            return np.mean((E + expit(Z @ b) * (H - E) - y) ** 2)
        self.b = minimize(loss, np.zeros(Z.shape[1]), method="L-BFGS-B").x
        return self

    def weight(self, Z):
        from scipy.special import expit
        return expit(np.column_stack([np.ones(len(Z)), Z]) @ self.b)
