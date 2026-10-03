"""Player-specific outcome spread (wOBA) for hitters.

Backtests showed contact-fragility profiles (bad K-BB, high chase, breaking-ball whiffs) do NOT
predict busts beyond the projection's mean, but DO predict wider outcomes in both directions,
as does limited MLB experience. This models log residual variance from those features.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from goose.data.http import ROOT

FEATS = ["kbb", "chase", "br_whiff", "age", "log_career"]
PATH = ROOT / "data" / "models" / "hitter_volatility.json"


def profile(seasons: pd.DataFrame, ids: pd.Series, season: int, career: pd.Series) -> pd.DataFrame:
    x = seasons[seasons.season == season].set_index("mlbam_id")
    out = pd.DataFrame(index=ids.index)
    out["kbb"] = ids.map((x.k - x.bb) / x.pa)
    out["chase"] = ids.map(x.o_swings / x.o_pitches)
    out["br_whiff"] = ids.map(x.br_whiffs / x.br_swings)
    out["age"] = ids.map(x.age)
    out["log_career"] = np.log1p(career)
    return out


def fit(residuals: pd.DataFrame) -> dict:
    """residuals: FEATS + 'resid' (actual - projected wOBA, season-demeaned)."""
    d = residuals.dropna(subset=FEATS + ["resid"])
    mu, sd = d[FEATS].mean(), d[FEATS].std()
    Z = np.c_[np.ones(len(d)), ((d[FEATS] - mu) / sd).to_numpy()]
    b = np.linalg.lstsq(Z, np.log(d.resid ** 2 + 1e-6), rcond=None)[0]
    scale = float(np.sqrt((d.resid ** 2).mean() / np.exp(Z @ b).mean()))
    model = {"b": b.tolist(), "mu": mu.to_dict(), "sd": sd.to_dict(), "scale": scale}
    PATH.write_text(json.dumps(model, indent=1))
    return model


def predict(prof: pd.DataFrame, model: dict | None = None) -> pd.Series:
    model = model or json.loads(PATH.read_text())
    mu, sd = pd.Series(model["mu"]), pd.Series(model["sd"])
    z = ((prof[FEATS] - mu) / sd).fillna(0.0).to_numpy()
    Z = np.c_[np.ones(len(prof)), z]
    return pd.Series(model["scale"] * np.sqrt(np.exp(Z @ np.array(model["b"]))), index=prof.index)
