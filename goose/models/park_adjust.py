"""Apply player-specific park fit to projected component rates.

Projected rates come from a history compiled partly in past home parks; this removes those parks'
effect on the player and applies his next home park (half his games):

    adj_o = g(fit_o in next park) / g(fit_o in past home parks),   g(f) = 1 + 0.5 * (f - 1)

for HR, extra-base hits and singles; BABIP and the XBH share are rebuilt from those.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.features.park_fit import fits


def _g(f):
    return 1 + 0.5 * (f - 1)


def adjust(rates: pd.DataFrame, ids: pd.Series, season: int, team_now: pd.Series, team_prev: pd.Series,
           team_next: pd.Series) -> pd.DataFrame:
    """rates: columns hr, babip, xbh (component rates). Returns adjusted copy + fit columns."""
    pl = pd.DataFrame({"mlbam_id": ids.to_numpy()}, index=rates.index)
    f_now = fits(pl, season, team_now)
    f_prev = fits(pl, season - 1, team_prev.where(team_prev.notna(), team_now))
    f_next = fits(pl, season, team_next)
    out = rates.copy()
    xbh = rates.xbh.clip(0.05, 0.6)
    for frame in (f_now, f_prev, f_next):
        frame["fit_h"] = (1 - xbh) * frame.fit_single + xbh * frame.fit_xbh
    past = {o: (f_now[f"fit_{o}"] + 0.6 * f_prev[f"fit_{o}"]) / 1.6 for o in ("hr", "xbh", "h")}
    adj = {o: _g(f_next[f"fit_{o}"]) / _g(past[o]) for o in ("hr", "xbh", "h")}
    out["hr"] = rates.hr * adj["hr"]
    out["babip"] = (rates.babip * adj["h"]).clip(0.2, 0.42)
    out["xbh"] = (rates.xbh * adj["xbh"] / adj["h"]).clip(0.05, 0.6)
    out["park_fit_hr"] = f_next.fit_hr.to_numpy()
    out["park_fit_xbh"] = f_next.fit_xbh.to_numpy()
    return out
