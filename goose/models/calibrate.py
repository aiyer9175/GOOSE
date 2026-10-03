"""Second-stage calibration of component rates.

For each component and experience segment, fit how much a projection's deviation from the
season's center actually carried into the next season (slope b):

    calibrated = center + b * (projection - center)

b < 1 means the first stage under-shrank that segment. Fit on out-of-sample backtest
projections only; weights are next-season denominators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = {"low": (0, 1000), "mid": (1000, 3000), "high": (3000, 1e9)}
CALIBRATED = ["k", "bb", "hbp", "hr", "babip", "xbh", "tri", "sb_att", "sb_succ"]


def segment(career: pd.Series) -> pd.Series:
    out = pd.Series("high", index=career.index)
    for name, (lo, hi) in SEGMENTS.items():
        out[(career > lo) & (career <= hi)] = name
    out[career <= 0] = "low"
    return out


def _center(df: pd.DataFrame, c: str) -> pd.Series:
    w = df[f"r_{c}_n"].clip(lower=1)
    m = df.assign(_w=w, _pw=df[f"r_{c}"] * w).groupby("season_prev")[["_pw", "_w"]].sum()
    return df.season_prev.map(m._pw / m._w)


def fit(oos: pd.DataFrame, actual: dict[str, tuple[pd.Series, pd.Series]]) -> dict:
    """oos: projection rows with r_{c}, season_prev, career_prev, mlbam_id.
    actual: c -> (num, den) Series indexed by (mlbam_id, season) for the target seasons."""
    seg = segment(oos.career_prev)
    slopes = {}
    for c in CALIBRATED:
        num, den = actual[c]
        key = pd.MultiIndex.from_arrays([oos.mlbam_id, oos.season_prev + 1])
        y = (num / den).reindex(key).to_numpy()
        w = den.reindex(key).to_numpy()
        p = oos[f"r_{c}"].to_numpy()
        ctr = _center(oos, c).to_numpy()
        for sname in SEGMENTS:
            sel = (seg == sname).to_numpy() & np.isfinite(y) & (np.nan_to_num(w) >= 20)
            if sel.sum() < 50:
                slopes[(c, sname)] = 1.0
                continue
            dx, dy, ww = p[sel] - ctr[sel], y[sel] - ctr[sel], w[sel]
            b = np.sum(ww * dx * dy) / np.sum(ww * dx ** 2)
            slopes[(c, sname)] = float(np.clip(b, 0.3, 1.2))
    return slopes


def apply(df: pd.DataFrame, slopes: dict, segments: tuple[str, ...] = ("low",)) -> pd.DataFrame:
    """Calibrate only the given segments (backtests: helps low-sample players, slightly hurts others)."""
    out = df.copy()
    seg = segment(df.career_prev)
    for c in CALIBRATED:
        ctr = _center(df, c)
        b = seg.map(lambda s: slopes.get((c, s), 1.0) if s in segments else 1.0)
        out[f"r_{c}"] = (ctr + b * (df[f"r_{c}"] - ctr)).clip(1e-4, 1 - 1e-4)
    return out


def hitter_actuals(seasons: pd.DataFrame) -> dict:
    s = seasons.set_index(["mlbam_id", "season"])
    h = s.single + s.double + s.triple + s.hr
    contact = s.ab - s.k + s.sf
    xbh = s.double + s.triple
    sb_att = s.sb.fillna(0) + s.cs.fillna(0)
    return {"k": (s.k, s.pa), "bb": (s.bb, s.pa), "hbp": (s.hbp, s.pa), "hr": (s.hr, contact),
            "babip": (h - s.hr, contact - s.hr), "xbh": (xbh, h - s.hr), "tri": (s.triple, xbh),
            "sb_att": (sb_att, h - xbh - s.hr + s.bb + s.hbp), "sb_succ": (s.sb.fillna(0), sb_att)}
