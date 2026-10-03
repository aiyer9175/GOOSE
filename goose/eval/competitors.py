"""Load FanGraphs-exported competitor projections and score them against actuals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FG = ROOT / "data" / "raw" / "fangraphs"
SYSTEMS = ["steamer", "zips", "thebat", "thebatx", "atc", "oopsy", "fgdc"]
# Systems whose PA is a real playing-time forecast (ZiPS hitters assume full-time).
PT_SYSTEMS = ["steamer", "thebat", "thebatx", "atc", "oopsy", "fgdc"]
PCT_COLS = ["K%", "BB%", "K-BB%", "LOB%", "GB%", "HR/FB"]


def load(kind: str, season: int) -> pd.DataFrame:
    """kind: 'hitters' | 'pitchers'. Long frame: one row per (system, mlbam_id)."""
    frames = []
    for s in SYSTEMS:
        path = FG / str(season) / f"{kind}_{s}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path).dropna(subset=["MLBAMID"])
        d = d.rename(columns={"MLBAMID": "mlbam_id", "SO": "K"})
        d["mlbam_id"] = d.mlbam_id.astype(int)
        d["system"] = s
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    for c in PCT_COLS:
        if c in out and out[c].max() > 1:
            out[c] = out[c] / 100
    if kind == "hitters":
        out["OPS"] = out.OBP + out.SLG
    else:
        out["FIP"] = (13 * out.HR + 3 * (out.BB + out.HBP) - 2 * out.K) / out.IP
    return out


def load_hitters(season: int) -> pd.DataFrame:
    return load("hitters", season)


def _ip(s: pd.Series) -> pd.Series:
    """MLB '123.1' notation (thirds) -> decimal innings."""
    s = pd.to_numeric(s, errors="coerce")
    return np.floor(s) + (s - np.floor(s)) * 10 / 3


def actual_pitchers(season: int) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data" / "processed" / "pitcher_seasons.parquet")
    p = p[p.season == season].rename(columns={
        "battersFaced": "TBF", "strikeOuts": "K", "baseOnBalls": "BB", "hitByPitch": "HBP",
        "homeRuns": "HR", "wins": "W", "saves": "SV", "holds": "HLD", "gamesStarted": "GS",
        "earnedRuns": "ER", "hits": "H", "sv_p_quality_start": "QS",
    })
    p["IP"] = _ip(p.inningsPitched)
    p["ERA"] = 9 * p.ER / p.IP
    p["WHIP"] = (p.BB + p.H) / p.IP
    p["K%"] = p.K / p.TBF
    p["BB%"] = p.BB / p.TBF
    p["K-BB%"] = p["K%"] - p["BB%"]
    p["HR/9"] = 9 * p.HR / p.IP
    p["FIP"] = (13 * p.HR + 3 * (p.BB + p.HBP) - 2 * p.K) / p.IP
    p["QS"] = p.QS.fillna(0)
    return p[["mlbam_id", "name", "TBF", "IP", "GS", "W", "SV", "HLD", "QS", "K", "ERA", "WHIP",
              "K%", "BB%", "K-BB%", "HR/9", "FIP"]]


def prior_career_tbf(season: int) -> pd.Series:
    p = pd.read_parquet(ROOT / "data" / "processed" / "pitcher_seasons.parquet")
    return p[p.season < season].groupby("mlbam_id").battersFaced.sum()


def actual_hitters(season: int) -> pd.DataFrame:
    h = pd.read_parquet(ROOT / "data" / "processed" / "hitter_seasons.parquet")
    h = h[h.season == season].rename(columns={
        "plateAppearances": "PA", "homeRuns": "HR", "runs": "R", "rbi": "RBI", "stolenBases": "SB",
        "strikeOuts": "K", "baseOnBalls": "BB", "hitByPitch": "HBP",
    })
    for c in ("avg", "obp", "slg"):
        h[c.upper()] = pd.to_numeric(h[c], errors="coerce")
    h["wOBA"] = pd.to_numeric(h["sv_woba"], errors="coerce")
    h["OPS"] = h.OBP + h.SLG
    h["ISO"] = h.SLG - h.AVG
    h["K%"] = h.K / h.PA
    h["BB%"] = h.BB / h.PA
    return h[["mlbam_id", "name", "PA", "HR", "R", "RBI", "SB", "AVG", "OBP", "SLG", "OPS", "ISO",
              "wOBA", "K%", "BB%"]]


def prior_career_pa(season: int) -> pd.Series:
    h = pd.read_parquet(ROOT / "data" / "processed" / "hitter_seasons.parquet")
    return h[h.season < season].groupby("mlbam_id").plateAppearances.sum()


def _wrmse(err: pd.Series, w: pd.Series, debias: bool) -> float:
    if debias:
        err = err - np.average(err, weights=w)
    return float(np.sqrt(np.average(err ** 2, weights=w)))


def score_rates(proj: pd.DataFrame, act: pd.DataFrame, stats: list[str], min_pa: int,
                ids: pd.Index | None = None, debias: bool = True, weight: str = "PA") -> pd.DataFrame:
    """Weighted RMSE on players every system projected and who reached min_pa of `weight`.

    debias=True removes each system's mean error first (league-level misses aren't skill).
    """
    common = set.intersection(*(set(g.mlbam_id) for _, g in proj.groupby("system")))
    a = act[(act[weight] >= min_pa) & act.mlbam_id.isin(common)]
    if ids is not None:
        a = a[a.mlbam_id.isin(ids)]
    rows = {}
    for s, g in proj.groupby("system"):
        m = a.merge(g, on="mlbam_id", suffixes=("", "_p"))
        rows[s] = {st: _wrmse(m[f"{st}_p"] - m[st], m[weight], debias) for st in stats}
        rows[s]["n"] = len(m)
    return pd.DataFrame(rows).T


def score_counting(proj: pd.DataFrame, act: pd.DataFrame, stats: list[str], min_proj_pa: float = 100,
                   volume: str = "PA") -> pd.DataFrame:
    """RMSE of counting stats over players each system projected for min_proj_pa+;
    players who never appeared count as 0. Restricted to the common set."""
    sets = [set(g.loc[g[volume] >= min_proj_pa, "mlbam_id"]) for _, g in proj.groupby("system")]
    common = set.intersection(*sets)
    rows = {}
    for s, g in proj.groupby("system"):
        m = g[g.mlbam_id.isin(common)].merge(act, on="mlbam_id", how="left", suffixes=("_p", ""))
        m[stats] = m[stats].fillna(0)
        rows[s] = {st: float(np.sqrt(np.mean((m[f"{st}_p"] - m[st]) ** 2))) if f"{st}_p" in m else np.nan
                   for st in stats}
        rows[s]["n"] = len(m)
    return pd.DataFrame(rows).T
