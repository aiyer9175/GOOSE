"""Fantasy position eligibility and projected games.

Eligibility inputs are MLB games / games started by fielding position for the last two seasons
(MLB Stats API fielding splits; LF/CF/RF kept separate, OF = their sum). The page applies each
league's own threshold (e.g. Yahoo 5 starts / 10 games, ESPN 20 games, Ottoneu 10 games or 5 starts
in the current or preceding year). Players with no MLB fielding games fall back to their listed
position.

Projected games (Ottoneu's games caps value hitters per game) = projected PA / PA-per-game, where
PA-per-game is the player's recent rate shrunk toward a prior that rises with his projected role.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.data import mlbapi
from goose.data.http import ROOT

FEAT = ROOT / "data" / "features"
POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]


def fielding_games(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Rows: mlbam_id, season, position, g, gs (summed over teams)."""
    f = pd.concat([mlbapi.season_stats(y, "fielding") for y in seasons], ignore_index=True)
    f = f[f.position.isin(POS)]
    return (f.groupby(["mlbam_id", "season", "position"])[["gamesPlayed", "gamesStarted"]].sum()
              .rename(columns={"gamesPlayed": "g", "gamesStarted": "gs"}).reset_index())


def eligibility(ids: pd.Series, season: int) -> dict[int, dict[str, list[int]]]:
    """{mlbam_id: {pos: [g_prev, gs_prev, g_cur, gs_cur]}} for the given hitters."""
    f = fielding_games((season - 1, season))
    f = f[f.mlbam_id.isin(set(ids))]
    of = f[f.position.isin(["LF", "CF", "RF"])].groupby(["mlbam_id", "season"])[["g", "gs"]].sum().reset_index()
    f = pd.concat([f, of.assign(position="OF")], ignore_index=True)
    out: dict[int, dict[str, list[int]]] = {}
    for (pid, pos), d in f.groupby(["mlbam_id", "position"]):
        prev = d[d.season == season - 1]
        cur = d[d.season == season]
        out.setdefault(int(pid), {})[pos] = [int(prev.g.sum()), int(prev.gs.sum()), int(cur.g.sum()), int(cur.gs.sum())]
    return out


def pitcher_eligibility(ids: pd.Series, season: int) -> dict[int, list[int]]:
    """{mlbam_id: [gs_prev, relief_prev, gs_cur, relief_cur]} from official MLB appearances
    (platforms use official starts, so an opener start counts as a start here)."""
    rows = []
    for y in (season - 1, season):
        d = mlbapi.season_stats(y, "pitching")
        d = d.groupby("mlbam_id")[["gamesPitched", "gamesStarted"]].sum().reset_index()
        rows.append(d.assign(season=y, relief=d.gamesPitched - d.gamesStarted))
    f = pd.concat(rows)
    f = f[f.mlbam_id.isin(set(ids))]
    out: dict[int, list[int]] = {}
    for pid, d in f.groupby("mlbam_id"):
        prev, cur = d[d.season == season - 1], d[d.season == season]
        out[int(pid)] = [int(prev.gamesStarted.sum()), int(prev.relief.sum()), int(cur.gamesStarted.sum()), int(cur.relief.sum())]
    return out


def projected_games(h: pd.DataFrame, season: int) -> pd.Series:
    """h: projection frame with mlbam_id, PA."""
    s = pd.read_parquet(FEAT / "hitter_seasons.parquet", columns=["mlbam_id", "season", "pa", "games"])
    s = s[s.season.between(season - 1, season)].assign(w=lambda d: np.where(d.season == season, 1.0, 0.6))
    agg = s.assign(pa_w=s.pa * s.w, g_w=s.games * s.w).groupby("mlbam_id")[["pa_w", "g_w"]].sum()
    pa_w = h.mlbam_id.map(agg.pa_w).fillna(0).to_numpy()
    g_w = h.mlbam_id.map(agg.g_w).fillna(0).to_numpy()
    prior = 2.9 + 1.3 * np.clip(h.PA.to_numpy() / 650, 0, 1)
    k = 60.0
    ppg = (pa_w + k * prior) / (g_w + k)
    return pd.Series(np.minimum(np.round(h.PA.to_numpy() / ppg), 162), index=h.index)
