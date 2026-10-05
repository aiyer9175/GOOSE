"""Player-specific park fit: how a hitter's own batted-ball mix plays in a given park.

Team-level park factors treat every hitter alike (Fenway "suppresses" HR overall because of lefties
facing deep right field). Here each park gets outcome rates by batter hand x spray zone
(pull/center/oppo) x launch-angle band x exit-velocity band, as multipliers on the league rate
(3-season window, shrunk toward 1). A hitter's fit for outcome o in park P is

    fit_o = sum_c n_c * p_league(o|c) * m_P(o|c) / sum_c n_c * p_league(o|c)

over his batted balls n_c in cells c. Outcomes: HR, XBH (2B+3B), 1B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "features"
OUTCOMES = {"hr": ["home_run"], "xbh": ["double", "triple"], "single": ["single"]}
SHRINK = 150.0  # pseudo batted balls toward the league rate per cell


def _bbe(year: int) -> pd.DataFrame:
    d = pd.read_parquet(RAW / f"{year}.parquet", columns=[
        "game_type", "batter", "stand", "home_team", "away_team", "inning_topbot", "type", "events",
        "launch_speed", "launch_angle", "hc_x", "hc_y"])
    d = d[(d.game_type == "R") & (d.type == "X") & d.launch_speed.notna() & d.launch_angle.notna()].copy()
    spray = np.degrees(np.arctan2(d.hc_x - 125.42, 198.27 - d.hc_y))
    pull = np.where(d.stand == "L", -spray, spray)  # negative = pull side
    d["zone"] = np.select([pull < -15, pull > 15], [0, 2], 1)
    d["la_b"] = np.digitize(d.launch_angle, [10, 25, 50])
    d["ev_b"] = (d.launch_speed >= 95).astype(int)
    d["park"] = d.home_team
    d["bat_team"] = np.where(d.inning_topbot == "Bot", d.home_team, d.away_team)
    for o, evs in OUTCOMES.items():
        d[o] = d.events.isin(evs).astype(float)
    d["season"] = year
    return d[["season", "batter", "stand", "zone", "la_b", "ev_b", "park", "bat_team", *OUTCOMES]]


CELL = ["stand", "zone", "la_b", "ev_b"]


def build(first: int = 2015, last: int = 2026) -> None:
    bbe = pd.concat([_bbe(y) for y in range(first, last + 1)], ignore_index=True)
    # Park multipliers per season t from seasons t-2..t.
    mults = []
    for t in range(first, last + 1):
        w = bbe[bbe.season.between(t - 2, t)]
        lg = w.groupby(CELL)[list(OUTCOMES)].mean()
        pk = w.groupby(["park"] + CELL)[list(OUTCOMES)].agg(["sum", "count"])
        for o in OUTCOMES:
            base = lg[o].reindex(pk.index.droplevel(0)).to_numpy()
            rate = (pk[(o, "sum")].to_numpy() + SHRINK * base) / (pk[(o, "count")].to_numpy() + SHRINK)
            pk[(o, "m")] = rate / np.maximum(base, 1e-6)
        m = pk.xs("m", axis=1, level=1).reset_index().assign(season=t)
        lgr = lg.reset_index().assign(season=t)
        mults.append(m.merge(lgr, on=CELL + ["season"], suffixes=("_m", "_lg")))
    pd.concat(mults, ignore_index=True).to_parquet(OUT / "park_cells.parquet", index=False)
    # Player batted-ball profiles per season (cell counts) and his home-park share.
    prof = bbe.groupby(["batter", "season"] + CELL).size().rename("n").reset_index()
    prof.to_parquet(OUT / "bbe_profiles.parquet", index=False)
    home = bbe.assign(home=(bbe.park == bbe.bat_team)).groupby(["batter", "season", "bat_team"]).home.agg(["sum", "size"])
    home.reset_index().to_parquet(OUT / "bbe_home.parquet", index=False)


def fits(players: pd.DataFrame, season: int, parks: pd.Series) -> pd.DataFrame:
    """fit_o for each player (index aligned with `players`, columns mlbam_id) in park `parks`
    (team abbreviation; NaN = neutral), using his batted balls from season-1..season and park
    multipliers as of `season`."""
    cells = pd.read_parquet(OUT / "park_cells.parquet")
    cells = cells[cells.season == season]
    prof = pd.read_parquet(OUT / "bbe_profiles.parquet")
    prof = prof[prof.season.between(season - 1, season)].assign(
        n=lambda d: d.n * np.where(d.season == season, 1.0, 0.6))
    prof = prof.groupby(["batter"] + CELL).n.sum().reset_index()
    lg = cells.drop_duplicates(CELL).set_index(CELL)[[f"{o}_lg" for o in OUTCOMES]]
    prof = prof.join(lg, on=CELL)
    out = pd.DataFrame(index=players.index, columns=[f"fit_{o}" for o in OUTCOMES], dtype=float)
    mtab = cells.set_index(["park"] + CELL)
    for pid, park, idx in zip(players.mlbam_id, parks, players.index):
        pr = prof[prof.batter == pid]
        if pr.empty or pd.isna(park):
            out.loc[idx] = 1.0
            continue
        key = pd.MultiIndex.from_frame(pr[CELL].assign(park=park)[["park"] + CELL])
        m = mtab.reindex(key)
        for o in OUTCOMES:
            num = (pr.n.to_numpy() * pr[f"{o}_lg"].to_numpy() * m[f"{o}_m"].fillna(1.0).to_numpy()).sum()
            den = (pr.n.to_numpy() * pr[f"{o}_lg"].to_numpy()).sum()
            out.loc[idx, f"fit_{o}"] = num / den if den > 0 else 1.0
    return out


if __name__ == "__main__":
    build()
    print("built park_cells / bbe_profiles / bbe_home")
