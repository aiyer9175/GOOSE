"""Park factors from Statcast: HR and non-HR hits per PA at each park vs. the same team's road games.

Three-season rolling (t-2..t), regressed halfway to neutral. Factor 1.10 = 10% more HR than neutral.
"""
from __future__ import annotations

import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
FEATURES = ROOT / "data" / "features"


def build(first: int = 2015, last: int = 2026) -> pd.DataFrame:
    rows = []
    for y in range(first, last + 1):
        d = pd.read_parquet(RAW / f"{y}.parquet", columns=["game_type", "events", "home_team", "away_team"])
        d = d[(d.game_type == "R") & d.events.notna()]
        d = d.assign(hr=d.events.eq("home_run"), h=d.events.isin(["single", "double", "triple"]), pa=1)
        home = d.groupby("home_team")[["hr", "h", "pa"]].sum()
        away = d.groupby("away_team")[["hr", "h", "pa"]].sum()
        for team in home.index:
            rows.append(dict(season=y, team=team, home_hr=home.loc[team, "hr"], home_h=home.loc[team, "h"],
                             home_pa=home.loc[team, "pa"], road_hr=away.loc[team, "hr"],
                             road_h=away.loc[team, "h"], road_pa=away.loc[team, "pa"]))
    raw = pd.DataFrame(rows).sort_values(["team", "season"])
    roll = raw.groupby("team")[[c for c in raw if c not in ("season", "team")]].transform(
        lambda x: x.rolling(3, min_periods=1).sum())
    raw[["park_hr", "park_h"]] = pd.DataFrame({
        "park_hr": (roll.home_hr / roll.home_pa) / (roll.road_hr / roll.road_pa),
        "park_h": (roll.home_h / roll.home_pa) / (roll.road_h / roll.road_pa)})
    raw[["park_hr", "park_h"]] = 1 + (raw[["park_hr", "park_h"]] - 1) * 0.5
    out = raw[["season", "team", "park_hr", "park_h"]]
    out.to_parquet(FEATURES / "park_factors.parquet", index=False)
    return out


if __name__ == "__main__":
    pf = build()
    print(pf[pf.season == 2026].sort_values("park_hr").to_string())
