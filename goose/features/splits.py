"""Hitter platoon splits from pitch-level Statcast: PA, wOBA numerator/denominator vs LHP and RHP.

Output: data/features/hitter_splits.parquet (one row per batter-season-pitcher hand).
"""
from __future__ import annotations

import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "features" / "hitter_splits.parquet"


def build(first: int = 2015, last: int = 2026) -> pd.DataFrame:
    rows = []
    for y in range(first, last + 1):
        d = pd.read_parquet(RAW / f"{y}.parquet", columns=["game_type", "batter", "stand", "p_throws", "events",
                                                          "woba_value", "woba_denom"])
        d = d[(d.game_type == "R") & d.events.notna()]
        g = d.groupby(["batter", "p_throws"]).agg(pa=("events", "size"), woba_num=("woba_value", "sum"),
                                                   woba_den=("woba_denom", "sum"), stand=("stand", "last"))
        rows.append(g.reset_index().assign(season=y))
    out = pd.concat(rows, ignore_index=True).rename(columns={"batter": "mlbam_id"})
    out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    o = build()
    lg = o.groupby(["stand", "p_throws"])[["woba_num", "woba_den"]].sum()
    print((lg.woba_num / lg.woba_den).round(3).unstack())
