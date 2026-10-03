"""Closer-role proxy from pitch data: games where a pitcher finished a close win for his team.

finish_close = last pitcher for the fielding team in a game it won by 1-3 runs (a save-type finish).
late_finish_close counts the last 45 days of the season, i.e. who held the 9th at season's end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "features" / "closer_proxy.parquet"


def build(first: int = 2015, last: int = 2026) -> pd.DataFrame:
    rows = []
    for y in range(first, last + 1):
        d = pd.read_parquet(RAW / f"{y}.parquet", columns=["game_pk", "game_type", "game_date", "inning_topbot",
                                                          "pitcher", "at_bat_number", "pitch_number",
                                                          "post_home_score", "post_away_score"])
        d = d[d.game_type == "R"].sort_values(["game_pk", "at_bat_number", "pitch_number"])
        d["fld_home"] = d.inning_topbot.eq("Top")
        last_by_side = d.groupby(["game_pk", "fld_home"]).tail(1)
        final = d.groupby("game_pk")[["post_home_score", "post_away_score"]].max()
        last_by_side = last_by_side.join(final, on="game_pk", rsuffix="_f")
        margin = np.where(last_by_side.fld_home, last_by_side.post_home_score_f - last_by_side.post_away_score_f,
                          last_by_side.post_away_score_f - last_by_side.post_home_score_f)
        last_by_side["close_win"] = (margin >= 1) & (margin <= 3)
        last_by_side["game_date"] = pd.to_datetime(last_by_side.game_date)
        cutoff = last_by_side.game_date.max() - pd.Timedelta(days=45)
        g = last_by_side.groupby("pitcher")
        out = pd.DataFrame({"finish_close": g.close_win.sum(),
                            "late_finish_close": last_by_side[last_by_side.game_date >= cutoff]
                            .groupby("pitcher").close_win.sum()}).fillna(0)
        out["season"] = y
        rows.append(out.reset_index().rename(columns={"pitcher": "mlbam_id"}))
    res = pd.concat(rows, ignore_index=True)
    res.to_parquet(OUT, index=False)
    return res


if __name__ == "__main__":
    r = build()
    print(r[r.season == 2026].sort_values("late_finish_close", ascending=False).head(8).to_string(index=False))
