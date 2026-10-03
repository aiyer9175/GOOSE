"""Expected batted-ball outcomes from exit velocity, launch angle and spray direction.

A league lookup table (pooled 2015-2026) gives P(1B, 2B, 3B, HR | EV bin, LA bin, spray bin)
for each batted ball; per season the expected counts are rescaled so league totals match
actual totals (handles ball/era changes). Unlike Savant's xwOBA, spray direction is included,
so pulled fly balls get credit for the power they actually produce.

Adds x_1b, x_2b, x_3b, x_hr columns to the hitter and pitcher game tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
FEATURES = ROOT / "data" / "features"
OUTCOMES = {"single": "x_1b", "double": "x_2b", "triple": "x_3b", "home_run": "x_hr"}
COLS = ["game_pk", "game_type", "batter", "pitcher", "type", "events", "launch_speed", "launch_angle",
        "hc_x", "hc_y", "stand"]


def _bbe(year: int) -> pd.DataFrame:
    d = pd.read_parquet(RAW / f"{year}.parquet", columns=COLS)
    d = d[(d.game_type == "R") & (d.type == "X") & d.launch_speed.notna() & d.launch_angle.notna()].copy()
    spray = np.degrees(np.arctan2(d.hc_x - 125.42, 198.27 - d.hc_y))
    # Mirror lefties so negative = pull side for everyone.
    d["pull_angle"] = np.where(d.stand == "L", -spray, spray)
    d["ev_bin"] = (d.launch_speed.clip(40, 121) // 3).astype(int)
    d["la_bin"] = (d.launch_angle.clip(-60, 72) // 3).astype(int)
    d["sp_bin"] = (d.pull_angle.clip(-50, 49.9) // 10).fillna(99).astype(int)
    d["season"] = year
    return d


def build(first: int = 2015, last: int = 2026) -> None:
    bbe = pd.concat([_bbe(y) for y in range(first, last + 1)], ignore_index=True)
    for ev, col in OUTCOMES.items():
        bbe[col] = (bbe.events == ev).astype(float)
    ycols = list(OUTCOMES.values())

    # Full table, with an EV/LA-only fallback for sparse cells or missing spray.
    full = bbe.groupby(["ev_bin", "la_bin", "sp_bin"])[ycols].agg(["mean", "size"])
    coarse = bbe.groupby(["ev_bin", "la_bin"])[ycols].mean()
    keys = pd.MultiIndex.from_frame(bbe[["ev_bin", "la_bin", "sp_bin"]])
    n = full[(ycols[0], "size")].reindex(keys).to_numpy()
    probs = full.xs("mean", axis=1, level=1).reindex(keys).to_numpy()
    fallback = coarse.reindex(pd.MultiIndex.from_frame(bbe[["ev_bin", "la_bin"]])).to_numpy()
    # Shrink sparse cells toward the EV/LA-only estimate (pseudo-count 30).
    wt = (n / (n + 30.0))[:, None]
    p = np.where(np.isnan(probs), fallback, wt * probs + (1 - wt) * fallback)
    p = np.where(bbe.sp_bin.to_numpy()[:, None] == 99, fallback, p)
    bbe[ycols] = np.nan_to_num(p)

    # Per-season calibration: league expected totals = league actual totals.
    actual = bbe.groupby("season")[[f"a_{c}" for c in ycols]].sum() if False else \
        bbe.assign(**{f"a_{c}": (bbe.events == ev).astype(float) for ev, c in OUTCOMES.items()}) \
           .groupby("season")[[f"a_{c}" for c in ycols]].sum()
    expected = bbe.groupby("season")[ycols].sum()
    scale = actual.to_numpy() / expected.to_numpy()
    scale = pd.DataFrame(scale, index=expected.index, columns=ycols)
    bbe[ycols] = bbe[ycols].to_numpy() * scale.reindex(bbe.season).to_numpy()

    for role, idcol in (("hitter", "batter"), ("pitcher", "pitcher")):
        agg = bbe.groupby(["game_pk", idcol])[ycols].sum().reset_index().rename(columns={idcol: "mlbam_id"})
        path = FEATURES / f"{role}_games.parquet"
        g = pd.read_parquet(path).drop(columns=ycols, errors="ignore")
        g = g.merge(agg, on=["game_pk", "mlbam_id"], how="left")
        g[ycols] = g[ycols].fillna(0.0)
        g.to_parquet(path, index=False)
        print(f"{role}_games: added {ycols}", flush=True)


if __name__ == "__main__":
    build()
