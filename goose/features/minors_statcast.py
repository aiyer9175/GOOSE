"""AAA Statcast (2025+) aggregated to hitter-season process counts, same definitions as MLB.

Feeds the process priors (whiff, chase, contact quality) for players whose MLB samples are thin.
No bat-tracking at AAA. Output: data/features/aaa_statcast_hitters.parquet with columns
prefixed 'aaasc_'.
"""
from __future__ import annotations

import pandas as pd

from goose.data.http import ROOT
from goose.features.player_game import _pa_flags, _pitch_flags

RAW = ROOT / "data" / "raw" / "statcast_minors"
OUT = ROOT / "data" / "features" / "aaa_statcast_hitters.parquet"
COLS = ["whiffs", "swings", "z_whiffs", "z_swings", "o_swings", "o_pitches", "z_pitches", "br_whiffs",
        "br_swings", "br_o_swings", "br_o_pitches", "pitches"]
PA_COLS = ["bbe", "barrels", "hard_hit", "gb", "ld", "pu", "pull_air", "ev_sum", "ev_n", "pa"]


def build() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("AAA_*.parquet")):
        year = int(path.stem.split("_")[1])
        d = pd.read_parquet(path)
        d = d[d.game_type == "R"].reset_index(drop=True)
        pf = _pitch_flags(d)
        pitch = pd.concat([d[["batter"]], pf[COLS]], axis=1).groupby("batter").sum()
        pa = d[d.events.notna()]
        paf = _pa_flags(pa)
        pas = pd.concat([pa[["batter"]], paf[PA_COLS]], axis=1).groupby("batter").sum()
        s = pitch.join(pas, how="outer").fillna(0)
        s["z_con_n"] = s.z_swings - s.z_whiffs
        s["season"] = year
        frames.append(s.reset_index().rename(columns={"batter": "mlbam_id"}))
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={c: f"aaasc_{c}" for c in out.columns if c not in ("mlbam_id", "season")})
    out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    o = build()
    print(o.groupby("season").size().to_dict(), "hitter-seasons")
