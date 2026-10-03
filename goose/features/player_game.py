"""Player-game component tables from pitch-level Statcast.

Everything is stored as additive counts (numerator and denominator), so any window
(season, half, last N days, as-of date) is a groupby-sum. Regular season only.

Outputs data/features/hitter_games.parquet and pitcher_games.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.data.http import ROOT

RAW = ROOT / "data" / "raw" / "statcast"
OUT = ROOT / "data" / "features"

COLS = [
    "game_pk", "game_date", "game_type", "home_team", "away_team", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "outs_when_up", "batter", "pitcher", "stand", "p_throws",
    "events", "description", "type", "zone", "bb_type", "launch_speed", "launch_angle",
    "launch_speed_angle", "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "hc_x", "hc_y", "pitch_type", "release_speed", "bat_speed",
]

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
         "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
STRIKEOUT = {"strikeout", "strikeout_double_play"}
WALK = {"walk", "intent_walk"}
FASTBALL = {"FF", "SI", "FC"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC"}


def load_season(year: int) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{year}.parquet", columns=COLS)
    df = df[df.game_type == "R"].copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df["season"] = year
    bot = df.inning_topbot == "Bot"
    df["bat_team"] = np.where(bot, df.home_team, df.away_team)
    df["fld_team"] = np.where(bot, df.away_team, df.home_team)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"], ignore_index=True)


def _pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.description
    zone = df.zone.between(1, 9)
    ozone = df.zone.between(11, 14)
    swing = d.isin(SWING)
    whiff = d.isin(WHIFF)
    f = pd.DataFrame(index=df.index)
    f["pitches"] = 1
    f["swings"] = swing
    f["whiffs"] = whiff
    f["called_strikes"] = d.eq("called_strike")
    f["z_pitches"] = zone
    f["o_pitches"] = ozone
    f["z_swings"] = zone & swing
    f["o_swings"] = ozone & swing
    f["z_whiffs"] = zone & whiff
    f["o_whiffs"] = ozone & whiff
    for name, group in (("fb", FASTBALL), ("br", BREAKING), ("os", OFFSPEED)):
        g = df.pitch_type.isin(group)
        f[f"{name}_pitches"] = g
        f[f"{name}_swings"] = g & swing
        f[f"{name}_whiffs"] = g & whiff
        f[f"{name}_o_pitches"] = g & ozone
        f[f"{name}_o_swings"] = g & ozone & swing
    return f.astype("int32")


def _pa_flags(pa: pd.DataFrame) -> pd.DataFrame:
    e = pa.events
    bbe = pa.type.eq("X")
    la, ev = pa.launch_angle, pa.launch_speed
    # Spray angle: negative = left field. Pulled = toward the batter's pull side.
    spray = np.degrees(np.arctan2(pa.hc_x - 125.42, 198.27 - pa.hc_y))
    pulled = np.where(pa.stand == "R", spray < -15, spray > 15)
    air = pa.bb_type.isin(["fly_ball", "line_drive"])
    f = pd.DataFrame(index=pa.index)
    f["pa"] = 1
    f["ab"] = ~e.isin(WALK | {"hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
                             "catcher_interf", "sac_bunt_double_play"})
    f["k"] = e.isin(STRIKEOUT)
    f["bb"] = e.isin(WALK)
    f["ibb"] = e.eq("intent_walk")
    f["hbp"] = e.eq("hit_by_pitch")
    f["single"] = e.eq("single")
    f["double"] = e.eq("double")
    f["triple"] = e.eq("triple")
    f["hr"] = e.eq("home_run")
    f["sf"] = e.isin(["sac_fly", "sac_fly_double_play"])
    f["bbe"] = bbe
    f["barrels"] = bbe & pa.launch_speed_angle.eq(6)
    f["hard_hit"] = bbe & (ev >= 95)
    f["sweet_spot"] = bbe & la.between(8, 32)
    f["gb"] = pa.bb_type.eq("ground_ball")
    f["fb"] = pa.bb_type.eq("fly_ball")
    f["ld"] = pa.bb_type.eq("line_drive")
    f["pu"] = pa.bb_type.eq("popup")
    f["pull_air"] = bbe & air & pulled
    f = f.astype("int32")
    f["ev_sum"] = np.where(bbe, ev.fillna(0), 0.0)
    f["ev_n"] = (bbe & ev.notna()).astype("int32")
    f["woba_num"] = pa.woba_value.fillna(0)
    f["woba_den"] = pa.woba_denom.fillna(0)
    # xwOBA numerator: speed/angle estimate on batted balls, actual value otherwise.
    f["xwoba_num"] = np.where(bbe, pa.estimated_woba_using_speedangle.fillna(pa.woba_value), pa.woba_value)
    f["xwoba_num"] = f.xwoba_num.fillna(0) * f.woba_den
    return f


def _outs_per_pa(pa: pd.DataFrame) -> pd.Series:
    """Outs recorded during each PA, from the next PA's outs in the same half-inning."""
    half = [pa.game_pk, pa.inning, pa.inning_topbot]
    nxt = pa.groupby(half, sort=False).outs_when_up.shift(-1)
    last = nxt.isna()
    # Last PA of a half inning ends it (3 outs) unless the game ended on a walk-off.
    end_outs = np.where(pa.events.isin(["home_run", "single", "double", "triple", "walk", "hit_by_pitch"])
                        & (pa.inning >= 9) & (pa.inning_topbot == "Bot"), pa.outs_when_up, 3)
    outs = np.where(last, end_outs, nxt) - pa.outs_when_up
    return pd.Series(np.clip(outs, 0, 3), index=pa.index)


def build_season(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_season(year)
    pf = _pitch_flags(df)
    pitch_level = pd.concat([df[["game_pk", "batter", "pitcher"]], pf], axis=1)
    bat_speed = df[["game_pk", "batter"]].assign(
        bs_sum=df.bat_speed.where(df.bat_speed >= 50).fillna(0),
        bs_n=(df.bat_speed >= 50).astype("int32"),
        fast_swings=(df.bat_speed >= 75).astype("int32"))
    fb_velo = df[["game_pk", "pitcher"]].assign(
        fbv_sum=np.where(df.pitch_type.isin(FASTBALL), df.release_speed.fillna(0), 0.0),
        fbv_n=(df.pitch_type.isin(FASTBALL) & df.release_speed.notna()).astype("int32"))

    pa = df[df.events.notna()].copy()
    pa_f = _pa_flags(pa)
    pa_f["outs"] = _outs_per_pa(pa).astype("int32")

    # Starters / lineup slots: the first nine distinct batters in each team's game.
    first = pa.drop_duplicates(["game_pk", "bat_team", "batter"])[["game_pk", "bat_team", "batter"]]
    first["slot"] = first.groupby(["game_pk", "bat_team"]).cumcount() + 1
    first = first[first.slot <= 9]
    # Opposing starting pitcher (first pitcher each batting team faced) and his hand.
    opp_sp = df.drop_duplicates(["game_pk", "bat_team"])[["game_pk", "bat_team", "pitcher", "p_throws"]]
    opp_sp = opp_sp.rename(columns={"pitcher": "opp_sp", "p_throws": "opp_sp_hand"})
    sp = df.drop_duplicates(["game_pk", "fld_team"])[["game_pk", "pitcher"]].assign(is_sp=1)

    keys = ["game_pk", "game_date", "season"]
    # ---- hitters
    hp = pitch_level.drop(columns="pitcher").groupby(["game_pk", "batter"]).sum()
    hb = bat_speed.groupby(["game_pk", "batter"]).sum()
    hpa = pd.concat([pa[["game_pk", "batter"]], pa_f.drop(columns="outs")], axis=1).groupby(["game_pk", "batter"]).sum()
    meta = df.drop_duplicates(["game_pk", "batter"])[keys + ["batter", "bat_team", "stand"]]
    h = meta.set_index(["game_pk", "batter"]).join([hpa, hp, hb], how="left").reset_index()
    h = h.merge(first.rename(columns={"slot": "lineup_slot"}), on=["game_pk", "bat_team", "batter"], how="left")
    h["started"] = h.lineup_slot.notna().astype("int8")
    h = h.merge(opp_sp, on=["game_pk", "bat_team"], how="left")
    h = h.rename(columns={"batter": "mlbam_id", "bat_team": "team"})

    # ---- pitchers
    pp = pitch_level.drop(columns="batter").groupby(["game_pk", "pitcher"]).sum()
    pv = fb_velo.groupby(["game_pk", "pitcher"]).sum()
    ppa = pd.concat([pa[["game_pk", "pitcher"]], pa_f], axis=1).groupby(["game_pk", "pitcher"]).sum()
    meta = df.drop_duplicates(["game_pk", "pitcher"])[keys + ["pitcher", "fld_team", "p_throws"]]
    p = meta.set_index(["game_pk", "pitcher"]).join([ppa, pp, pv], how="left").reset_index()
    p = p.merge(sp, on=["game_pk", "pitcher"], how="left")
    p["is_sp"] = p.is_sp.fillna(0).astype("int8")
    p = p.rename(columns={"pitcher": "mlbam_id", "fld_team": "team", "pa": "bf"})
    return h, p


def build_all(first: int = 2015, last: int = 2026) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hs, ps = [], []
    for y in range(first, last + 1):
        h, p = build_season(y)
        hs.append(h)
        ps.append(p)
        print(f"{y}: {len(h):,} hitter-games, {len(p):,} pitcher-games", flush=True)
    pd.concat(hs, ignore_index=True).to_parquet(OUT / "hitter_games.parquet", index=False)
    pd.concat(ps, ignore_index=True).to_parquet(OUT / "pitcher_games.parquet", index=False)


if __name__ == "__main__":
    build_all()
