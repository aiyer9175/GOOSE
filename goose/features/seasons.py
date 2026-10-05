"""Season-level hitter and pitcher feature tables (one row per player-season).

Built from the player-game tables plus MLB API season lines, MiLB lines, IL history,
options, draft pedigree and bios. Everything for season t is known at the end of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.features import context as ctx
from goose.data.http import ROOT

PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"

HITTER_COUNTS = [
    "pa", "ab", "k", "bb", "ibb", "hbp", "single", "double", "triple", "hr", "sf", "bbe", "barrels",
    "hard_hit", "sweet_spot", "gb", "fb", "ld", "pu", "pull_air", "ev_sum", "ev_n", "woba_num", "woba_den",
    "xwoba_num", "pitches", "swings", "whiffs", "z_pitches", "o_pitches", "z_swings", "o_swings",
    "z_whiffs", "o_whiffs", "br_swings", "br_whiffs", "br_o_pitches", "br_o_swings", "fb_swings",
    "fb_whiffs", "bs_sum", "bs_n", "fast_swings", "started", "x_1b", "x_2b", "x_3b", "x_hr",
]

# Process metrics (num, den) and their shrinkage constants are in stabilization_hitters.csv.
HITTER_PROCESS = {
    "whiff": ("whiffs", "swings"), "z_con": ("z_con_n", "z_swings"), "chase": ("o_swings", "o_pitches"),
    "z_swing": ("z_swings", "z_pitches"), "barrel": ("barrels", "bbe"), "pull_air": ("pull_air", "bbe"),
    "hard_hit": ("hard_hit", "bbe"), "gb": ("gb", "bbe"), "ld": ("ld", "bbe"), "pu": ("pu", "bbe"),
    "ev": ("ev_sum", "ev_n"), "bat_speed": ("bs_sum", "bs_n"), "fast_swing": ("fast_swings", "bs_n"),
    "br_whiff": ("br_whiffs", "br_swings"), "br_chase": ("br_o_swings", "br_o_pitches"),
    "xwoba": ("xwoba_num", "woba_den"),
}


def _age(birth: pd.Series, season: pd.Series) -> pd.Series:
    """Baseball age: age on July 1 of the season."""
    b = pd.to_datetime(birth, errors="coerce")
    ref = pd.to_datetime(season.astype(str) + "-07-01")
    return (ref - b).dt.days / 365.25


def _team_games() -> pd.DataFrame:
    g = pd.read_parquet(FEATURES / "hitter_games.parquet",
                        columns=["game_pk", "season", "team", "game_date", "opp_sp_hand"])
    return g.drop_duplicates(["game_pk", "team"])


def hitter_seasons() -> pd.DataFrame:
    hg = pd.read_parquet(FEATURES / "hitter_games.parquet")
    hg["z_con_n"] = hg.z_swings - hg.z_whiffs
    tg = _team_games()
    season_end = hg.groupby("season").game_date.max()

    keys = ["mlbam_id", "season"]
    s = hg.groupby(keys)[HITTER_COUNTS + ["z_con_n"]].sum()
    s["games"] = hg.groupby(keys).size()
    s["lineup_slot"] = hg[hg.started == 1].groupby(keys).lineup_slot.mean()

    # Platoon: share of starts that came against LHSP, vs. the team's LHSP share.
    st = hg[hg.started == 1]
    s["starts_vs_l"] = st[st.opp_sp_hand == "L"].groupby(keys).size()
    s["starts_vs_r"] = st[st.opp_sp_hand == "R"].groupby(keys).size()

    # Team at season end and role over the last 45 days.
    last = hg.sort_values("game_date").groupby(keys).tail(1).set_index(keys)
    s["team"] = last.team
    s["last_game"] = last.game_date
    cutoff = hg.season.map(season_end) - pd.Timedelta(days=45)
    late = hg[hg.game_date >= cutoff]
    s["late_starts"] = late.groupby(keys).started.sum()
    s["late_pa"] = late.groupby(keys).pa.sum()
    tg_late = tg[tg.game_date >= tg.season.map(season_end) - pd.Timedelta(days=45)]
    s["late_team_games"] = s.reset_index().set_index(["team", "season"]).index.map(
        tg_late.groupby(["team", "season"]).size()).to_numpy()
    s["team_games"] = s.reset_index().set_index(["team", "season"]).index.map(
        tg.groupby(["team", "season"]).size()).to_numpy()
    lhsp = tg.assign(l=tg.opp_sp_hand.eq("L")).groupby(["team", "season"]).l.mean()
    s["team_lhsp_share"] = s.reset_index().set_index(["team", "season"]).index.map(lhsp).to_numpy()
    s = s.fillna({"starts_vs_l": 0, "starts_vs_r": 0, "late_starts": 0, "late_pa": 0})
    s["start_share"] = s.started / s.team_games
    s["late_start_share"] = s.late_starts / s.late_team_games
    s["on_roster_at_end"] = (s.last_game >= s.index.get_level_values("season").map(season_end)
                             - pd.Timedelta(days=10)).astype(int)
    s["vs_l_share"] = s.starts_vs_l / s.started.clip(lower=1)
    s["platoon_ratio"] = s.vs_l_share / s.team_lhsp_share
    s = s.reset_index()
    s["absent"] = 0
    s = pd.concat([s, absent_rows(s, "hitting")], ignore_index=True)

    # MLB API season line: SB/CS/R/RBI/G and Savant extras.
    api = pd.read_parquet(PROCESSED / "hitter_seasons.parquet")
    api = api.rename(columns={"stolenBases": "sb", "caughtStealing": "cs", "runs": "r", "rbi": "rbi",
                              "gamesPlayed": "g_api", "sacBunts": "sh"})
    keep = ["mlbam_id", "season", "sb", "cs", "r", "rbi", "g_api", "sh", "spd_sprint_speed",
            "sv_fielding_run_value", "sv_n_outs_above_average", "position"]
    s = s.merge(api[[c for c in keep if c in api]], on=keys, how="left")

    # Bios, age, debut, position group.
    bios = pd.read_parquet(PROCESSED / "players.parquet").drop_duplicates("mlbam_id")
    s = s.merge(bios[["mlbam_id", "name", "birth_date", "bats", "primary_position", "mlb_debut"]],
                on="mlbam_id", how="left")
    s["age"] = _age(s.birth_date, s.season)
    s["years_since_debut"] = s.season - pd.to_datetime(s.mlb_debut, errors="coerce").dt.year
    s["pos_group"] = s.primary_position.map(ctx.POS_GROUP).fillna("DH")

    # Career MLB PA before this season and prior-season volume.
    s = s.sort_values(keys)
    s["career_pa_before"] = s.groupby("mlbam_id").pa.cumsum() - s.pa

    # IL, options, pedigree.
    s = s.merge(ctx.il_by_season(), on=keys, how="left")
    s[["il_days", "il_stints", "il_frac"]] = s[["il_days", "il_stints", "il_frac"]].fillna(0)
    s["il_end_spell_days"] = s.get("il_end_spell_days", pd.Series(0, index=s.index)).fillna(0)
    for c in ("il_arm", "il_shoulder", "il_surgery", "il_at_season_end"):
        s[c] = s[c].fillna(False).astype(int)
    s = s.merge(ctx.options_used(), on=keys, how="left")
    s["options_left"] = s.options_left.fillna(3)
    s = s.merge(ctx.draft_pedigree(), on="mlbam_id", how="left")
    s["undrafted"] = s.draft_pick.isna().astype(int)
    s = s.merge(ctx.rehab_by_season(), on=keys, how="left")
    s[["rehab_n", "rehab_late", "rehab_month"]] = s[["rehab_n", "rehab_late", "rehab_month"]].fillna(0)

    s = s.merge(milb_hitting_season(), on=keys, how="left")
    s = s.merge(prospect_threat(s), on=keys, how="left")
    ped_path = PROCESSED / "pedigree.parquet"
    if ped_path.exists():
        s = s.merge(pd.read_parquet(ped_path)[["mlbam_id", "season", "fv_peak", "fv_last", "best_rank",
                                               "last_listed"]], on=keys, how="left")
    aaa_sc = FEATURES / "aaa_statcast_hitters.parquet"
    if aaa_sc.exists():
        s = s.merge(pd.read_parquet(aaa_sc), on=keys, how="left")
    return s


def milb_hitting_season() -> pd.DataFrame:
    """AAA and AA counts per player-season, normalized to their league-season environment.

    Each event count is rescaled by (MLB rate / MiLB league rate) for that season, so a line in
    the hitter-friendly PCL and one in the International League land on the same MLB scale.
    The model still fits a competition-level factor per level on top of this.
    """
    from goose.data import mlbapi
    m = ctx.milb_lines("hitting")
    m = m[m.level.isin(["AAA", "AA"])]
    m = m.rename(columns={"plateAppearances": "pa", "atBats": "ab", "strikeOuts": "k", "baseOnBalls": "bb",
                          "hitByPitch": "hbp", "homeRuns": "hr", "doubles": "double", "triples": "triple",
                          "hits": "h", "sacFlies": "sf", "stolenBases": "sb", "caughtStealing": "cs"})
    cols = ["pa", "ab", "k", "bb", "hbp", "hr", "double", "triple", "h", "sf", "sb", "cs"]
    sport = {"AAA": 11, "AA": 12}
    leagues = {(y, lvl): mlbapi.milb_team_leagues(y, sid) for y in m.season.unique() for lvl, sid in sport.items()}
    m["league_id"] = [leagues[(y, l)].get(t) for y, l, t in zip(m.season, m.level, m.team_id)]
    events = ["k", "bb", "hbp", "hr", "double", "triple", "h"]
    lg = m.groupby(["season", "league_id"])[events + ["pa"]].sum()
    lg_rate = lg[events].div(lg.pa, axis=0)
    api = pd.read_parquet(PROCESSED / "hitter_seasons.parquet")
    mlb = api.rename(columns={"strikeOuts": "k", "baseOnBalls": "bb", "hitByPitch": "hbp", "homeRuns": "hr",
                              "doubles": "double", "triples": "triple", "hits": "h", "plateAppearances": "pa"})
    mlb = mlb.groupby("season")[events + ["pa"]].sum()
    mlb_rate = mlb[events].div(mlb.pa, axis=0)
    ratio = mlb_rate.reindex(m.season).to_numpy() / lg_rate.reindex(
        pd.MultiIndex.from_arrays([m.season, m.league_id])).to_numpy()
    ratio = np.nan_to_num(ratio, nan=1.0)
    m[events] = m[events].to_numpy() * ratio
    g = m.groupby(["mlbam_id", "season", "level"])[cols].sum().unstack("level")
    g.columns = [f"milb_{lvl.lower()}_{c}" for c, lvl in g.columns]
    age = m.groupby(["mlbam_id", "season"]).age.min().rename("milb_age")
    lvl_age = m.groupby(["season", "level"]).age.mean()
    rel = m.assign(age_rel=m.age - m.set_index(["season", "level"]).index.map(lvl_age).to_numpy())
    rel = rel.sort_values("pa").groupby(["mlbam_id", "season"]).age_rel.last().rename("milb_age_rel")
    return g.join(age).join(rel).fillna({c: 0 for c in g.columns}).reset_index()


def prospect_threat(s: pd.DataFrame) -> pd.DataFrame:
    """Best proxy-prospect score in the player's org at his position group (AA/AAA), excluding himself."""
    ps = ctx.prospect_scores()
    bios = pd.read_parquet(PROCESSED / "players.parquet").drop_duplicates("mlbam_id")
    ps = ps.merge(bios[["mlbam_id", "primary_position"]], on="mlbam_id", how="left")
    ps["pos_group"] = ps.primary_position.map(ctx.POS_GROUP)
    teams = pd.read_parquet(PROCESSED / "mlb_teams.parquet")
    org_by_abbrev = teams.set_index("abbrev").team_id
    s = s[["mlbam_id", "season", "team", "pos_group"]].copy()
    s["org_id"] = s.team.map(org_by_abbrev)
    # Top two prospects per org/position/season; take the best one that isn't the player.
    top = ps.sort_values("score", ascending=False).groupby(["parent_org_id", "pos_group", "season"]).head(2)
    m = s.merge(top[["mlbam_id", "parent_org_id", "pos_group", "season", "score"]].rename(
        columns={"mlbam_id": "prospect_id", "parent_org_id": "org_id"}),
        on=["org_id", "pos_group", "season"], how="left")
    m = m[m.prospect_id != m.mlbam_id]
    out = m.groupby(["mlbam_id", "season"]).score.max().rename("prospect_threat")
    return out.reset_index()


def shrink_process(df: pd.DataFrame, prefix: str, stab: pd.DataFrame, league: pd.DataFrame,
                   metrics: dict[str, tuple[str, str]], weights=(1.0, 0.6, 0.35)) -> pd.DataFrame:
    """Empirical-Bayes process metrics from the last three seasons toward the league mean.

    df must have lag columns {col}_l0, {col}_l1, {col}_l2 for every num/den column.
    """
    out = pd.DataFrame(index=df.index)
    for m, (n, d) in metrics.items():
        num = sum(w * df[f"{n}_l{i}"].fillna(0) for i, w in enumerate(weights))
        den = sum(w * df[f"{d}_l{i}"].fillna(0) for i, w in enumerate(weights))
        k = stab.loc[m, "k"]
        mu = league.loc[m]
        out[f"{prefix}{m}"] = (num + k * mu) / (den + k)
        out[f"{prefix}{m}_n"] = den
    return out


PITCHER_COUNTS = [
    "bf", "outs", "k", "bb", "ibb", "hbp", "single", "double", "triple", "hr", "sf", "ab", "bbe", "barrels",
    "hard_hit", "gb", "fb", "ld", "pu", "ev_sum", "ev_n", "woba_num", "woba_den", "xwoba_num", "pitches",
    "swings", "whiffs", "called_strikes", "z_pitches", "o_pitches", "z_swings", "o_swings", "z_whiffs",
    "fbv_sum", "fbv_n", "fb_pitches", "br_pitches", "os_pitches", "br_whiffs", "br_swings", "os_whiffs",
    "os_swings", "fb_whiffs", "fb_swings", "is_sp", "x_1b", "x_2b", "x_3b", "x_hr",
]


def pitcher_seasons() -> pd.DataFrame:
    pg = pd.read_parquet(FEATURES / "pitcher_games.parquet")
    # A start counts as a real start only if it went 3+ IP or 50+ pitches; openers' short starts
    # (by design) are relief work. Official starts are kept as gs_official.
    pg["gs_official"] = pg.is_sp
    pg["opener_gs"] = ((pg.is_sp == 1) & (pg.outs <= 6) & (pg.pitches <= 40)).astype(int)
    pg["is_sp"] = ((pg.is_sp == 1) & ((pg.outs >= 9) | (pg.pitches >= 50))).astype("int8")
    keys = ["mlbam_id", "season"]
    s = pg.groupby(keys)[PITCHER_COUNTS + ["gs_official", "opener_gs"]].sum()
    s["games"] = pg.groupby(keys).size()
    sp = pg[pg.is_sp == 1]
    s["sp_outs"] = sp.groupby(keys).outs.sum()
    s["sp_bf"] = sp.groupby(keys).bf.sum()
    s["sp_pitches"] = sp.groupby(keys).pitches.sum()
    last = pg.sort_values("game_date").groupby(keys).tail(1).set_index(keys)
    s["team"] = last.team
    season_end = pg.groupby("season").game_date.max()
    late = pg[pg.game_date >= pg.season.map(season_end) - pd.Timedelta(days=45)]
    s["late_gs"] = late.groupby(keys).is_sp.sum()
    s["late_outs"] = late.groupby(keys).outs.sum()
    s["on_roster_at_end"] = (s.index.get_level_values("season").map(season_end) - last.game_date
                             <= pd.Timedelta(days=10)).astype(int)
    s = s.fillna({"sp_outs": 0, "sp_bf": 0, "sp_pitches": 0, "late_gs": 0, "late_outs": 0}).reset_index()
    s["absent"] = 0
    s = pd.concat([s, absent_rows(s, "pitching")], ignore_index=True)
    s["ip"] = s.outs / 3
    s["gs"] = s.is_sp
    s["sp_share"] = (s.gs / s.games.replace(0, np.nan)).fillna(0)
    s["ip_per_gs"] = (s.sp_outs / 3) / s.gs.replace(0, np.nan)
    s["pitches_per_gs"] = s.sp_pitches / s.gs.replace(0, np.nan)

    api = pd.read_parquet(PROCESSED / "pitcher_seasons.parquet")
    api = api.rename(columns={"wins": "w", "losses": "l", "saves": "sv", "holds": "hld", "earnedRuns": "er",
                              "runs": "r_allowed", "sv_p_quality_start": "qs", "blownSaves": "bs",
                              "gamesFinished": "gf"})
    keep = ["mlbam_id", "season", "w", "l", "sv", "hld", "er", "r_allowed", "qs", "bs", "gf",
            "sv_arm_angle", "sv_pitch_hand"]
    s = s.merge(api[[c for c in keep if c in api]], on=keys, how="left")

    bios = pd.read_parquet(PROCESSED / "players.parquet").drop_duplicates("mlbam_id")
    s = s.merge(bios[["mlbam_id", "name", "birth_date", "throws", "primary_position", "mlb_debut"]],
                on="mlbam_id", how="left")
    s["age"] = _age(s.birth_date, s.season)
    s["years_since_debut"] = s.season - pd.to_datetime(s.mlb_debut, errors="coerce").dt.year
    s = s.sort_values(keys)
    s["career_bf_before"] = s.groupby("mlbam_id").bf.cumsum() - s.bf
    s["career_gs_before"] = s.groupby("mlbam_id").gs.cumsum() - s.gs

    s = s.merge(ctx.il_by_season(), on=keys, how="left")
    s[["il_days", "il_stints", "il_frac"]] = s[["il_days", "il_stints", "il_frac"]].fillna(0)
    s["il_end_spell_days"] = s.get("il_end_spell_days", pd.Series(0, index=s.index)).fillna(0)
    for c in ("il_arm", "il_shoulder", "il_surgery", "il_at_season_end"):
        s[c] = s[c].fillna(False).astype(int)
    s = s.merge(ctx.options_used(), on=keys, how="left")
    s["options_left"] = s.options_left.fillna(3)
    s = s.merge(ctx.draft_pedigree(), on="mlbam_id", how="left")
    s["undrafted"] = s.draft_pick.isna().astype(int)
    s = s.merge(ctx.rehab_by_season(), on=keys, how="left")
    s[["rehab_n", "rehab_late", "rehab_month"]] = s[["rehab_n", "rehab_late", "rehab_month"]].fillna(0)
    s = s.merge(milb_pitching_season(), on=keys, how="left")
    s["total_ip"] = s.ip + s.milb_aaa_ip.fillna(0) + s.milb_aa_ip.fillna(0) + s.milb_low_ip.fillna(0)
    return s


def milb_pitching_season() -> pd.DataFrame:
    """AAA/AA pitching counts normalized to league-season environment (like hitters), plus total
    MiLB innings at every level (rehab and development innings count toward workload)."""
    from goose.data import mlbapi
    m = ctx.milb_lines("pitching")
    m = m.rename(columns={"battersFaced": "bf", "strikeOuts": "k", "baseOnBalls": "bb", "hitByPitch": "hbp",
                          "homeRuns": "hr", "hits": "h", "gamesStarted": "gs", "doubles": "double",
                          "triples": "triple", "atBats": "ab", "sacFlies": "sf"})
    low = m[~m.level.isin(["AAA", "AA"])].groupby(["mlbam_id", "season"]).ip.sum().rename("milb_low_ip")
    m = m[m.level.isin(["AAA", "AA"])].copy()
    sport = {"AAA": 11, "AA": 12}
    leagues = {(y, lvl): mlbapi.milb_team_leagues(y, sid) for y in m.season.unique() for lvl, sid in sport.items()}
    m["league_id"] = [leagues[(y, l)].get(t) for y, l, t in zip(m.season, m.level, m.team_id)]
    events = ["k", "bb", "hbp", "hr", "h"]
    lg = m.groupby(["season", "league_id"])[events + ["bf"]].sum()
    lg_rate = lg[events].div(lg.bf, axis=0)
    api = pd.read_parquet(PROCESSED / "pitcher_seasons.parquet").rename(
        columns={"strikeOuts": "k", "baseOnBalls": "bb", "hitByPitch": "hbp", "homeRuns": "hr", "hits": "h",
                 "battersFaced": "bf"})
    mlb = api.groupby("season")[events + ["bf"]].sum()
    mlb_rate = mlb[events].div(mlb.bf, axis=0)
    ratio = mlb_rate.reindex(m.season).to_numpy() / lg_rate.reindex(
        pd.MultiIndex.from_arrays([m.season, m.league_id])).to_numpy()
    m[events] = m[events].to_numpy() * np.nan_to_num(ratio, nan=1.0)
    cols = ["bf", "ip", "gs", "k", "bb", "hbp", "hr", "h", "ab", "sf"]
    g = m.groupby(["mlbam_id", "season", "level"])[cols].sum().unstack("level")
    g.columns = [f"milb_{lvl.lower()}_{c}" for c, lvl in g.columns]
    g = g.join(low, how="outer")
    return g.fillna(0).reset_index()


def absent_rows(s: pd.DataFrame, group: str) -> pd.DataFrame:
    """Player-seasons with no MLB games that still need a projection for the next season:
    - recent MLB regulars who missed the season injured (100+ IL days), and
    - upper-minors players (AA/AAA: 150+ PA, or 8+ GS / 40+ IP for pitchers).
    MLB counts are zero; team is the last MLB team, else the MiLB parent org.
    """
    vol = "pa" if group == "hitting" else "bf"
    have = set(zip(s.mlbam_id, s.season))
    mlb_vol = s.set_index(["mlbam_id", "season"])[vol]
    m = ctx.milb_lines(group)
    m = m[m.level.isin(["AAA", "AA"])]
    if group == "hitting":
        mv = m.groupby(["mlbam_id", "season"]).plateAppearances.sum()
        prospects = mv[mv >= 150]
        regular = 150
    else:
        gs = m.groupby(["mlbam_id", "season"]).gamesStarted.sum()
        ip = m.groupby(["mlbam_id", "season"]).ip.sum()
        prospects = gs[(gs >= 8) | (ip >= 40)]
        regular = 160  # batters faced, ~40 IP
    il = ctx.il_by_season()
    hurt = il[il.il_days >= 100][["mlbam_id", "season"]]
    prior = [mlb_vol.reindex(pd.MultiIndex.from_arrays([hurt.mlbam_id, hurt.season - lag])).fillna(0).to_numpy()
             for lag in (1, 2)]
    hurt = hurt[(prior[0] >= regular) | (prior[1] >= regular)]
    cand = pd.concat([prospects.reset_index()[["mlbam_id", "season"]], hurt], ignore_index=True).drop_duplicates()
    cand = cand[[k not in have for k in zip(cand.mlbam_id, cand.season)]]
    cand = cand[cand.season.between(2015, 2026)]

    # Team: most recent MLB team before the season, else the MiLB parent org.
    last_team = s.sort_values("season").groupby("mlbam_id")[["season", "team"]].apply(
        lambda d: dict(zip(d.season, d.team)))
    teams = pd.read_parquet(PROCESSED / "mlb_teams.parquet").set_index("team_id").abbrev
    org = m.sort_values("plateAppearances" if group == "hitting" else "ip").groupby(
        ["mlbam_id", "season"]).parent_org_id.last()
    team = []
    for pid, yr in zip(cand.mlbam_id, cand.season):
        hist = last_team.get(pid, {})
        prev = [hist[y] for y in sorted(hist) if y < yr]
        team.append(prev[-1] if prev else teams.get(org.get((pid, yr))))
    out = cand.assign(team=team, absent=1)
    num = [c for c in s.columns if c not in ("mlbam_id", "season", "team", "absent", "last_game")
           and pd.api.types.is_numeric_dtype(s[c])]
    for c in num:
        out[c] = 0.0
    return out
