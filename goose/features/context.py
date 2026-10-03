"""Context features from transactions, rosters, drafts and MiLB stats.

All features for season t use only information available at the end of season t
(before the offseason), so training rows and live projections see the same inputs.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from goose.data.http import ROOT

PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"

IL_PLACE = re.compile(r"placed .* on the (?:\d+)-day (?:injured|disabled) list", re.I)
IL_TRANSFER = re.compile(r"transferred .* to the 60-day (?:injured|disabled) list", re.I)
IL_OFF = re.compile(r"(?:activated|reinstated|returned) .* from the (?:\d+)-day (?:injured|disabled) list", re.I)
RETRO = re.compile(r"retroactive to (\w+ \d+, \d{4})", re.I)
ARM = re.compile(r"tommy john|ucl|elbow|ulnar|flexor", re.I)
SHOULDER = re.compile(r"shoulder|labrum|rotator|capsule", re.I)
SURGERY = re.compile(r"surgery|reconstruction|repair", re.I)

SEASON_START = {y: pd.Timestamp(f"{y}-03-28") for y in range(2000, 2028)}
SEASON_START[2020] = pd.Timestamp("2020-07-23")
SEASON_END = {y: pd.Timestamp(f"{y}-10-01") for y in range(2000, 2028)}
SEASON_END[2020] = pd.Timestamp("2020-09-27")
SEASON_DAYS = {y: (SEASON_END[y] - SEASON_START[y]).days for y in SEASON_START}


def transactions() -> pd.DataFrame:
    t = pd.read_parquet(PROCESSED / "transactions.parquet")
    t["date"] = pd.to_datetime(t.effective_date.fillna(t.date), errors="coerce")
    return t.dropna(subset=["date"]).sort_values(["mlbam_id", "date"], ignore_index=True)


def il_spells(t: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per IL spell: start, end, arm/shoulder/surgery flags."""
    t = transactions() if t is None else t
    desc = t.description.fillna("")
    place = desc.str.contains(IL_PLACE.pattern, regex=True, flags=re.I) | \
        desc.str.contains(IL_TRANSFER.pattern, regex=True, flags=re.I)
    off = desc.str.contains(IL_OFF.pattern, regex=True, flags=re.I)
    # Rehab assignments and number changes happen while on the IL; anything else
    # (option, outright, release, trade, DFA...) means the player left the IL.
    rehab = desc.str.contains("rehab", case=False)
    other = ~(place | off | rehab) & ~t.type_code.isin(["NUM", "SC"])
    ev = t[place | off | other].copy()
    ev["kind"] = np.select([place, off], ["on", "off"], "other")[(place | off | other).to_numpy()]
    retro = ev.description.str.extract(RETRO)[0]
    ev["when"] = pd.to_datetime(retro, format="%B %d, %Y", errors="coerce").fillna(ev.date)
    spells = []
    for pid, g in ev.groupby("mlbam_id", sort=False):
        start = None
        flags = dict(arm=False, shoulder=False, surgery=False)
        for row in g.itertuples(index=False):
            if start is not None and row.when > SEASON_END[start.year]:
                spells.append((pid, start, SEASON_END[start.year], *flags.values()))
                start = None
            if row.kind == "on":
                if start is None:
                    start = row.when
                    flags = dict(arm=False, shoulder=False, surgery=False)
                flags["arm"] |= bool(ARM.search(row.description))
                flags["shoulder"] |= bool(SHOULDER.search(row.description))
                flags["surgery"] |= bool(SURGERY.search(row.description))
            elif start is not None:
                spells.append((pid, start, row.when, *flags.values()))
                start = None
        if start is not None:  # still on the IL: close at that season's end
            spells.append((pid, start, min(SEASON_END[start.year], pd.Timestamp.today()), *flags.values()))
    return pd.DataFrame(spells, columns=["mlbam_id", "start", "end", "arm", "shoulder", "surgery"])


def il_by_season(spells: pd.DataFrame | None = None) -> pd.DataFrame:
    """IL days within each season window, stint counts and injury-type flags."""
    s = il_spells() if spells is None else spells
    rows = []
    for y in range(2012, 2027):
        lo, hi = SEASON_START[y], SEASON_END[y]
        end = s.end.fillna(pd.Timestamp("2099-01-01"))
        sel = (s.start <= hi) & (end >= lo)
        g = s[sel].assign(days=(np.minimum(end[sel], hi) - np.maximum(s.start[sel], lo)).dt.days.clip(lower=0))
        agg = g.groupby("mlbam_id").agg(il_days=("days", "sum"), il_stints=("days", "size"),
                                         il_arm=("arm", "max"), il_shoulder=("shoulder", "max"),
                                         il_surgery=("surgery", "max"))
        # Still on the IL when the season ended -> likely to start next season hurt.
        agg["il_at_season_end"] = g.assign(open=end[sel] >= hi).groupby("mlbam_id").open.max()
        agg["season"] = y
        rows.append(agg.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["il_frac"] = out.il_days / out.season.map(SEASON_DAYS)
    return out


def options_used(t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Option years used through each season (any optional assignment in a season = one year).

    Simplification: ignores the 20-day rule and fourth-option eligibility.
    """
    t = transactions() if t is None else t
    opt = t[t.type_code == "OPT"].assign(season=lambda d: d.date.dt.year)
    per = opt.drop_duplicates(["mlbam_id", "season"])[["mlbam_id", "season"]]
    rows = []
    for y in range(2012, 2027):
        used = per[per.season <= y].groupby("mlbam_id").size().rename("options_used").reset_index()
        used["season"] = y
        rows.append(used)
    out = pd.concat(rows, ignore_index=True)
    out["options_left"] = (3 - out.options_used).clip(lower=0)
    return out


def draft_pedigree() -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED / "draft.parquet").sort_values("draft_year")
    # Keep the draft the player signed in (the last time he was picked).
    d = d.drop_duplicates("mlbam_id", keep="last")
    d["log_pick"] = np.log(d.draft_pick.clip(lower=1))
    return d[["mlbam_id", "draft_year", "draft_round", "draft_pick", "log_pick", "signing_bonus"]]


def milb_lines(group: str) -> pd.DataFrame:
    """MiLB season lines collapsed to player-season-level (AAA..A), numeric."""
    m = pd.read_parquet(PROCESSED / f"milb_{group}.parquet")
    num = m.columns.difference(["mlbam_id", "name", "season", "level", "team_id", "parent_org_id",
                                "avg", "obp", "slg", "ops", "babip", "era", "whip", "inningsPitched"])
    for c in num:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    if "inningsPitched" in m:
        ip = pd.to_numeric(m.inningsPitched, errors="coerce")
        m["ip"] = np.floor(ip) + (ip - np.floor(ip)) * 10 / 3
    return m


def prospect_scores() -> pd.DataFrame:
    """Proxy prospect score per player-season from AA/AAA performance relative to level and age.

    Placeholder until FanGraphs Board FV grades are available.
    """
    m = milb_lines("hitting")
    m = m[m.level.isin(["AAA", "AA"]) & (m.plateAppearances >= 100)].copy()
    m["ops"] = pd.to_numeric(m.ops, errors="coerce")
    grp = m.groupby(["season", "level"])
    m["ops_z"] = (m.ops - grp.ops.transform("mean")) / grp.ops.transform("std")
    m["age_rel"] = m.age - grp.age.transform("mean")
    m["score"] = m.ops_z - 0.45 * m.age_rel + np.where(m.level == "AAA", 0.3, 0.0)
    ped = draft_pedigree().set_index("mlbam_id")
    m["score"] += np.where(m.mlbam_id.map(ped.draft_pick).fillna(999) <= 40, 0.5, 0.0)
    best = m.sort_values("score").drop_duplicates(["mlbam_id", "season"], keep="last")
    return best[["mlbam_id", "season", "level", "parent_org_id", "age", "plateAppearances", "score"]]


POS_GROUP = {"C": "C", "1B": "CI", "3B": "CI", "2B": "MI", "SS": "MI",
             "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH", "TWP": "DH"}


def rehab_by_season(t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rehab assignments per player-season: count and how late the last one was.

    A late-season rehab stint means the player was close to returning; the strongest
    available signal that a player who missed the year will be ready next spring.
    """
    t = transactions() if t is None else t
    r = t[t.description.str.contains("rehab assignment", case=False, na=False)].copy()
    r["season"] = r.date.dt.year
    g = r.groupby(["mlbam_id", "season"])
    out = pd.DataFrame({"rehab_n": g.size(), "rehab_last": g.date.max()}).reset_index()
    out["rehab_late"] = (out.rehab_last.dt.month >= 8).astype(int)
    out["rehab_month"] = out.rehab_last.dt.month + out.rehab_last.dt.day / 31
    return out.drop(columns="rehab_last")
