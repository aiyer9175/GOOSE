"""Surgery-recovery adjustment from the RosterResource injury report (end of 2026).

A procedure gets a typical recovery time by role (published ranges; no historical version of the
report exists), which sets the return date and the share of 2027 left. Playing time after a
mid-season return comes from a model fit on 165 historical returns from 90+ day IL stints
(data/models/return_ratios.json): the share of healthy pace players actually reached for the rest of
the season, split into "returned as planned" (headline) vs. setbacks (carried in P10 and the
probability). A first-year innings cap after elbow reconstruction applies only to pitchers who
haven't returned yet. Every adjusted player is flagged with the reason.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from goose.data.http import ROOT

REPORT = ROOT / "data" / "processed" / "injury_report_2026.parquet"
OPENING_DAY, SEASON_END = pd.Timestamp("2027-03-26"), pd.Timestamp("2027-09-26")
SEASON_DAYS = (SEASON_END - OPENING_DAY).days

# (pattern, pitcher months, hitter months, elbow reconstruction?) - first match wins.
RULES = [
    (r"revision", 16, 10, True),
    (r"tommy john(?!.*brace)|ucl reconstruction", 14, 8, True),
    (r"internal brace|ucl repair|ucl/flexor|ucl surgery", 10, 6, True),
    (r"flexor", 7, 5, False),
    (r"thoracic outlet", 7, 5, False),
    (r"arthroscopic (elbow|shoulder)", 5, 3, False),
    (r"labrum|shoulder surgery", 12, 6, False),
    (r"acl|patellar tendon \(rupture|ruptured patellar|lisfranc", 10, 9, False),
    (r"elbow surgery|stress fracture", 6, 4, False),
    (r"lat surgery", 6, 5, False),
    (r"arthroscopic (hip|knee|wrist|ankle)", 4, 3, False),
    (r"hip surgery|ankle surgery|knee surgery|torn tendon", 6, 5, False),
    (r"core muscle|sports hernia", 2, 2, False),
    (r"wrist surgery|finger surgery|biceps|radius|shin|back surgery|blood clot", 4, 3, False),
]


def recovery(desc: str, pitcher: bool) -> tuple[float, bool] | None:
    d = str(desc).lower()
    if "surgery" not in d and "tommy john" not in d and "brace" not in d:
        return None
    for pat, mp, mh, elbow in RULES:
        if re.search(pat, d):
            return (mp if pitcher else mh), elbow
    return (5.0 if pitcher else 4.0), False


def adjustments(ids: pd.Series, pitcher: bool) -> pd.DataFrame:
    """Per player: share of 2027 available, ramp factor, IP cap, and a note."""
    rep = pd.read_parquet(REPORT)
    rep["MLBAMID"] = pd.to_numeric(rep.MLBAMID, errors="coerce")
    rep = rep.dropna(subset=["MLBAMID", "date"]).sort_values("date").drop_duplicates("MLBAMID", keep="last")
    rep = rep.set_index(rep.MLBAMID.astype(int))
    out = pd.DataFrame(index=ids.index, data={"avail": 1.0, "ramp": 1.0, "cap": np.inf, "injury_note": "",
                                               "arm": False, "pending": False})
    for i, pid in ids.items():
        if pid not in rep.index:
            continue
        r = rep.loc[pid]
        rec = recovery(r["Injury / Surgery"], pitcher)
        if rec is None:
            continue
        months, elbow = rec
        back = r.date + pd.DateOffset(months=int(months)) + pd.Timedelta(days=int((months % 1) * 30))
        avail = float(np.clip((SEASON_END - back).days / SEASON_DAYS, 0, 1))
        ramp = 1.0  # post-return pace comes from the historical return model (see apply in project)
        # First-full-season cap only for pitchers who haven't come back yet; anyone who already
        # returned and pitched (e.g. 100+ IP post-TJ) has shown health and goes through the model.
        not_back_yet = back > pd.Timestamp("2026-10-01")
        cap = 120.0 if (pitcher and elbow and not_back_yet) else np.inf
        if avail >= 1 and np.isinf(cap) and not not_back_yet:
            continue
        arm = bool(re.search(r"elbow|ucl|tommy|brace|flexor|shoulder|labrum|thoracic", str(r["Injury / Surgery"]).lower()))
        out.loc[i] = [avail, ramp, cap, f"{r['Injury / Surgery']} ({r.date:%b %Y}); est. return {back:%b %Y}", arm,
                      not_back_yet]
    return out
