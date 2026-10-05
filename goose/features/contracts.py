"""Contracts from RosterResource payroll exports (data/raw/user/payroll/<Team>-Payroll-2026.xlsx).

Per player: team, contract section (guaranteed / arbitration / pre-arb / off the 40-man), AAV,
2026 salary, 2027 status (guaranteed $, ARB n, Pre-ARB, FREE AGENT), years remaining after 2026.
FanGraphs ids are mapped to MLBAM via the Chadwick register (name fallback).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from goose.data.http import ROOT

SRC = ROOT / "data" / "raw" / "user" / "payroll"
OUT = ROOT / "data" / "processed" / "contracts_2026.parquet"
SECTIONS = {"Guaranteed": "guaranteed", "Eligible For Arb": "arb", "Not Yet Eligible For Arb": "pre_arb",
            "No Longer On 40-Man Roster": "off_40"}
TEAM_ABBR = {"Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR", "Braves": "ATL",
             "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "AZ", "Dodgers": "LAD",
             "Giants": "SF", "Guardians": "CLE", "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM",
             "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
             "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "Royals": "KC",
             "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY"}


def _money(v) -> float:
    if isinstance(v, (int, float)) and not pd.isna(v):
        return float(v)
    m = re.match(r"^-?\$([\d,\.]+)", str(v).strip())
    return float(m.group(1).replace(",", "")) * (-1 if str(v).strip().startswith("-") else 1) if m else np.nan


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace(" jr", "")).strip()


def build() -> pd.DataFrame:
    rows = []
    for f in sorted(SRC.glob("*-Payroll-2026.xlsx")):
        team = TEAM_ABBR[f.name.split("-Payroll")[0]]
        for sheet, d in pd.read_excel(f, sheet_name=None).items():
            if sheet not in SECTIONS or "Player" not in d:
                continue
            for r in d.itertuples(index=False):
                rd = r._asdict()
                rows.append(dict(team=team, section=SECTIONS[sheet], player=rd["Player"], fg_id=str(rd.get("playerId")),
                                 age=rd.get("Age"), service=rd.get("_2"), contract=rd.get("Contract"),
                                 aav=_money(rd.get("AAV")), sal_2026=_money(d.loc[0, 2026] if False else rd.get("_6")),
                                 s2027=rd.get("_7"), s2028=rd.get("_8"), s2029=rd.get("_9"), s2030=rd.get("_10")))
    c = pd.DataFrame(rows)
    c["sal_2027"] = c.s2027.map(_money)
    c["status_2027"] = np.select([c.sal_2027.notna(), c.s2027.astype(str).str.contains("FREE AGENT"),
                                  c.s2027.astype(str).str.startswith("ARB"), c.s2027.astype(str).str.contains("Pre-ARB")],
                                 ["guaranteed", "free_agent", "arb", "pre_arb"], "unknown")
    fut = c[["s2027", "s2028", "s2029", "s2030"]].map(lambda v: not pd.isna(_money(v)))
    c["years_left"] = fut.sum(axis=1)
    m = pd.read_parquet(ROOT / "data" / "processed" / "chadwick_fg_mlbam.parquet")
    c["mlbam_id"] = c.fg_id.map(dict(zip(m.key_fangraphs.astype(str), m.key_mlbam)))
    bios = pd.read_parquet(ROOT / "data" / "processed" / "players.parquet").drop_duplicates("mlbam_id")
    by_name = dict(zip(bios.name.map(_norm), bios.mlbam_id))
    c["mlbam_id"] = c.mlbam_id.fillna(c.player.map(lambda n: by_name.get(_norm(n))))
    c = c.drop(columns=["s2027", "s2028", "s2029", "s2030"])
    c.to_parquet(OUT, index=False)
    return c


if __name__ == "__main__":
    c = build()
    print(len(c), "rows;", c.mlbam_id.notna().mean().round(3), "mapped;", c.section.value_counts().to_dict())
    print(c.status_2027.value_counts().to_dict())
    print(c[c.player.isin(["Colt Emerson", "Willson Contreras", "Aaron Judge", "Willy Adames"])][
        ["player", "team", "section", "contract", "aav", "sal_2026", "sal_2027", "status_2027", "years_left"]].to_string(index=False))
