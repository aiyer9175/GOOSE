"""Public Fantrax leagues by league ID (the code in the league URL), via Fantrax's open API:
  /fxea/general/getLeagueInfo?leagueId=  settings, scoring, roster slots, teams
  /fxea/general/getTeamRosters?leagueId= rosters (Fantrax player ids; salary/contract when the league uses them)
  /fxea/general/getPlayerIds?sport=MLB   Fantrax id -> name, team, position
No login needed. Scoring is translated into GOOSE's format where a stat has a GOOSE equivalent;
near-equivalents are listed so the page can say what was approximated.
"""
from __future__ import annotations

import json
import re
import time

import requests

from .http import RAW

API = "https://www.fantrax.com/fxea/general"
OUT = RAW / "fantrax"

# Fantrax short names -> GOOSE category keys (categories leagues)
HIT_CATS = {"HR": "HR", "R": "R", "RBI": "RBI", "SB": "SB", "AVG": "AVG", "OBP": "OBP", "SLG": "SLG", "OPS": "OPS",
            "TB": "TB", "H": "H", "BB": "BB", "XBH": "XBH", "1B": "1B", "2B": "2B", "3B": "3B", "SO": "K", "K": "K",
            "CS": "CS", "SBN": "NSB", "NSB": "NSB", "ISO": "ISO", "HBP": "HBP", "R+RBI": "R+RBI", "PA": "PA", "AB": "AB"}
PIT_CATS = {"W": "W", "SV": "SV", "K": "K", "ERA": "ERA", "WHIP": "WHIP", "QS": "QS", "HLD": "HLD", "HD": "HLD",
            "SV+H": "SVHD", "SVH": "SVHD", "SV+HLD": "SVHD", "L": "L", "IP": "IP", "K/9": "K/9", "BB/9": "BB/9",
            "HR/9": "HR/9", "K/BB": "K/BB", "W+QS": "W+QS", "GS": "GS"}
APPROX = {"MWHIP": ("WHIP", "Modified WHIP ≈ WHIP"), "WCS": ("W", "W + CG + SHO ≈ W"),
          "IRS+H": ("HLD", "inherited runners stranded + holds ≈ holds"), "K-BB": ("K", "K − 0.5 BB ≈ K"),
          "2B+3B": ("XBH", "2B + 3B ≈ XBH"), "1B+BB": ("H", "1B + BB ≈ H")}
# Fantrax short names -> GOOSE point keys (points leagues)
HIT_PTS = {"1B": "b1", "2B": "b2", "3B": "b3", "HR": "hr", "R": "r", "RBI": "rbi", "BB": "bb", "SB": "sb", "CS": "cs",
           "SO": "k", "K": "k", "HBP": "hbp", "HP": "hbp", "TB": "tb", "AB": "ab", "H": "h", "PA": "pa"}
PIT_PTS = {"IP": "ip", "K": "k", "W": "w", "L": "l", "SV": "sv", "HLD": "hld", "HD": "hld", "ER": "er", "H": "h",
           "BB": "bb", "HR": "hr", "HB": "hbp", "HBP": "hbp", "QS": "qs", "GS": "gs"}
SLOT = {"C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS", "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
        "UT": "UTIL", "DH": "UTIL", "MI": "MI", "CI": "CI", "INF": "UTIL", "SP": "SP", "RP": "RP", "P": "P"}


def _get(path: str, **params) -> dict:
    r = requests.get(f"{API}/{path}", params=params, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"  # Fantrax sends UTF-8 without a charset (team names with curly quotes, accents)
    return json.loads(r.text)


def player_ids(max_age_h: float = 24) -> dict:
    f = OUT / "player_ids.json"
    if f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600:
        return json.loads(f.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    d = _get("getPlayerIds", sport="MLB")
    f.write_text(json.dumps(d))
    return d


def _name(n: str) -> str:
    last, _, first = n.partition(", ")
    return f"{first} {last}".strip() if first else n


def settings(info: dict) -> dict:
    ss = info.get("scoringSystem", {})
    kind = ss.get("type", "")
    def num(v: dict) -> float:  # {"Default": "1.0"} or {"Default": "points0.8", "SP": ...}: use the default weight
        raw = str(v.get("Default", next(iter(v.values()), 0)))
        m = re.search(r"-?\d+(\.\d+)?", raw)
        return float(m.group(0)) if m else 0.0
    cats = {g: {k: num(v) for k, v in d.items()} for g, d in ss.get("scoringCategories", {}).items()}
    hit, pit = cats.get("HITTING", {}), cats.get("PITCHING", {})
    notes, out = [], {"type": kind}
    if "POINTS" in kind:
        out["mode"] = "pts"
        out["hw"] = {HIT_PTS[k]: w for k, w in hit.items() if w and k in HIT_PTS}
        out["pw"] = {PIT_PTS[k]: w for k, w in pit.items() if w and k in PIT_PTS}
        # Combined stats spread onto their parts.
        for k, parts in {"XBH": ("b2", "b3", "hr")}.items():
            if hit.get(k):
                for q in parts: out["hw"][q] = out["hw"].get(q, 0) + hit[k]
        for k in ("SVH", "SV+H", "SV+HLD"):
            if pit.get(k):
                for q in ("sv", "hld"): out["pw"][q] = out["pw"].get(q, 0) + pit[k]
        done = {"XBH", "SVH", "SV+H", "SV+HLD"}
        miss = [k for k, w in {**hit, **pit}.items() if w and k not in HIT_PTS and k not in PIT_PTS and k not in done]
    else:
        out["mode"] = "cats"
        def tr(d, table):
            keys, miss = [], []
            for k, w in d.items():
                if not w:
                    continue
                if k in table:
                    keys.append(table[k])
                elif k in APPROX:
                    keys.append(APPROX[k][0]); notes.append(APPROX[k][1])
                else:
                    miss.append(k)
            return list(dict.fromkeys(keys)), miss
        out["hc"], m1 = tr(hit, HIT_CATS)
        out["pc"], m2 = tr(pit, PIT_CATS)
        miss = m1 + m2
    if miss:
        notes.append("not modeled: " + ", ".join(miss))
    ri = info.get("rosterInfo", {})
    roster = {k: 0 for k in ["C", "1B", "2B", "3B", "SS", "OF", "MI", "CI", "UTIL", "SP", "RP", "P"]}
    for pos, c in ri.get("positionConstraints", {}).items():
        if pos in SLOT:
            roster[SLOT[pos]] += int(c.get("maxActive", 0) or 0)
    roster["bench"] = max(0, int(ri.get("maxTotalPlayers", 0) or 0) - int(ri.get("maxTotalActivePlayers", 0) or 0))
    roster["teams"] = len(info.get("teamInfo", {}))
    out["roster"] = roster
    out["notes"] = notes
    return out


def league(league_id: str) -> dict:
    info = _get("getLeagueInfo", leagueId=league_id)
    ros = _get("getTeamRosters", leagueId=league_id)
    ids = player_ids()
    players = []
    for team in ros.get("rosters", {}).values():
        for it in team.get("rosterItems", []):
            p = ids.get(it["id"], {})
            players.append({"team": team.get("teamName", ""), "name": _name(p.get("name", it["id"])), "fg": "", "fgm": "",
                            "salary": float(it.get("salary") or 0), "pos": it.get("position", ""), "mlb_team": p.get("team")})
    s = settings(info)
    return {"id": league_id, "platform": "fantrax", "name": info.get("leagueName", league_id),
            "scoring": s["type"].replace("_", " ").title(), "season": info.get("seasonYear"), "teams": s["roster"]["teams"],
            "settings": s, "players": players, "salaries": any(p["salary"] for p in players)}
