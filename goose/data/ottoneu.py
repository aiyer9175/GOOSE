"""Public Ottoneu leagues: rosters (CSV export) and settings, by league ID.

Ottoneu pages sit behind a Cloudflare browser check that blocks plain HTTP clients, so requests go
through curl_cffi with a Chrome TLS fingerprint. No login is needed for public leagues.
"""
from __future__ import annotations

import csv
import io
import re

from curl_cffi import requests

BASE = "https://ottoneu.fangraphs.com"
SCORING = {"FanGraphs Points": "ottoFg", "SABR Points": "ottoSabr", "Old School": "ottoOld", "Classic": "ottoClassic"}


def _get(path: str) -> str:
    r = requests.get(f"{BASE}{path}", impersonate="chrome", timeout=60)
    r.raise_for_status()
    return r.text


def settings(league_id: int) -> dict:
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", _get(f"/{league_id}/settings")))
    def field(name: str) -> str | None:
        m = re.search(re.escape(name) + r"[ |]+([^|]+?) \|", t)
        return m.group(1).strip() if m else None
    name = re.search(r"League Settings[ |]+([^|]+?) \|", t)
    scoring = field("Scoring System") or ""
    preset = next((v for k, v in SCORING.items() if k.lower() in scoring.lower()), None)
    teams = field("Number of Teams")
    return {"id": league_id, "name": name.group(1).strip() if name else str(league_id), "scoring": scoring,
            "preset": preset, "teams": int(teams) if teams and teams.isdigit() else None, "lineups": field("Lineups")}


def team_cap(league_id: int, team_id: str) -> dict:
    """Cap line from a team page: "Salary Cap: $400 (base) + $6 (loans in) - $50 (loans out)",
    "Cap Used: $209 (salary) + $92 (cap penalties)". The roster export has salaries only."""
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _get(f"/{league_id}/team/{team_id}")))
    amt = lambda pat: int(m.group(1)) if (m := re.search(pat, t)) else 0
    base = amt(r"\$(\d+) \(base\)") or 400
    loans_in, loans_out = amt(r"\$(\d+) \(loans in\)"), amt(r"\$(\d+) \(loans out\)")
    return {"cap": base + loans_in - loans_out, "loans_in": loans_in, "loans_out": loans_out,
            "penalties": amt(r"\$(\d+) \(cap penalties\)")}


def rosters(league_id: int) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(_get(f"/{league_id}/rosterexport?csv=1"))))
    return [{"team": r["Team Name"], "team_id": r.get("TeamID", ""), "name": r["Name"], "fg": r.get("FG MajorLeagueID", ""), "fgm": r.get("FG MinorLeagueID", ""),
             "salary": float(str(r.get("Salary", "0")).replace("$", "").replace(",", "") or 0), "pos": r.get("Position(s)", "")}
            for r in rows if r.get("Name")]


def league(league_id: int) -> dict:
    players = rosters(league_id)
    ids = {p["team"]: p["team_id"] for p in players}
    caps = {}
    for name, tid in ids.items():
        try:
            caps[name] = team_cap(league_id, tid)
        except Exception:  # cap details are a bonus; salaries still work without them
            pass
    return {**settings(league_id), "players": players, "caps": caps}
