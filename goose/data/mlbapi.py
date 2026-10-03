"""MLB Stats API: season stats, qualified pools, and player bios.

Player IDs here are MLBAM ids, the same ids Baseball Savant uses.
"""
from __future__ import annotations

import json

import pandas as pd

from .http import RAW, cached_text

BASE = "https://statsapi.mlb.com/api/v1"
OUT = RAW / "mlbapi"


def season_stats(season: int, group: str, pool: str = "all", refresh: bool = False) -> pd.DataFrame:
    """group: 'hitting' | 'pitching' | 'fielding'; pool: 'all' | 'qualified'."""
    path = OUT / "stats" / f"{group}_{pool}_{season}.json"
    params = {
        "stats": "season", "group": group, "season": season, "sportId": 1,
        "playerPool": pool, "limit": 5000, "gameType": "R",
    }
    data = json.loads(cached_text(path, f"{BASE}/stats", params=params, refresh=refresh))
    rows = []
    for split in data["stats"][0]["splits"] if data.get("stats") else []:
        row = {
            "mlbam_id": split["player"]["id"],
            "name": split["player"]["fullName"],
            "team": split.get("team", {}).get("abbreviation") or split.get("team", {}).get("name"),
            "season": season,
        }
        if "position" in split:
            row["position"] = split["position"].get("abbreviation")
        row.update(split["stat"])
        rows.append(row)
    return pd.DataFrame(rows)


def people(ids: list[int], refresh: bool = False) -> pd.DataFrame:
    """Bio info (birth date, handedness, primary position, debut) for a list of ids."""
    rows = []
    ids = sorted(set(int(i) for i in ids))
    for start in range(0, len(ids), 150):
        chunk = ids[start:start + 150]
        path = OUT / "people" / f"people_{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        params = {"personIds": ",".join(map(str, chunk))}
        data = json.loads(cached_text(path, f"{BASE}/people", params=params, refresh=refresh, pause=0.2))
        for p in data.get("people", []):
            rows.append({
                "mlbam_id": p["id"],
                "name": p.get("fullName"),
                "birth_date": p.get("birthDate"),
                "bats": p.get("batSide", {}).get("code"),
                "throws": p.get("pitchHand", {}).get("code"),
                "primary_position": p.get("primaryPosition", {}).get("abbreviation"),
                "mlb_debut": p.get("mlbDebutDate"),
                "height": p.get("height"),
                "weight": p.get("weight"),
                "active": p.get("active"),
                "current_team_id": p.get("currentTeam", {}).get("id"),
            })
    return pd.DataFrame(rows)


MILB_SPORTS = {11: "AAA", 12: "AA", 13: "A+", 14: "A"}


def milb_season_stats(season: int, group: str, refresh: bool = False) -> pd.DataFrame:
    """Minor league season lines for AAA..A, one row per player-team-level, with parent org."""
    frames = []
    for sport_id, level in MILB_SPORTS.items():
        path = OUT / "milb" / f"{group}_{sport_id}_{season}.json"
        params = {"stats": "season", "group": group, "season": season, "sportId": sport_id,
                  "playerPool": "all", "limit": 10000, "gameType": "R"}
        data = json.loads(cached_text(path, f"{BASE}/stats", params=params, refresh=refresh))
        teams = milb_teams(season, sport_id, refresh)
        rows = []
        for split in data["stats"][0]["splits"] if data.get("stats") else []:
            tid = split.get("team", {}).get("id")
            row = {"mlbam_id": split["player"]["id"], "name": split["player"].get("fullName"),
                   "season": season, "level": level, "team_id": tid,
                   "parent_org_id": teams.get(tid)}
            row.update(split["stat"])
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def milb_team_leagues(season: int, sport_id: int, refresh: bool = False) -> dict[int, int]:
    path = OUT / "teams" / f"teams_{sport_id}_{season}.json"
    data = json.loads(cached_text(path, f"{BASE}/teams", params={"sportId": sport_id, "season": season},
                                  refresh=refresh))
    return {t["id"]: t.get("league", {}).get("id") for t in data.get("teams", [])}


def milb_teams(season: int, sport_id: int, refresh: bool = False) -> dict[int, int]:
    path = OUT / "teams" / f"teams_{sport_id}_{season}.json"
    data = json.loads(cached_text(path, f"{BASE}/teams", params={"sportId": sport_id, "season": season},
                                  refresh=refresh))
    return {t["id"]: t.get("parentOrgId") for t in data.get("teams", [])}


def mlb_teams(season: int, refresh: bool = False) -> pd.DataFrame:
    path = OUT / "teams" / f"teams_1_{season}.json"
    data = json.loads(cached_text(path, f"{BASE}/teams", params={"sportId": 1, "season": season},
                                  refresh=refresh))
    return pd.DataFrame([{"team_id": t["id"], "abbrev": t["abbreviation"], "name": t["name"],
                          "league": t["league"]["name"], "division": t["division"]["name"]}
                         for t in data["teams"]])


def draft(year: int, refresh: bool = False) -> pd.DataFrame:
    path = OUT / "draft" / f"{year}.json"
    data = json.loads(cached_text(path, f"{BASE}/draft/{year}", refresh=refresh))
    rows = []
    for rnd in data["drafts"]["rounds"]:
        for p in rnd["picks"]:
            if "person" not in p:
                continue
            rows.append({"mlbam_id": p["person"]["id"], "draft_year": year,
                         "draft_round": rnd.get("round"), "draft_pick": p.get("pickNumber"),
                         "signing_bonus": pd.to_numeric(p.get("signingBonus"), errors="coerce"),
                         "pick_value": pd.to_numeric(p.get("pickValue"), errors="coerce"),
                         "draft_team_id": p.get("team", {}).get("id"),
                         "school_class": p.get("school", {}).get("schoolClass")})
    return pd.DataFrame(rows)


def transactions(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    path = OUT / "transactions" / f"{start}_{end}.json"
    data = json.loads(cached_text(path, f"{BASE}/transactions",
                                  params={"startDate": start, "endDate": end, "sportId": 1}, refresh=refresh))
    rows = []
    for t in data.get("transactions", []):
        if "person" not in t:
            continue
        rows.append({"mlbam_id": t["person"]["id"], "date": t.get("date") or t.get("effectiveDate"),
                     "effective_date": t.get("effectiveDate"), "type_code": t.get("typeCode"),
                     "type": t.get("typeDesc"), "from_team_id": t.get("fromTeam", {}).get("id"),
                     "to_team_id": t.get("toTeam", {}).get("id"), "description": t.get("description")})
    return pd.DataFrame(rows)


def roster(team_id: int, roster_type: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """roster_type: 'depthChart' | '40Man' | 'active' | 'fullRoster'."""
    path = OUT / "rosters" / f"{roster_type}_{team_id}_{season}.json"
    data = json.loads(cached_text(path, f"{BASE}/teams/{team_id}/roster",
                                  params={"rosterType": roster_type, "season": season}, refresh=refresh))
    return pd.DataFrame([{"team_id": team_id, "mlbam_id": r["person"]["id"], "name": r["person"]["fullName"],
                          "position": r["position"]["abbreviation"],
                          "status": r.get("status", {}).get("description"), "order": i}
                         for i, r in enumerate(data.get("roster", []))])
