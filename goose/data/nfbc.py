"""NFBC ADP and average auction values (public, nfc.shgn.com/adp/baseball).

The page posts to /adp.data.php and gets back HTML table rows: rank, player, team, positions,
ADP (or $ for auctions), min, max, difference, # picks. Contest types used:
  0 all non-auction drafts, 889 Main Event, 890 Online Championship, 893 Draft Champions
  (50-round draft-and-hold), 896 average auction values.
NFBC ids are mapped to MLBAM by folded name, with team and pitcher/hitter as tie-breakers.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

import pandas as pd
import requests

from .http import RAW

URL = "https://nfc.shgn.com/adp.data.php"
CONTESTS = {"all": 0, "main": 889, "oc": 890, "dc": 893, "aav": 896}
OUT = RAW / "nfbc"
TEAM_FIX = {"ARZ": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH", "OAK": "ATH"}


def _rows(html: str) -> pd.DataFrame:
    out = []
    for tr in re.findall(r"<tr.*?</tr>", html, re.S):
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) < 9 or not cells[1]:
            continue
        num = lambda v: float(v.replace("$", "")) if v.replace("$", "").replace(".", "").isdigit() else None
        out.append(dict(rank=int(cells[0]), name=cells[1], team=TEAM_FIX.get(cells[2], cells[2]), pos=cells[3],
                        adp=num(cells[4]), min=num(cells[5]), max=num(cells[6]), picks=num(cells[8])))
    return pd.DataFrame(out)


SIZES = (10, 12, 15)  # league sizes NFBC reports ADP for


def fetch(contest: str, start: str = "", end: str = "", teams: int = 0) -> pd.DataFrame:
    data = dict(team_id=0, time_period="", from_date=start, to_date=end, num_teams=teams, draft_type=CONTESTS[contest],
                sport="baseball", position="", league_teams=0, as_board="")
    r = requests.post(URL, data=data, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return _rows(r.text)


def latest(season: int, min_rows: int = 150) -> tuple[dict[str, pd.DataFrame], str]:
    """ADP for each contest from the new draft season (after Oct 1 of `season`) when enough drafts
    exist; otherwise the site's default window (the previous preseason)."""
    start, end = f"{season}-10-01", dt.date.today().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    def pull(a: str, b: str) -> dict[str, pd.DataFrame]:
        out = {k: fetch(k, a, b) for k in CONTESTS}
        out.update({f"all_{n}": fetch("all", a, b, n) for n in SIZES})
        return out
    fresh = pull(start, end)
    if len(fresh["all"]) >= min_rows:
        res, label = fresh, f"{season + 1} drafts since {start}"
    else:
        res, label = pull("", ""), f"{season} preseason drafts (no {season + 1} drafts yet)"
    for k, d in res.items():
        d.to_csv(OUT / f"adp_{k}.csv", index=False)
    return res, label


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s.replace(".", ""))
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def map_ids(adp: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    """players: mlbam_id, name, team, is_pitcher. Returns mlbam_id per ADP row (NaN if unmatched)."""
    p = players.assign(key=players.name.map(_fold))
    groups = {k: g for k, g in p.groupby("key")}
    ids = []
    for r in adp.itertuples():
        g = groups.get(_fold(r.name))
        if g is None:
            ids.append(None)
            continue
        if len(g) > 1:
            pit = r.pos.replace(" ", "").split(",") == ["P"]
            g2 = g[g.is_pitcher == pit]
            g = g2 if len(g2) else g
            g3 = g[g.team == r.team]
            g = g3 if len(g3) else g
        ids.append(int(g.mlbam_id.iloc[0]))
    return pd.Series(ids, index=adp.index, dtype="Int64")
