"""Build a self-contained HTML projection browser from data/projections/2027_*.csv.

Usage: python scripts/build_browser.py [output.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.features.eligibility import eligibility, pitcher_eligibility, projected_games  # noqa: E402
from goose.data import nfbc  # noqa: E402
from goose.data import ottoneu  # noqa: E402
from goose.data import fantrax  # noqa: E402

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "data" / "projections"

HIT_COLS = {
    "id": "mlbam_id", "name": "Name", "team": "Team_2026", "age": "Age_2027", "pos": "Pos", "pa26": "PA_2026",
    "career": "career_PA", "pa": "PA", "pa10": "PA_p10", "pa50": "PA_p50", "pa90": "PA_p90", "p300": "P_300PA",
    "avg": "AVG", "obp": "OBP", "slg": "SLG", "ops": "OPS", "woba": "wOBA", "woba10": "wOBA_p10",
    "woba90": "wOBA_p90", "hr": "HR", "r": "R", "rbi": "RBI", "sb": "SB", "cs": "CS", "bb": "BB", "k": "K",
    "kpct": "K%", "bbpct": "BB%", "babip": "BABIP", "iso": "ISO", "vol": "volatility_pctile", "ptrisk": "pt_risk",
    "frag": "fragility", "opts": "options_left", "threat": "prospect_threat", "il": "il_days", "focus": "focus",
    "fa": "free_agent", "pah": "PA_healthy", "status": "status", "injury": "injury_note",
    "role": "platoon_role", "palshare": "pa_vs_lhp_share", "wvr": "wOBA_vsR", "wvl": "wOBA_vsL", "pfull": "P_full_season",
    "ab": "AB", "h": "H", "b1": "1B", "b2": "2B", "b3": "3B", "hbp": "HBP", "sf": "SF", "g": "G", "bats": "Bats",
}
PIT_COLS = {
    "id": "mlbam_id", "name": "Name", "team": "Team_2026", "age": "Age_2027", "throws": "Throws",
    "ip26": "IP_2026", "gs26": "GS_2026", "tip26": "total_IP_2026", "ip": "IP", "ip10": "IP_p10", "ip50": "IP_p50",
    "ip90": "IP_p90", "p100": "P_100IP", "p150": "P_150IP", "gs": "GS", "tbf": "TBF", "era": "ERA", "fip": "FIP",
    "whip": "WHIP", "k": "K", "bb": "BB", "hr": "HR", "kpct": "K%", "bbpct": "BB%", "kbb": "K-BB%", "hr9": "HR/9",
    "w": "W", "l": "L", "qs": "QS", "arm": "arm_injury_2y", "il": "il_days_2026", "frag": "fragility",
    "opts": "options_left", "focus": "focus", "fa": "free_agent", "iph": "IP_healthy", "status": "status",
    "stuff": "stuff_plus", "override": "override", "injury": "injury_note", "pfull": "P_full_season",
    "rank": "rotation_rank", "prot": "P_rotation", "role": "role_2026",
    "h": "H", "er": "ER", "hbp": "HBP", "sv": "SV", "hld": "HLD",
}


RP_COLS = {
    "id": "mlbam_id", "name": "Name", "team": "Team_2026", "age": "Age_2027", "throws": "Throws",
    "ip26": "IP_2026", "g26": "G_2026", "gs26": "GS_2026", "sv26": "SV_2026", "hld26": "HLD_2026",
    "ip": "IP", "ip10": "IP_p10", "ip50": "IP_p50", "ip90": "IP_p90", "sv": "SV", "sv10": "SV_p10",
    "sv50": "SV_p50", "sv90": "SV_p90", "p20sv": "P_20SV", "p10sv": "P_10SV", "hld": "HLD", "era": "ERA",
    "fip": "FIP", "whip": "WHIP", "k": "K", "bb": "BB", "kpct": "K%", "bbpct": "BB%", "kbb": "K-BB%",
    "w": "W", "l": "L", "il": "il_days_2026", "frag": "fragility", "fa": "free_agent", "swing": "swingman",
    "stuff": "stuff_plus", "override": "override", "status": "status", "injury": "injury_note",
    "svshare": "team_sv_share", "h": "H", "er": "ER", "hbp": "HBP", "hr": "HR", "tbf": "TBF", "pconv": "P_convert",
}


def records(df: pd.DataFrame, cols: dict) -> list[dict]:
    cols = {k: v for k, v in cols.items() if v in df}
    out = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    out = out.replace([np.inf, -np.inf], np.nan)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return json.loads(out.to_json(orient="records"))


def ottoneu_leagues() -> list[dict]:
    """Preload leagues listed in data/raw/user/ottoneu_leagues.txt and fantrax_leagues.txt (one ID per line)."""
    f = ROOT / "data" / "raw" / "user" / "ottoneu_leagues.txt"
    out = []
    for line in (f.read_text().split() if f.exists() else []):
        try:
            lg = ottoneu.league(int(line))
            out.append(lg)
            print(f"Ottoneu {line}: {lg['name']} ({lg['scoring']}, {lg['teams']} teams), {len(lg['players'])} rostered")
        except Exception as e:
            print(f"Ottoneu {line}: failed ({e})")
    fx = ROOT / "data" / "raw" / "user" / "fantrax_leagues.txt"
    for line in (fx.read_text().split() if fx.exists() else []):
        try:
            lg = fantrax.league(line)
            out.append(lg)
            print(f"Fantrax {line}: {lg['name']} ({lg['scoring']}, {lg['season']}, {lg['teams']} teams), {len(lg['players'])} rostered")
        except Exception as e:
            print(f"Fantrax {line}: failed ({e})")
    return out


def fg_map(ids: pd.Series) -> dict[str, int]:
    """FanGraphs id -> MLBAM id for projected players (Ottoneu roster exports carry FG ids)."""
    m = pd.read_parquet(ROOT / "data" / "processed" / "chadwick_fg_mlbam.parquet").dropna(subset=["key_fangraphs", "key_mlbam"])
    m = m[m.key_mlbam.isin(set(ids))]
    return {str(int(f)) if str(f).replace(".0", "").isdigit() else str(f): int(i) for f, i in zip(m.key_fangraphs, m.key_mlbam)}


def adp_block(h: pd.DataFrame, p: pd.DataFrame, r: pd.DataFrame) -> dict:
    """NFBC ADP / auction values keyed by MLBAM id: {contest: {id: [adp, min, max, picks]}}."""
    try:
        sets, label = nfbc.latest(2026)
    except Exception as e:  # offline: use the last saved pull
        print("NFBC fetch failed, using cached:", e)
        sets = {f.stem[4:]: pd.read_csv(f) for f in nfbc.OUT.glob("adp_*.csv")}
        label = "cached NFBC pull"
    players = pd.concat([h[["mlbam_id", "Name", "Team_2026"]].assign(is_pitcher=False),
                         p[["mlbam_id", "Name", "Team_2026"]].assign(is_pitcher=True),
                         r[["mlbam_id", "Name", "Team_2026"]].assign(is_pitcher=True)]).drop_duplicates(["mlbam_id", "is_pitcher"])
    players = players.rename(columns={"Name": "name", "Team_2026": "team"})
    out = {"label": label, "sets": {}}
    for k, d in sets.items():
        ids = nfbc.map_ids(d, players)
        m = d.assign(id=ids).dropna(subset=["id", "adp"])
        out["sets"][k] = {str(int(i)): [a, lo, hi, n] for i, a, lo, hi, n in zip(m.id, m.adp, m["min"], m["max"], m.picks)}
        print(f"ADP {k}: {len(d)} rows, {ids.notna().mean():.0%} matched; unmatched top-300:",
              d[ids.isna() & (d["rank"] <= 300)].name.head(8).tolist())
    return out


def main(dest: Path) -> None:
    h = pd.read_csv(PROJ / "2027_hitters.csv")
    p = pd.read_csv(PROJ / "2027_pitchers.csv")
    r = pd.read_csv(PROJ / "2027_relievers.csv")
    h["G"] = projected_games(h, 2026)
    bios = pd.read_parquet(ROOT / "data" / "processed" / "players.parquet").drop_duplicates("mlbam_id").set_index("mlbam_id")
    h["Bats"] = h.mlbam_id.map(bios.bats)
    elig = eligibility(h.mlbam_id, 2026)
    adp = adp_block(h, p, r)
    data = {"hitters": records(h, HIT_COLS), "pitchers": records(p, PIT_COLS), "relievers": records(r, RP_COLS),
            "elig": {str(k): v for k, v in elig.items()}, "adp": adp,
            "leagues": ottoneu_leagues(),
            "fgmap": fg_map(pd.concat([h.mlbam_id, p.mlbam_id, r.mlbam_id])),
            "pelig": {str(k): v for k, v in pitcher_eligibility(pd.concat([p.mlbam_id, r.mlbam_id]), 2026).items()}}
    html = (ROOT / "scripts" / "browser_template.html").read_text()
    html = html.replace("<!--__FANTASY__-->", (ROOT / "scripts" / "fantasy_template.html").read_text())
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    dest.write_text(html)
    print(f"wrote {dest} ({dest.stat().st_size / 1e3:.0f} KB): {len(h)} hitters, {len(p)} pitchers")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "browser.html")
