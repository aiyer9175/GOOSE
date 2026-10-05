"""Build a self-contained HTML projection browser from data/projections/2027_*.csv.

Usage: python scripts/build_browser.py [output.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
}


RP_COLS = {
    "id": "mlbam_id", "name": "Name", "team": "Team_2026", "age": "Age_2027", "throws": "Throws",
    "ip26": "IP_2026", "g26": "G_2026", "gs26": "GS_2026", "sv26": "SV_2026", "hld26": "HLD_2026",
    "ip": "IP", "ip10": "IP_p10", "ip50": "IP_p50", "ip90": "IP_p90", "sv": "SV", "sv10": "SV_p10",
    "sv50": "SV_p50", "sv90": "SV_p90", "p20sv": "P_20SV", "p10sv": "P_10SV", "hld": "HLD", "era": "ERA",
    "fip": "FIP", "whip": "WHIP", "k": "K", "bb": "BB", "kpct": "K%", "bbpct": "BB%", "kbb": "K-BB%",
    "w": "W", "l": "L", "il": "il_days_2026", "frag": "fragility", "fa": "free_agent", "swing": "swingman",
    "stuff": "stuff_plus", "override": "override", "status": "status", "injury": "injury_note",
    "svshare": "team_sv_share",
}


def records(df: pd.DataFrame, cols: dict) -> list[dict]:
    cols = {k: v for k, v in cols.items() if v in df}
    out = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    out = out.replace([np.inf, -np.inf], np.nan)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return json.loads(out.to_json(orient="records"))


def main(dest: Path) -> None:
    h = pd.read_csv(PROJ / "2027_hitters.csv")
    p = pd.read_csv(PROJ / "2027_pitchers.csv")
    r = pd.read_csv(PROJ / "2027_relievers.csv")
    data = {"hitters": records(h, HIT_COLS), "pitchers": records(p, PIT_COLS), "relievers": records(r, RP_COLS)}
    html = (ROOT / "scripts" / "browser_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    dest.write_text(html)
    print(f"wrote {dest} ({dest.stat().st_size / 1e3:.0f} KB): {len(h)} hitters, {len(p)} pitchers")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "browser.html")
