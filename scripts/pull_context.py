"""Pull context data: MiLB stats, drafts, transactions, current rosters/depth charts.

Outputs (data/processed/): milb_hitting.parquet, milb_pitching.parquet, draft.parquet,
transactions.parquet, rosters_2026.parquet, mlb_teams.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data import mlbapi  # noqa: E402
from goose.data.http import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SEASONS = range(2012, 2027)


def main() -> None:
    teams = mlbapi.mlb_teams(2026)
    teams.to_parquet(PROCESSED / "mlb_teams.parquet", index=False)

    for group in ("hitting", "pitching"):
        df = pd.concat([mlbapi.milb_season_stats(y, group) for y in SEASONS if y != 2020], ignore_index=True)
        df.to_parquet(PROCESSED / f"milb_{group}.parquet", index=False)
        print(f"milb {group}: {len(df):,} rows", flush=True)

    d = pd.concat([mlbapi.draft(y) for y in range(2005, 2027)], ignore_index=True)
    d.to_parquet(PROCESSED / "draft.parquet", index=False)
    print(f"draft: {len(d):,} picks", flush=True)

    frames = []
    for y in SEASONS:
        for start, end in ((f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")):
            frames.append(mlbapi.transactions(start, end))
    t = pd.concat(frames, ignore_index=True).drop_duplicates()
    t.to_parquet(PROCESSED / "transactions.parquet", index=False)
    print(f"transactions: {len(t):,}", flush=True)

    r = []
    for tid in teams.team_id:
        for rt in ("40Man", "depthChart"):
            df = mlbapi.roster(tid, rt, 2026)
            df["roster_type"] = rt
            r.append(df)
    pd.concat(r, ignore_index=True).to_parquet(PROCESSED / "rosters_2026.parquet", index=False)
    print("rosters done", flush=True)


if __name__ == "__main__":
    main()
