"""Pull the 2026 qualified pools plus 2015-2026 season history from MLB API + Savant.

Usage:
    python scripts/pull_data.py                 # leaderboards + MLB API, 2015-2026
    python scripts/pull_data.py --first 2023    # shorter history
    python scripts/pull_data.py --refresh       # re-download the current season's files

Outputs (data/processed/):
    qualified_2026_hitters.csv / qualified_2026_pitchers.csv
    hitter_seasons.parquet / pitcher_seasons.parquet   one row per (mlbam_id, season)
    pitch_arsenal_seasons.parquet                       one row per (pitcher, season, pitch_type)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data import mlbapi, savant  # noqa: E402
from goose.data.http import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
CURRENT = 2026


def _prefixed(df: pd.DataFrame, prefix: str, keep: tuple[str, ...] = ("player_id",)) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop(columns=[c for c in ("name", "year", "player_name", "pitcher_name") if c in df])
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c not in keep})


def _merge_on_player(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return base
    extra = extra.drop_duplicates("player_id")
    return base.merge(extra, how="left", left_on="mlbam_id", right_on="player_id").drop(columns="player_id")


def hitter_season(year: int, refresh: bool) -> pd.DataFrame:
    base = mlbapi.season_stats(year, "hitting", refresh=refresh)
    if base.empty:
        return base
    cust = savant.custom(year, "batter", refresh=refresh)
    cust = cust.drop(columns=[c for c in ("name", "year", "pa", "ab") if c in cust])
    df = _merge_on_player(base, cust.rename(columns=lambda c: c if c == "player_id" else f"sv_{c}"))
    df = _merge_on_player(df, _prefixed(savant.exit_velo(year, "batter", refresh=refresh), "ev_"))
    df = _merge_on_player(df, _prefixed(savant.bat_tracking(year, refresh=refresh), "bt_"))
    df = _merge_on_player(df, _prefixed(savant.swing_path(year, refresh=refresh), "sw_"))
    if year >= 2026:
        timing = savant.swing_timing(year, refresh=refresh).drop(columns=["team_name"], errors="ignore")
        df = _merge_on_player(df, _prefixed(timing, "tm_"))
    df = _merge_on_player(df, _prefixed(savant.sprint_speed(year, refresh=refresh), "spd_"))
    return df


def pitcher_season(year: int, refresh: bool) -> pd.DataFrame:
    base = mlbapi.season_stats(year, "pitching", refresh=refresh)
    if base.empty:
        return base
    cust = savant.custom(year, "pitcher", refresh=refresh)
    cust = cust.drop(columns=[c for c in ("name", "year", "pa") if c in cust])
    df = _merge_on_player(base, cust.rename(columns=lambda c: c if c == "player_id" else f"sv_{c}"))
    df = _merge_on_player(df, _prefixed(savant.exit_velo(year, "pitcher", refresh=refresh), "ev_"))
    arm = savant.arm_angles(year, refresh=refresh)
    df = _merge_on_player(df, _prefixed(arm, "arm_"))
    return df


def arsenal_season(year: int, refresh: bool) -> pd.DataFrame:
    df = savant.pitch_arsenal_stats(year, refresh=refresh)
    if not df.empty:
        df["season"] = year
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2015)
    ap.add_argument("--last", type=int, default=CURRENT)
    ap.add_argument("--refresh", action="store_true", help="re-download the current season")
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    hitters, pitchers, arsenals = [], [], []
    for year in range(args.first, args.last + 1):
        refresh = args.refresh and year == CURRENT
        hitters.append(hitter_season(year, refresh))
        pitchers.append(pitcher_season(year, refresh))
        arsenals.append(arsenal_season(year, refresh))
        print(f"{year}: {len(hitters[-1])} hitters, {len(pitchers[-1])} pitchers, "
              f"{len(arsenals[-1])} pitcher-pitch rows", flush=True)

    hit = pd.concat(hitters, ignore_index=True)
    pit = pd.concat(pitchers, ignore_index=True)
    ars = pd.concat(arsenals, ignore_index=True)

    bios = mlbapi.people(pd.concat([hit.mlbam_id, pit.mlbam_id]).unique().tolist())
    bios.to_parquet(PROCESSED / "players.parquet", index=False)

    hit.to_parquet(PROCESSED / "hitter_seasons.parquet", index=False)
    pit.to_parquet(PROCESSED / "pitcher_seasons.parquet", index=False)
    ars.to_parquet(PROCESSED / "pitch_arsenal_seasons.parquet", index=False)

    user_timing = ROOT / "data" / "raw" / "user" / "bat-tracking-swing-timing.csv"
    if user_timing.exists():
        t = pd.read_csv(user_timing).rename(columns={"id": "mlbam_id", "year": "season"})
        t.to_parquet(PROCESSED / "swing_timing_by_pitch_group.parquet", index=False)

    bio_cols =["mlbam_id", "birth_date", "bats", "throws", "primary_position", "mlb_debut"]
    for group, label in (("hitting", "hitters"), ("pitching", "pitchers")):
        q = mlbapi.season_stats(CURRENT, group, pool="qualified", refresh=args.refresh)
        q = q.merge(bios[bio_cols], on="mlbam_id", how="left")
        q.to_csv(PROCESSED / f"qualified_{CURRENT}_{label}.csv", index=False)
        print(f"qualified {label} {CURRENT}: {len(q)}")


if __name__ == "__main__":
    main()
