"""Pitch-level Statcast via Savant's search CSV, one day per request.

Savant caps a search at 25k rows, so we pull day by day and write one parquet
per season under data/raw/statcast/<year>.parquet. Days are cached
individually so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

from .http import RAW, cached_text

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
OUT = RAW / "statcast"


def _day(d: date, refresh: bool = False) -> pd.DataFrame:
    params = {
        "all": "true", "hfGT": "R|", "hfSea": f"{d.year}|", "player_type": "pitcher",
        "game_date_gt": d.isoformat(), "game_date_lt": d.isoformat(), "type": "details",
    }
    text = cached_text(OUT / "days" / str(d.year) / f"{d.isoformat()}.csv", URL,
                       params=params, refresh=refresh, pause=1.0)
    if not text.strip() or text.lstrip().startswith("<"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= 25000:
        raise RuntimeError(f"{d} hit the 25k row cap; split the query")
    return df


def season(year: int, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    start = start or date(year, 3, 15)
    end = min(end or date(year, 10, 5), date.today())
    frames = []
    d = start
    while d <= end:
        df = _day(d)
        if len(df):
            frames.append(df)
            print(f"{d}: {len(df):>5} pitches", flush=True)
        d += timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    path = OUT / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


MINORS_URL = "https://baseballsavant.mlb.com/statcast-search-minors/csv"


def minors_season(year: int, level: str = "AAA") -> pd.DataFrame:
    """Minor-league Statcast (Savant has AAA and Single-A from 2025), one request per day."""
    out_dir = RAW / "statcast_minors"
    frames = []
    d = date(year, 3, 25)
    end = min(date(year, 9, 30), date.today())
    while d <= end:
        params = {"all": "true", "minors": "true", "hfSea": f"{year}|", "hfLevel": f"{level}|",
                  "player_type": "batter", "game_date_gt": d.isoformat(), "game_date_lt": d.isoformat(),
                  "type": "details"}
        text = cached_text(out_dir / "days" / f"{level}_{d.isoformat()}.csv", MINORS_URL, params=params, pause=1.0)
        if text.strip() and not text.lstrip().startswith("<"):
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            if len(df):
                frames.append(df)
                print(f"{level} {d}: {len(df):>5}", flush=True)
        d += timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_dir / f"{level}_{year}.parquet", index=False)
    return out
