"""Prospect pedigree from FanGraphs Board exports (data/raw/fangraphs/board/board_<year>.csv).

Each list is treated as known before season <year>. Features for season t use lists <= t:
peak FV, most recent FV, best Top-100 rank. FanGraphs ids are mapped to MLBAM through the
Chadwick register; minor leaguers with 'sa' ids fall back to name + birth year (+/- 1).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from goose.data.http import ROOT

BOARD = ROOT / "data" / "raw" / "fangraphs" / "board"
PROCESSED = ROOT / "data" / "processed"


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for suf in (" jr.", " jr", " sr.", " ii", " iii", " iv"):
        s = s.replace(suf, "")
    return "".join(ch for ch in s if ch.isalpha() or ch == " ").strip()


def _fv(v) -> float:
    s = str(v).strip()
    return float(s.rstrip("+")) + (2.5 if s.endswith("+") else 0.0) if s and s != "nan" else np.nan


def load_board() -> pd.DataFrame:
    frames = []
    for path in sorted(BOARD.glob("board_*.csv")):
        year = int(path.stem.split("_")[1])
        d = pd.read_csv(path, usecols=["Top 100", "Org Rk", "Name", "Org", "Pos", "FV", "Age", "PlayerId"])
        d["list_year"] = year
        frames.append(d)
    b = pd.concat(frames, ignore_index=True)
    b["fv"] = b.FV.map(_fv)
    b["top100"] = pd.to_numeric(b["Top 100"], errors="coerce")
    b["birth_year"] = (b.list_year - pd.to_numeric(b.Age, errors="coerce")).round()
    return b


def match_ids(b: pd.DataFrame) -> pd.DataFrame:
    fg = pd.read_parquet(PROCESSED / "chadwick_fg_mlbam.parquet")
    fg_map = dict(zip(fg.key_fangraphs.astype(str), fg.key_mlbam))
    b = b.copy()
    b["mlbam_id"] = b.PlayerId.astype(str).map(fg_map)

    bios = pd.read_parquet(PROCESSED / "players.parquet").drop_duplicates("mlbam_id")
    bios["key"] = bios.name.map(_norm)
    bios["by"] = pd.to_datetime(bios.birth_date, errors="coerce").dt.year
    by_name = bios.groupby("key")
    miss = b.mlbam_id.isna()
    found = []
    for i, row in b[miss].iterrows():
        k = _norm(row.Name)
        if k not in by_name.groups:
            found.append(np.nan)
            continue
        cand = by_name.get_group(k)
        if np.isfinite(row.birth_year):
            cand = cand[(cand.by - row.birth_year).abs() <= 1]
        found.append(cand.mlbam_id.iloc[0] if len(cand) == 1 else np.nan)
    b.loc[miss, "mlbam_id"] = found
    return b


def pedigree_by_season(seasons=range(2015, 2028)) -> pd.DataFrame:
    """Per (mlbam_id, season t): pedigree known by the end of season t (lists with year <= t)."""
    b = match_ids(load_board()).dropna(subset=["mlbam_id"])
    b["mlbam_id"] = b.mlbam_id.astype(int)
    rows = []
    for t in seasons:
        k = b[b.list_year <= t].sort_values("list_year")
        if k.empty:
            continue
        g = k.groupby("mlbam_id")
        out = pd.DataFrame({"fv_peak": g.fv.max(), "fv_last": g.fv.last(), "best_rank": g.top100.min(),
                            "years_listed": g.list_year.nunique(), "last_listed": g.list_year.max()})
        out["season"] = t
        rows.append(out.reset_index())
    ped = pd.concat(rows, ignore_index=True)
    ped["ever_top100"] = ped.best_rank.notna().astype(int)
    return ped


if __name__ == "__main__":
    b = match_ids(load_board())
    print("match rate by list year:\n", b.groupby("list_year").mlbam_id.apply(lambda s: s.notna().mean()).round(3))
    print("match rate, Top 100 only:", round(b[b.top100.notna()].mlbam_id.notna().mean(), 3))
    ped = pedigree_by_season()
    ped.to_parquet(PROCESSED / "pedigree.parquet", index=False)
    print(ped[ped.season == 2026].sort_values("fv_peak", ascending=False).head(5).to_string())
