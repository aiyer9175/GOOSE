"""Download the full-season Statcast CSVs from the shared Google Drive folder and
convert each to parquet under data/raw/statcast_drive/<year>.parquet.

Usage: python scripts/pull_drive_statcast.py [--years 2015 2016 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "statcast_drive"
FOLDER = "https://drive.google.com/drive/folders/1ssRW2_AuZOMLLxiTcoS3Wd-heb-ea84S"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
    for f in sorted(files, key=lambda f: f.path):
        year = int(Path(f.path).stem.split("_")[-1])
        if args.years and year not in args.years:
            continue
        csv, pq = OUT / f.path, OUT / f"{year}.parquet"
        if pq.exists():
            print(f"{year}: parquet exists, skipping", flush=True)
            continue
        if not csv.exists():
            print(f"{year}: downloading", flush=True)
            gdown.download(id=f.id, output=str(csv), quiet=True)
        df = pd.read_csv(csv, low_memory=False)
        df.to_parquet(pq, index=False)
        print(f"{year}: {len(df):,} rows, {csv.stat().st_size / 1e9:.2f} GB csv -> "
              f"{pq.stat().st_size / 1e9:.2f} GB parquet", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
